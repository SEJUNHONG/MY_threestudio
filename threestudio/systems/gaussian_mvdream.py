import os
from dataclasses import dataclass, field

import numpy as np
import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *
from threestudio.utils.integration import horizontal_fov
from torch.cuda.amp import autocast

from threestudio.models.geometry.gaussian_base import BasicPointCloud, Camera


@threestudio.register("gaussian-splatting-mvdream-system")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)
        image: bool = False
        refinement: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False

        if self.cfg.refinement:
            raise ValueError("Gaussian refinement has no mesh output; use the NeRF/DMTet refine configuration")
        self.register_buffer("background_tensor", torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device=self.device
        ), persistent=False)

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        if self.cfg.image and self.prompt_utils.image is None:
            raise ValueError("ImageDream requires system.prompt_processor.image_path")

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            self.optim_num = 2
            return [optim, net_optim]
        self.optim_num = 1
        return [optim]

    def on_load_checkpoint(self, checkpoint):
        num_pts = checkpoint["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        for name in ("max_radii2D", "xyz_gradient_accum", "denom"):
            checkpoint["state_dict"].setdefault("geometry." + name, getattr(self.geometry, name).clone())
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        lr_max_step = self.geometry.cfg.position_lr_max_steps
        scale_lr_max_steps = self.geometry.cfg.scale_lr_max_steps

        if self.global_step < lr_max_step:
            self.geometry.update_xyz_learning_rate(self.global_step)

        if self.global_step < scale_lr_max_steps:
            self.geometry.update_scale_learning_rate(self.global_step)

        bs = batch["c2w"].shape[0]
        # One call preserves share_aug_bg across the complete multi-view batch.
        all_backgrounds = self.background(dirs=torch.nn.functional.normalize(batch["rays_d"], dim=-1))
        renders = []
        comp_rgb_bgs = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        normals = []
        depths = []
        for batch_idx in range(bs):
            view_batch = dict(batch, batch_idx=batch_idx, view_background=all_backgrounds[batch_idx:batch_idx + 1])
            fovy = batch["fovy"][batch_idx]
            fovx = horizontal_fov(fovy, batch["width"], batch["height"])
            w2c, proj, cam_p = get_cam_info_gaussian(
                c2w=batch["c2w"][batch_idx], fovx=fovx, fovy=fovy, znear=0.1, zfar=100
            )

            # import pdb; pdb.set_trace()
            viewpoint_cam = Camera(
                FoVx=fovx,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
            )

            with autocast(enabled=False):
                render_pkg = self.renderer(
                    viewpoint_cam, self.background_tensor, **view_batch
                )
                renders.append(render_pkg["render"])
                comp_rgb_bgs.append(render_pkg["comp_rgb_bg"])
                viewspace_points.append(render_pkg["viewspace_points"])
                visibility_filters.append(render_pkg["visibility_filter"])
                radiis.append(render_pkg["radii"])
                if render_pkg.__contains__("normal"):
                    normals.append(render_pkg["normal"])
                if render_pkg.__contains__("depth"):
                    depths.append(render_pkg["depth"])

        outputs = {
            "comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1),
            "comp_rgb_bg": torch.cat(comp_rgb_bgs, dim=0),
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radiis,
        }
        if len(normals) > 0:
            outputs.update(
                {
                    "comp_normal": torch.stack(normals, dim=0).permute(0, 2, 3, 1),
                    #"comp_depth": torch.stack(depths, dim=0).permute(0, 2, 3, 1),
                }
            )
        if depths:
            outputs["comp_depth"] = torch.stack(depths).permute(0, 2, 3, 1)
        return outputs

    def training_step(self, batch, batch_idx):
        if self.optim_num == 1:
            opt = self.optimizers()
        else:
            opt, net_opt = self.optimizers()
        if self.trainer.precision not in (32, "32", "32-true"):
            raise ValueError("Gaussian manual optimization currently requires precision=32-true")
        opt.zero_grad(set_to_none=True)
        if self.optim_num > 1:
            net_opt.zero_grad(set_to_none=True)
        out = self(batch)

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        # import pdb; pdb.set_trace()
        viewspace_point_tensor = out["viewspace_points"]
        if not self.cfg.image:
            guidance_out = self.guidance(
                guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False
            )
        else:
            guidance_out = self.guidance(
                guidance_inp, self.prompt_utils, comp_rgb_bg=out["comp_rgb_bg"], **batch, rgb_as_latents=False
            )

        loss_sds = 0.0
        loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )
        xyz_mean = None
        if not self.cfg.refinement:
            if self.cfg.loss["lambda_position"] > 0.0:
                xyz_mean = self.geometry.get_xyz.norm(dim=-1)
                loss_position = xyz_mean.mean()
                self.log(f"train/loss_position", loss_position)
                loss += self.C(self.cfg.loss["lambda_position"]) * loss_position

            if self.cfg.loss["lambda_opacity"] > 0.0:
                scaling = self.geometry.get_scaling.norm(dim=-1)
                loss_opacity = (
                    scaling.detach().unsqueeze(-1) / self.geometry.get_opacity
                ).sum()
                self.log(f"train/loss_opacity", loss_opacity)
                loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

            if self.cfg.loss["lambda_scales"] > 0.0:
                scale_sum = torch.sum(self.geometry.get_scaling)
                self.log(f"train/scales", scale_sum)
                loss += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

            if self.cfg.loss["lambda_tv_loss"] > 0.0:
                loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
                    out["comp_rgb"].permute(0, 3, 1, 2)
                )
                self.log(f"train/loss_tv", loss_tv)
                loss += loss_tv

        else:
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log(f"train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(self.cfg.loss["lambda_normal_consistency"])
        #if (
        #    out.__contains__("comp_depth")
        #    and self.cfg.loss["lambda_depth_tv_loss"] > 0.0
        #):
        #    loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
        #        tv_loss(out["comp_normal"].permute(0, 3, 1, 2))
        #        + tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
        #    )
        #    self.log(f"train/loss_depth_tv", loss_depth_tv)
        #    loss += loss_depth_tv

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Gaussian training loss")
        self.manual_backward(loss)
        iteration = self.global_step
        # Step existing parameters BEFORE densification replaces Parameter objects.
        opt.step()
        if self.optim_num > 1:
            # Exactly one Lightning step count per rendered batch. FP32 only.
            net_opt.optimizer.step()
        self.geometry.update_states(iteration, visibility_filter, radii, viewspace_point_tensor)
        opt.zero_grad(set_to_none=True)
        if self.optim_num > 1:
            net_opt.zero_grad(set_to_none=True)

        return {"loss": loss} # _sds}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.global_step,
        )
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            r"(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
