"""Experimental ImageDream cross-attention adaptation on four-view RGB/camera groups.
No historical training claim. Frozen VAE/CLIP; epsilon loss, FP32, one object/step.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Avoid importing threestudio's eager CUDA registry just for tensor utilities.
spec = importlib.util.spec_from_file_location("integration", Path(__file__).resolve().parents[1] / "threestudio/utils/integration.py")
integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integration)

def read_records(manifest):
    manifest = Path(manifest).resolve()
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("Manifest is empty")
    for record in records:
        if len(record["views"]) != 4 or not record.get("prompt"):
            raise ValueError("Each record needs prompt and exactly four ordered views")
        for view in record["views"]:
            matrix = np.asarray(view["c2w"], dtype=np.float32)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
                raise ValueError("c2w must be a finite homogeneous 4x4 Blender camera-to-world matrix")
            if not np.allclose(matrix[:3,:3].T @ matrix[:3,:3], np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(matrix[:3,:3]), 1, atol=1e-3):
                raise ValueError("c2w rotation must be right-handed and orthonormal")
            view["image"] = str((manifest.parent / view["image"]).resolve())
        record["reference"] = str((manifest.parent / record["reference"]).resolve())
        for path in [record["reference"]] + [v["image"] for v in record["views"]]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    return records

def load_rgb(path):
    with Image.open(path) as image:
        image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(canvas, image).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)

def tensor_image(image, device):
    return torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float().to(device) / 127.5 - 1

def epsilon_loss(model, latents, timesteps, context, noise):
    if model.parameterization != "eps":
        raise ValueError("Only epsilon-prediction ImageDream checkpoints are supported")
    noisy = model.q_sample(latents, timesteps, noise)
    noisy = integration.append_group_view(noisy, 4, zero=True)
    prediction = model.apply_model(noisy, integration.append_group_view(timesteps, 4), context)
    prediction = integration.remove_group_view(prediction, 4)
    if prediction.shape != noise.shape:
        raise ValueError("Prediction and noise shape mismatch")
    return F.mse_loss(prediction.float(), noise.float())

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--output", default="outputs/finetune")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition-dropout", type=float, default=0.1)
    p.add_argument("--validate-data-only", action="store_true")
    args = p.parse_args()
    if args.steps < 1 or args.lr <= 0 or not 0 <= args.condition_dropout <= 1:
        p.error("steps/lr must be positive and condition-dropout must be in [0,1]")
    records = read_records(args.manifest)
    if args.validate_data_only:
        print(json.dumps({"objects": len(records), "views_per_object": 4}))
        return
    if not torch.cuda.is_available():
        p.error("Fine-tuning requires CUDA and installed ImageDream dependencies")
    for path in (args.checkpoint, args.model_config):
        if not Path(path).is_file():
            p.error("Base checkpoint and model config must exist")
    from imagedream.model_zoo import build_model
    from imagedream.camera_utils import normalize_camera
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model("sd-v2.1-base-4view-ipmv", config_path=args.model_config, ckpt_path=args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    selected = []
    for name, parameter in model.named_parameters():
        if name.startswith("model.") and ".attn2." in name:
            parameter.requires_grad_(True)
            selected.append((name, parameter))
    if not selected:
        raise RuntimeError("No cross-attention parameters found; incompatible model")
    optimizer = torch.optim.AdamW([p for _, p in selected], lr=args.lr)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "trainable_names": [n for n, _ in selected], "scope": "cross-attention adaptation of pretrained ImageDream; not SD-to-multiview training from scratch"}, indent=2))
    for step in range(args.steps):
        record = random.choice(records)
        images = torch.stack([tensor_image(load_rgb(v["image"]), "cuda") for v in record["views"]])
        reference = load_rgb(record["reference"])
        with torch.no_grad():
            latents = model.get_first_stage_encoding(model.encode_first_stage(images))
            dropped = random.random() < args.condition_dropout
            text = model.get_learned_conditioning(["" if dropped else record["prompt"]]).repeat(4, 1, 1)
            ip = model.get_learned_image_conditioning(reference).repeat(4, 1, 1)
            ip_image = model.get_first_stage_encoding(model.encode_first_stage(tensor_image(reference, "cuda")[None]))
            if dropped:
                ip = torch.zeros_like(ip)
                ip_image = torch.zeros_like(ip_image)
            cameras = normalize_camera(torch.tensor([v["c2w"] for v in record["views"]], device="cuda", dtype=torch.float32)).flatten(1)
            context = {"context": integration.append_group_view(text, 4), "camera": integration.append_group_view(cameras, 4, zero=True), "ip": integration.append_group_view(ip, 4), "ip_img": ip_image, "num_frames": 5}
            t = torch.randint(0, len(model.alphas_cumprod), (1,), device="cuda").repeat(4)
            noise = torch.randn_like(latents)
        optimizer.zero_grad(set_to_none=True)
        loss = epsilon_loss(model, latents, t, context, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite fine-tuning loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in selected], 1.0)
        optimizer.step()
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step + 1, "epsilon_mse": loss.item()}) + "\n")
        print(step + 1, loss.item(), flush=True)
    # Plain full state_dict is loadable by ImageDream build_model via ckpt_path.
    torch.save(model.state_dict(), out / "model.pt")
    torch.save({"optimizer": optimizer.state_dict(), "steps": args.steps}, out / "optimizer.pt")
    # Automatic resume is intentionally not implemented in this experimental trainer.

if __name__ == "__main__":
    main()
