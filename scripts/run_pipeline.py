"""One entry point for existing SDS, MVDream and ImageDream configurations."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "sd-nerf": "dreamfusion-sd.yaml",
    "mvdream-nerf": "mvdream-nerf-coarse.yaml",
    "mvdream-3dgs": "mvdream-3dgs-coarse.yaml",
    "imagedream-nerf": "imagedream-nerf-coarse.yaml",
    "imagedream-3dgs": "imagedream-3dgs-coarse.yaml",
}

def build_command(args):
    command = [sys.executable, str(ROOT / "launch.py"), "--config", str(ROOT / "configs" / CONFIGS[args.pipeline]), "--train", "system.prompt_processor.prompt=" + json.dumps(args.prompt)]
    if args.pipeline.startswith("imagedream"):
        for key in ("image", "checkpoint", "model_config"):
            path = getattr(args, key)
            if path is None or not Path(path).is_file():
                raise ValueError("ImageDream requires existing --image, --checkpoint and --model-config paths")
        command += ["system.prompt_processor.image_path=" + json.dumps(str(Path(args.image).resolve())), "system.guidance.ckpt_path=" + json.dumps(str(Path(args.checkpoint).resolve())), "system.guidance.config_path=" + json.dumps(str(Path(args.model_config).resolve()))]
    elif args.image or args.model_config:
        raise ValueError("--image and --model-config are only supported by ImageDream pipelines")
    elif args.checkpoint:
        if args.pipeline == "sd-nerf":
            raise ValueError("SD uses its configured pretrained model; --checkpoint here is for MVDream/ImageDream weights")
        if not Path(args.checkpoint).is_file():
            raise ValueError("Checkpoint does not exist")
        command += ["system.guidance.ckpt_path=" + json.dumps(str(Path(args.checkpoint).resolve()))]
    if args.smoke:
        command += ["trainer.max_steps=2", "trainer.limit_val_batches=0", "trainer.num_sanity_val_steps=0"]
    return command + args.overrides

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline", choices=CONFIGS, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--image")
    p.add_argument("--checkpoint")
    p.add_argument("--model-config")
    p.add_argument("--smoke", action="store_true", help="Two real GPU training steps, not a mock")
    p.add_argument("--dry-run", action="store_true", help="Print argument vector without importing CUDA dependencies")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    try:
        command = build_command(args)
    except ValueError as error:
        p.error(str(error))
    print(json.dumps(command, indent=2), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
