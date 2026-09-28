"""Report missing runtime components before allocating diffusion weights."""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/preflight.json")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    modules = ["torch", "pytorch_lightning", "omegaconf", "diffusers", "transformers", "mvdream", "imagedream", "diff_gaussian_rasterization", "simple_knn", "tinycudann", "nvdiffrast", "plyfile"]
    report = {"python": sys.version, "modules": {n: importlib.util.find_spec(n) is not None for n in modules}, "nvcc": shutil.which("nvcc"), "cuda_available": False}
    if report["modules"]["torch"]:
        import torch
        report.update(torch=str(torch.__version__), torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    # Isolated process: importing the registry can load all optional native modules.
    if all(report["modules"].values()) and report["cuda_available"]:
        result = subprocess.run([sys.executable, "-c", "import threestudio"], capture_output=True, text=True)
        report["registry_import_ok"] = result.returncode == 0
        report["registry_error"] = result.stderr[-4000:]
    else:
        report["registry_import_ok"] = False
    report["ready"] = all(report["modules"].values()) and report["cuda_available"] and report["registry_import_ok"]
    report["scope"] = "Import readiness only; rasterizer ABI and training require a real GPU smoke run"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.strict and not report["ready"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
