# Installation and GPU smoke test

The source snapshot dates to February 2024. The following is a **candidate compatible stack**, not a GPU environment tested during this repair. CPU checks ran separately using the available runtime in VALIDATION.md. Do not mix these application versions with the unrelated Tensor Holography 2021 environment.

## Candidate environment

Linux, NVIDIA GPU, CUDA toolkit 11.8 (including nvcc), C++ build tools and Git are required for native extensions. Driver/toolkit compatibility must match your GPU. No minimum VRAM was measured.

```bash
conda create -n my-threestudio python=3.10
conda activate my-threestudio
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install xformers==0.0.23.post1
python -m pip install -r requirements-integration.txt

git clone https://github.com/ashawkey/diff-gaussian-rasterization.git extern/diff-gaussian-rasterization
git -C extern/diff-gaussian-rasterization checkout d986da0d4cf2dfeb43b9a379b6e9fa0a7f3f7eea
git -C extern/diff-gaussian-rasterization submodule update --init --recursive
python -m pip install --no-build-isolation ./extern/diff-gaussian-rasterization

git clone https://github.com/DSaurus/simple-knn.git extern/simple-knn
git -C extern/simple-knn checkout 8e18d9b2ec09f932b6208172614544267b067a7c
python -m pip install --no-build-isolation ./extern/simple-knn
python scripts/preflight.py --strict
```

`requirements-integration.txt` retains the original repository's full dependency surface and supplies missing ImageDream/plyfile plus constraints for key Python libraries. It is **not a complete lockfile**: several native and transitive original dependencies remain unpinned. [Recorded references](dependency-refs.json) select public commits existing by the snapshot date; they do not prove which exact commits were installed on the original workstation. Retain a `pip freeze` and the extension commits after a successful build.

## Weights and configuration

Use MVDream/ImageDream weights from their official projects under their terms. ImageDream expects `sd-v2.1-base-4view-ipmv.pt` and the matching `sd_v2_base_ipmv.yaml`. A checkout for the YAML is optional; the installed imagedream package also includes configs.

```bash
git clone https://github.com/bytedance/ImageDream.git extern/ImageDream
git -C extern/ImageDream checkout 26c3972e586f0c8d2f6c6b297aa9d792d06abebb
```

The matching config is `extern/ImageDream/extern/ImageDream/imagedream/configs/sd_v2_base_ipmv.yaml`. Supply the downloaded checkpoint path separately; this package does not include it. Only load trusted checkpoints.

## Run the integration

```bash
python scripts/run_pipeline.py --pipeline mvdream-3dgs --prompt "a ceramic corgi" --smoke
python scripts/run_pipeline.py --pipeline imagedream-3dgs --prompt "a ceramic corgi" --image /path/reference.png --checkpoint /path/sd-v2.1-base-4view-ipmv.pt --model-config extern/ImageDream/extern/ImageDream/imagedream/configs/sd_v2_base_ipmv.yaml --smoke
```

`--smoke` runs two real training steps, disables validation and does not use fake model backends. It does not cover densification thresholds at later steps. Run longer with the original schedule to exercise topology changes. Remove `--smoke` for the configured experiment. Use one GPU and `precision=32-true` for the Gaussian system.

The registry currently eagerly imports native and model modules, including ImageDream and Gaussian extensions even for some other routes. The preflight tool detects missing modules and attempts an isolated full registry import when prerequisites exist. It does not guarantee rasterizer ABI compatibility or model-weight availability.

## Typical failures

- Four-output ABI error: use the recorded ashawkey rasterizer commit and rebuild against this PyTorch/CUDA environment.
- Missing reference image: pass `--image`; RGB and RGBA files are accepted by guidance.
- Incomplete view groups: keep batch size divisible by `n_view=4` for multi-view guidance.
- Non-finite loss/OOM: preserve error output and environment report; no automatic claim of a corrected training recipe is made.
- CPU test conflict with ROS `launch`: use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` as documented; the repository's `launch.py` otherwise collides with ROS pytest plugins.
