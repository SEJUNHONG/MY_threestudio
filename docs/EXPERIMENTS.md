# Experiment protocol

Do not fill a results table from paper benchmarks. Run and retain your own artifacts.

1. Save `scripts/preflight.py` output, `pip freeze`, Git commit, rasterizer SHA, GPU/driver, resolution, seed and model checkpoint identifier/hash.
2. Run each route with `--smoke`; retain loss logs and check gradients are finite.
3. Run longer through at least one densification/pruning event. Record Gaussian point count, optimizer steps and successful checkpoint save/resume.
4. Evaluate SD-NeRF, MVDream-NeRF, MVDream-3DGS, ImageDream-NeRF and ImageDream-3DGS using matched prompts and seeds. Keep image conditioning identical where applicable.
5. For adaptation, split by object identity; compare base and adapted ImageDream using the same renderer, prompts, reference images, camera trajectories and seeds. Training epsilon MSE is not itself a view-consistency metric.
6. Separate renderer-only latency from end-to-end optimization time. CUDA-synchronize before/after timing; report warmup, number of samples, median/p95, peak memory, resolution, precision and point/sample count.
7. Save turntable RGB/depth/normal frames and PLY results. Visual inspection should identify Janus artifacts, floaters and geometry/image consistency failures; do not substitute selected images for aggregate measurements.

Use `docs/experiment-template.json` per run. Null values mean not measured. No speedup or quality result is supplied by this template.
