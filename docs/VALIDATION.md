# Validation record

Repair validated 2026-09-28 against base `c894bfcaba28378aee5b7ec09e50f9b2b2516c6a`.

- Python 3.10.12, PyTorch 2.8.0+cu128; CUDA unavailable.
- **18 CPU tests passed** using `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`.
- Modified/new Python source syntax checked; `git diff --check` passed.
- Common launcher dry-run generated the MVDream-3DGS two-step command successfully.
- Preflight executed and correctly reported **not ready**: diffusion libraries and CUDA extensions absent, no GPU.

Tests isolate the actual method bodies through AST loading to avoid the repository's eager native imports. Rasterizers, denoisers and Lightning wrappers are lightweight substitutes. This allows meaningful shape, gradient and optimizer-order regressions but does **not** test extension compilation, actual CUDA derivatives, complete registry/config instantiation or actual Lightning Trainer lifecycle. No model weights were downloaded or used.

Covered: per-view renderer background shapes and gradients for both rasterizers; all-view system aggregation; rectangular FoV and camera CPU placement; optimizer-before-densification ordering with one counter increment; ImageDream reference-view ordering; both SDS branches' gradient flow and alpha schedule initialization; legacy checkpoint statistic migration; fine-tuning epsilon-loss gradient; invalid camera manifest; complete-view config wiring; launcher input validation.

Not validated: target candidate PyTorch 2.1.2 environment, real multi-view inference, real fine-tuning/checkpoint reload, Gaussian topology mutation with CUDA optimizer state, optical/rendered depth correctness, NeRF/DMTet refinement, DDP, AMP, long-run stability, speedup, view consistency or final 3D quality. The original pre-commit workflow is retained; the new focused CPU workflow has not run on GitHub.

Next: build the documented environment, run preflight with `--strict`, run both representation routes with `--smoke`, then a longer run through densification and checkpoint resume. Retain failed logs as well as successful results. See [experiment protocol](EXPERIMENTS.md) before reporting performance.

[Machine-readable report](validation-report.json).
