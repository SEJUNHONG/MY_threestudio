# Integration repair audit

Base: SEJUNHONG/MY_threestudio `c894bfcaba28378aee5b7ec09e50f9b2b2516c6a`.
Repair date: 2026-09-28. All findings below refer to this snapshot.

| Finding | Repair | CPU evidence |
|---|---|---|
| Renderer interpreted CHW channel count 3 as batch size and reshaped backgrounds incorrectly | Preserve `[1,H,W,3]` per view | Execute both renderer forwards with fake rasterizer |
| System kept only the final view's background | Concatenate all backgrounds into BHWC | Actual forward with four distinct view backgrounds |
| Unshaded renderer trained a neural background that did not contribute to RGB | Rasterize over black, alpha-compose neural background | Background gradient is nonzero |
| Per-view background calls defeated shared augmentation | One full-batch background call, normalized directions | Forward routing check |
| Raw `loss.backward(retain_graph=True)` bypassed Lightning interface | `manual_backward`, no unnecessary graph retention | Actual manual training_step test |
| Densification replaced Parameters before optimizer step | Step Gaussian/background optimizers, then update topology | Ordered event regression |
| Both Lightning optimizer wrappers advanced global step | Count the Gaussian step only; auxiliary FP32 optimizer uses underlying step | One-count test; real Lightning trainer still requires GPU verification |
| Gaussian accumulators were ordinary tensors | Registered persistent buffers; legacy loads default missing stats | Buffer/checkpoint regression |
| `alphas_cumprod` missing in both multi-view SDS branches | Clone model schedule into a nonpersistent buffer | SDS branch gradient and configure tests |
| Image conditioning absent or RGB-only input failed downstream RGBA composition | Required image config, explicit error, RGBA conversion | Missing image and mock conditioning tests |
| Background RGB collapsed to a scalar | Reduce B/H/W only | Colored-background regression |
| Added reference-view timesteps appended at batch end | Insert within each view group, matching latents and context | Different timesteps across two groups |
| FoVx equaled FoVy for rectangular renders | Derive FoVx from width/height | Non-square camera test |
| Incompatible rasterizer builds failed with unclear tuple errors | Require four outputs: RGB/radii/depth/alpha | ABI error tests |

The rasterizer reference is ashawkey/diff-gaussian-rasterization `d986da0d4cf2dfeb43b9a379b6e9fa0a7f3f7eea`; its Python source returns exactly those four outputs. This is not interchangeable with every upstream Gaussian rasterizer.

Deliberate behavior changes: the unshaded renderer now uses the neural background. Its old `renderer.invert_bg_prob` field is retained as deprecated for configuration parsing, but no longer controls image color; use `background.random_aug`. Checkpoints and experimental comparisons must note this change.

Scope limits: depth-to-normal mapping inherits the original depth convention and was not validated against a physical scene; Gaussian mesh extraction and NeRF/DMTet refinement were not exercised. Unsupported Gaussian `refinement=true` fails explicitly because no mesh is returned. DDP, AMP and exact stochastic resume are not validated. New buffers preserve densification statistics, not all sources of randomness.

The supplied ImageDream model already implements image conditioning/cross-attention. The new fine-tuner adapts that existing mechanism; no new attention architecture or historical trained model is claimed.
