# Experimental multi-view fine-tuning

This script was added during the repair. It is **cross-attention adaptation of pretrained ImageDream**, not a reproduced historical experiment, not full multi-view training of vanilla SD, and not a verified quality improvement.

`finetune_multiview.py` freezes VAE/CLIP and all model parameters except UNet names matching `.attn2.`. It minimizes noise-prediction MSE over four target views, with camera context, text context and reference image tokens/latent. The extra fifth reference slot is excluded from the loss. A shared timestep is used within each object group. Text/image condition dropout is joint (default 0.1), and camera conditions remain active.

## JSONL data contract

One JSON object per line, relative image paths resolved from the manifest directory:

```json
{"prompt":"a ceramic corgi","reference":"object01/reference.png","views":[{"image":"object01/front.png","c2w":[[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]]},{"image":"object01/right.png","c2w":[[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]]},{"image":"object01/back.png","c2w":[[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]]},{"image":"object01/left.png","c2w":[[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]]}]}
```

The repeated matrices above are **schema placeholders, not valid four-view training labels**. Replace each with the corresponding real, calibrated Blender camera-to-world transform and keep the view ordering consistent. The loader checks matrix shape, homogeneous row, finite values and proper orthonormal rotation, but cannot infer correct camera axes, framing, object normalization or whether labels match the images. Target groups must show the same object; split by object identity, not by individual image.

Images are resized to 256×256 and RGBA is composited over white. Prepare square images and consistent object framing beforehand; do not silently distort training data. No dataset is bundled.

```bash
python scripts/finetune_multiview.py --manifest data/train.jsonl --checkpoint /path/base.pt --model-config /path/sd_v2_base_ipmv.yaml --validate-data-only
python scripts/finetune_multiview.py --manifest data/train.jsonl --checkpoint /path/base.pt --model-config /path/sd_v2_base_ipmv.yaml --steps 1000 --lr 1e-5 --output outputs/adaptation
```

The training run requires CUDA and the dependency environment. It uses FP32, one object group per optimizer step, gradient clipping and no automatic resume. At the end it writes `model.pt` (full state dictionary for ImageDream `build_model`), optimizer state, config and training metrics. Full model checkpoints may be large. It does not provide held-out diffusion validation or model selection; use separate held-out objects and the experiment protocol before claiming gains.

Feed the result into either representation:

```bash
python scripts/run_pipeline.py --pipeline imagedream-3dgs --prompt "a ceramic corgi" --image /path/reference.png --checkpoint outputs/adaptation/model.pt --model-config /path/sd_v2_base_ipmv.yaml --smoke
```

Only the epsilon-loss routing has been tested with a tiny fake denoiser. Loading real ImageDream weights, selecting all intended attention modules, CUDA memory behavior, the actual training run and checkpoint reload into real guidance remain unverified.
