"""Tensor contracts shared by NeRF/ImageDream/3DGS; no CUDA imports."""
import torch

def validate_view_batch(batch_size, n_view):
    if n_view < 1 or batch_size < 1 or batch_size % n_view:
        raise ValueError("Batch must contain complete contiguous multi-view groups")

def append_group_view(value, n_view, zero=False):
    validate_view_batch(value.shape[0], n_view)
    groups = value.reshape(-1, n_view, *value.shape[1:])
    extra = torch.zeros_like(groups[:, :1]) if zero else groups[:, -1:]
    return torch.cat((groups, extra), dim=1).flatten(0, 1)

def remove_group_view(value, n_view):
    validate_view_batch(value.shape[0], n_view + 1)
    return value.reshape(-1, n_view + 1, *value.shape[1:])[:, :-1].flatten(0, 1)

def background_rgb(background):
    if background is None or background.ndim != 4 or background.shape[-1] != 3:
        raise ValueError("Expected background [B,H,W,3] for image conditioning")
    if not torch.isfinite(background).all():
        raise ValueError("Background contains non-finite values")
    return background.detach().mean(dim=(0, 1, 2)).clamp(0, 1).cpu().numpy() * 255

def unpack_rasterizer(result):
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise RuntimeError("This integration requires the ashawkey rasterizer ABI: RGB, radii, depth, alpha (four tensors). See docs/INTEGRATION.md.")
    rgb, radii, depth, alpha = result
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("Rasterizer RGB must have shape [3,H,W]")
    if depth.shape != (1, *rgb.shape[1:]) or alpha.shape != depth.shape:
        raise ValueError("Rasterizer depth/alpha must have shape [1,H,W]")
    if radii.ndim != 1:
        raise ValueError("Rasterizer radii must have shape [N]")
    return rgb, radii, depth, alpha

def horizontal_fov(fovy, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return 2 * torch.atan(torch.tan(fovy / 2) * (float(width) / float(height)))
