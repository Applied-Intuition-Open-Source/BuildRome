# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Model construction and constants for single-image scene inference.

The SS and SLat denoisers are TRELLIS.2 flow transformers with LoRA adapters
plus two geometric conditioning branches driven by the monocular depth pack:
the depth lift (DINO features splatted onto visible voxels) and, for SS, the
signed-clearance branch. Absolute position never enters the model; windows
only see geometry through these branches.
"""
import numpy as np
import torch
from PIL import Image, ImageOps

SS_DEC = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
BASE_WEIGHTS = {"ss": "microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16",
                "slat": "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16"}
SLAT_DEC = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
DINO = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# Shape-latent standardization of the TRELLIS.2 img2shape latents.
SLAT_MEAN = [0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
             -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
             0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
             -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665]
SLAT_STD = [5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
            5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
            4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
            5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691]

# TRELLIS.2-4B sampler recipe (pipeline.json): warped t-schedule, CFG 7.5
# rescaled, applied only in the high-noise interval.
SAMPLER_PARAMS = {
    "ss": dict(steps=12, guidance_strength=7.5, guidance_rescale=0.7,
               guidance_interval=(0.6, 1.0), rescale_t=5.0),
    "slat": dict(steps=12, guidance_strength=7.5, guidance_rescale=0.5,
                 guidance_interval=(0.6, 1.0), rescale_t=3.0),
}

_BASE_ARGS = dict(
    model_channels=1536, cond_channels=1024, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
    pe_mode="rope", share_mod=True, initialization="scaled",
    qk_rms_norm=True, qk_rms_norm_cross=True, dtype="bfloat16",
)
SS_MODEL_ARGS = dict(_BASE_ARGS, resolution=16, in_channels=8, out_channels=8, depth_lift_tau_m=0.28)
SLAT_MODEL_ARGS = dict(_BASE_ARGS, resolution=64, in_channels=32, out_channels=32,
                       use_checkpoint=False, depth_lift_tau_m=0.07)


def _load_hf_state(path: str) -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    parts = path.split("/")
    return load_file(hf_hub_download(f"{parts[0]}/{parts[1]}", "/".join(parts[2:]) + ".safetensors"))


def _denoiser_state(ckpt_path: str) -> dict:
    """Denoiser weights from a Lightning .ckpt (``denoiser.*`` keys)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {k[len("denoiser."):]: v for k, v in ck["state_dict"].items() if k.startswith("denoiser.")}


def _build(name: str, args: dict, kind: str, ckpt_path: str, device):
    """Build a denoiser from either a Lightning .ckpt (full denoiser state) or
    a released .safetensors delta (LoRA + branches) over the public base."""
    from easydict import EasyDict as edict
    from safetensors.torch import load_file

    from build_rome.utils.model_wrapper_utils import build_single_model

    if ckpt_path.endswith(".safetensors"):
        sd = {**_load_hf_state(BASE_WEIGHTS[kind]), **load_file(ckpt_path)}
    else:
        sd = _denoiser_state(ckpt_path)
    rank = next(v.shape[0] for k, v in sd.items() if k.endswith("lora_A"))
    model = build_single_model(edict({"name": name, "args": args,
                                      "lora": {"r": int(rank), "alpha": 2 * int(rank)}}))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    buffers = {n for n, _ in model.named_buffers()}
    missing = [k for k in missing if k not in buffers]
    assert not missing and not unexpected, f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}"
    print(f"[{name}] loaded {ckpt_path}")
    return model.to(device).eval()


def build_ss_model(ckpt_path: str, device):
    return _build("LiftedSSFlowModel", SS_MODEL_ARGS, "ss", ckpt_path, device)


def build_slat_model(ckpt_path: str, device):
    return _build("LiftedSLatFlowModel", SLAT_MODEL_ARGS, "slat", ckpt_path, device)


def load_image(path: str, size: int = 512) -> torch.Tensor:
    """EXIF-orient, center square crop, resize to size^2 -> [3, S, S] in [0, 1]."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    if w != h:
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
