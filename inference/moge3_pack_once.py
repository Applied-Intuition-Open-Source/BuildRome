# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""One image -> one MoGe-3 ViT-G conditioning pack (npz, load_da3_pack format).

Runs under the MoGe-3 environment (its dependency pins may conflict with the
TRELLIS one); inference/geometry.py invokes it as a subprocess. The pack is
built on the center square crop of the image, the same view inference
conditions on.

    <moge3-venv>/bin/python inference/moge3_pack_once.py \
        --image photo.png --out pack.npz [--model-path model.pt]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOGE3_MODEL_ID = "Ruicheng/moge-3-vitg"
IMAGE_SIZE = 512  # inference runs on the 512^2 conditioning view; match it


@torch.no_grad()
def build_moge_pack(model, image: Image.Image, device: str, anchor_height: bool = False) -> dict:
    """One view's MoGe prediction -> the npz-ready pack dict.

    MoGe runs on the 512^2 LANCZOS-resized view, the floor fit uses MoGe's own
    predicted points + normals, normalized intrinsics are scaled to pixel
    units, and invalid-mask pixels play the sky role.
    """
    from build_rome.modules.lifting import DA3_PACK_VERSION, camera_to_yaw_transform, fit_floor_plane

    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    t = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float().to(device) / 255.0
    pred = model.infer(t)
    up, cam_h, inlier_frac, ok = fit_floor_plane(
        pred["points"].float().reshape(-1, 3), pred["normal"].float().reshape(-1, 3),
        anchor_height=anchor_height,
    )
    mask = pred["mask"].bool()
    depth = (torch.nan_to_num(pred["depth"].float(), nan=0.0, posinf=0.0) * mask).cpu().numpy()
    H, W = depth.shape
    K = pred["intrinsics"].float().clone()
    K[0] *= W  # normalized -> pixel intrinsics
    K[1] *= H
    return {
        "depth": depth.astype(np.float16),
        "conf": np.ones((H, W), np.float16),  # MoGe has no confidence map
        "sky": (~mask).cpu().numpy(),
        "K": K.cpu().numpy().astype(np.float32),
        "cam_to_yaw": camera_to_yaw_transform(up, cam_h).cpu().numpy().astype(np.float32),
        "camera_height": np.float32(cam_h),
        "floor_ok": np.bool_(ok),
        "floor_inlier_frac": np.float32(inlier_frac),
        "ground_mode": "anchor" if anchor_height else "cascade",
        "version": np.int32(DA3_PACK_VERSION),
        "model": MOGE3_MODEL_ID,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default=None,
                    help=f"local model.pt (default: HF {MOGE3_MODEL_ID})")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ground", choices=["anchor", "cascade"], default="cascade",
                    help="cascade (default): height-histogram mode + plane fit, the 2nd-percentile "
                         "anchor as fallback when the fit fails; anchor: floor at the 2nd percentile of "
                         "point heights along the normal-consensus up (sinks below reflections)")
    args = ap.parse_args()

    from moge.model.v3 import MoGeModel

    model = MoGeModel.from_pretrained(args.model_path or MOGE3_MODEL_ID)
    model = model.to(args.device).eval()
    # Same view the model conditions on (inference load_image): EXIF-orient +
    # CENTER SQUARE CROP.
    image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    w, h = image.size
    if w != h:
        s = min(w, h)
        image = image.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    pack = build_moge_pack(model, image, args.device, anchor_height=(args.ground == "anchor"))
    tmp = args.out + ".tmp.npz"
    np.savez_compressed(tmp, **pack)
    os.replace(tmp, args.out)
    print(f"[moge3-pack] {args.out}: floor_ok={bool(pack['floor_ok'])} "
          f"camera_height={float(pack['camera_height']):.2f} m")


if __name__ == "__main__":
    main()
