#!/usr/bin/env python3
# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Single image -> metric 3D scene mesh (command line front end of
``inference.pipeline.ScenePipeline``).

Writes <out>/<stem>/{scene.glb, layout_bands.json, layout_topview.png,
spiral.mp4}. scene.glb is metric, glTF Y-up, camera at the origin looking
along -Z. See ``inference/pipeline.py`` for the depth-band scheme.

    python inference/image_to_scene.py --image photo.jpg \
        --ss-ckpt ss_delta.safetensors --slat-ckpt slat_delta.safetensors --out results/

Python:

    from build_rome import ScenePipeline
    pipe = ScenePipeline.from_pretrained("ss_delta.safetensors", "slat_delta.safetensors")
    pipe("photo.jpg").save("results/photo")
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.pipeline import ScenePipeline  # noqa: E402


def run(args) -> None:
    pipe = ScenePipeline.from_pretrained(args.ss_ckpt, args.slat_ckpt, moge3_python=args.moge3_python,
                                         moge3_cache=args.moge3_cache)
    scene = pipe(args.image, seed=args.seed, depth_pack=args.depth_pack,
                 band_near_m=args.band_near_m, band_growth=args.band_growth, band_p=args.band_p,
                 band_min_coverage=args.band_min_coverage, band_count=args.band_count,
                 band_min_depth_m=args.band_min_depth_m, band_overlap_frac=args.band_overlap_frac,
                 band_simplify=args.band_simplify, cross_band=args.cross_band,
                 max_inflated_voxels=args.max_inflated_voxels)
    scene.save(os.path.join(args.out, scene.stem), spiral_video=args.spiral_video)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--ss-ckpt", required=True, help="SS denoiser checkpoint (.safetensors delta or .ckpt)")
    ap.add_argument("--slat-ckpt", required=True, help="SLat denoiser checkpoint (.safetensors delta or .ckpt)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth-pack", default=None,
                    help="precomputed depth pack (.npz); default: MoGe-3 on the input image")
    ap.add_argument("--moge3-python", default=sys.executable,
                    help="python of an environment with MoGe-3 installed")
    ap.add_argument("--moge3-cache", default=os.path.expanduser("~/.cache/image_to_scene/moge3"),
                    help="MoGe-3 pack cache (keyed by image content)")
    ap.add_argument("--band-near-m", type=float, default=9.0,
                    help="native (1x) windows for window centers up to this depth")
    ap.add_argument("--band-growth", type=float, default=2.0, help="window-size multiplier per further band")
    ap.add_argument("--band-p", type=float, default=0.99, help="cloud depth quantile the last band must reach")
    ap.add_argument("--band-min-coverage", type=float, default=0.05,
                    help="drop far bands covering less than this fraction of the image")
    ap.add_argument("--band-count", type=int, default=None, help="force exactly this many bands")
    ap.add_argument("--band-min-depth-m", type=float, default=None,
                    help="keep bands until the far edge reaches this depth")
    ap.add_argument("--band-overlap-frac", type=float, default=0.25,
                    help="band overlap as a fraction of the previous band's window size")
    ap.add_argument("--band-simplify", type=float, default=4.0,
                    help="simplify bands with a > 1 to 1/this of their faces (0 = off)")
    ap.add_argument("--cross-band", action=argparse.BooleanOptionalAction, default=True,
                    help="pin each band's overlap with the previous band to that band's latents")
    ap.add_argument("--max-inflated-voxels", type=int, default=100_000,
                    help="SLat joint-decode single-pass cap; lower it if the decode runs out of memory")
    ap.add_argument("--spiral-video", action=argparse.BooleanOptionalAction, default=True,
                    help="also render spiral.mp4 (orbit render next to the input)")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
