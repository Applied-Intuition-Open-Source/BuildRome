# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Single image -> metric 3D scene mesh, as a reusable pipeline object.

    from build_rome import ScenePipeline

    pipe = ScenePipeline.from_pretrained("ss_delta.safetensors", "slat_delta.safetensors")
    scene = pipe("photo.jpg")            # SceneOutput
    scene.save("results/photo")          # scene.glb, layout_bands.json, layout_topview.png, spiral.mp4
    scene.mesh.export("scene.glb")       # trimesh.Scene, metric, glTF Y-up, camera at the origin

``from_pretrained`` loads the SS and SLat denoisers (public TRELLIS.2 base +
the fine-tuned deltas), the two decoders and the DINOv3 encoders once;
every call then only runs MoGe-3 on the photo and the two generation stages.
Weights can be local files, a directory holding ``ss_delta.safetensors`` and
``slat_delta.safetensors``, or a Hugging Face repo id with those two files.

The scene is split along depth into bands. Band 0 uses native 3 m windows
(64^3 voxels, 4.7 cm); each further band's windows are ``band_growth`` times
larger in metric space, until the far edge reaches the ``band_p`` depth
quantile of the MoGe-3 cloud (bands covering < ``band_min_coverage`` of the
image are dropped). Each band is generated in its own canonical frame: the
evidence is scaled by 1/a about the camera's floor footpoint, a 3 m window
layout is placed where the cloud has support (plus a one-window ring), and
the decoded mesh is scaled back by a. Bands run nearest-first; with
cross-band outpainting each band's latents in the overlap with the previous
band are pinned to that band's latents, and the nearer band keeps its
geometry in the overlap.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.geometry import apply_scene_scale, ensure_moge3_pack, filtered_cloud, load_lift_kwargs  # noqa: E402
from inference.models import DINO, load_image  # noqa: E402
from inference.stages import (SS_AR_ERODE, load_slat_models, load_ss_models, run_slat_stage,  # noqa: E402
                              run_ss_stage, window_order)

WINDOW_STRIDE_M = 2.4375   # 13 latent cells: 0.5625 m (3-cell) overlap between windows
WINDOW_DEPTH_START_M = 1.5 # nearest window center: its front face at the camera plane
GROUND_Z_OFF_M = -0.25     # window bases sit this far below the fitted floor (training convention)
SUPPORT_MIN_PTS = 150      # cloud points inside a window for it to count as evidence
HEIGHT_TARGET_M = 2.8      # a band grows until its p99 content height fits this (canonical m)
BAND_COLORS = ["tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]
DELTA_FILES = {"ss": "ss_delta.safetensors", "slat": "slat_delta.safetensors"}
B2G = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)  # floor frame Z-up -> glTF Y-up


# ----------------------------------------------------------------- band layout

def _p99_height(pts, lo, hi):
    m = (pts[:, 1] >= lo) & (pts[:, 1] < hi)
    return float(torch.quantile(pts[m, 2], 0.99)) if int(m.sum()) >= 50 else 0.0


def band_schedule(pts, p_depth_m, near_max_m, growth, stride_m, depth_start_m, C, overlap_m,
                  n_pix, min_coverage=0.0, band_count=None, min_depth_m=None, overlap_frac=0.0):
    """[{a, rows (canonical window-center depths), front_m, far_m, coverage}].

    Band 0 is the native lattice (a = 1) while rows stay within near_max_m and
    the p99 height of each row's depth slab fits HEIGHT_TARGET_M. Further bands
    are one row each, starting overlap inside the previous far edge, with
        a = max(previous a, depth rule, p99 height(slab) / HEIGHT_TARGET_M)
    where the depth rule multiplies by ``growth`` on every band beyond
    near_max_m. Bands are added until far_m >= p_depth_m, then the list is cut
    after the last band whose slab holds >= min_coverage of the image pixels
    (pixel coverage, not point density, which falls as 1/z^2).

    Overrides: band_count = exactly this many bands (coverage cut ignored);
    min_depth_m = keep bands until the far edge reaches this depth.
    overlap = max(overlap_m, overlap_frac x previous band's window size)."""
    rows, d = [], depth_start_m
    while d <= near_max_m + 1e-9:
        if _p99_height(pts, d - C / 2.0, d + C / 2.0) > HEIGHT_TARGET_M:
            break
        rows.append(d)
        d += stride_m
    bands = []
    if rows:
        far = rows[-1] + C / 2.0
        bands.append(dict(a=1.0, rows=rows, front_m=depth_start_m - C / 2.0, far_m=far))
    else:
        far = depth_start_m - C / 2.0 + overlap_m
    a, a_depth = 1.0, 1.0
    want_far = max(p_depth_m, min_depth_m or 0.0)
    while far < want_far - 1e-6 or (band_count and len(bands) < band_count):
        front = far - max(overlap_m, overlap_frac * a * C)
        if front >= near_max_m - 1e-9:
            a_depth *= growth
        a = max(a, a_depth)
        for _ in range(8):  # the slab grows with a: iterate to a fixed point
            a_h = _p99_height(pts, front, front + a * C) / HEIGHT_TARGET_M
            if a_h <= a + 1e-3:
                break
            a = a_h
        far = front + a * C
        bands.append(dict(a=a, rows=[(front + a * C / 2.0) / a], front_m=front, far_m=far))
    for b in bands:
        m = (pts[:, 1] >= b["front_m"]) & (pts[:, 1] < b["far_m"])
        b["coverage"] = float(m.sum()) / float(n_pix)
    if band_count:
        return bands[:band_count]
    if min_coverage > 0:
        keep = [i for i, b in enumerate(bands) if b["coverage"] >= min_coverage
                or (min_depth_m and b["front_m"] < min_depth_m)]
        bands = bands[:(keep[-1] if keep else 0) + 1]
    return bands


def band_layout(pts_c, rows, stride_m, C, z_off_m):
    """Windows for one band in its canonical frame: rows x a lateral lattice
    spanning the cloud; kept when >= SUPPORT_MIN_PTS points fall in the cube
    (evidence) or within one stride of an evidence window (ring).
    Returns (layout [(depth, lateral, z_off_m)], evidence flags) or None."""
    half = C / 2.0
    pts = pts_c[(pts_c[:, 2] >= 0.0) & (pts_c[:, 2] < C)]
    if not pts.shape[0]:
        return None
    lmin, lmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    k_lo = math.ceil((lmin - half) / stride_m - 1e-9)
    k_hi = math.floor((lmax + half) / stride_m + 1e-9)
    cand = [(float(d), k * stride_m, z_off_m) for d in rows for k in range(k_lo, k_hi + 1)]
    ct = torch.tensor(cand, device=pts.device)
    inside = ((pts[None, :, 1] - ct[:, 0, None]).abs() < half) & ((pts[None, :, 0] - ct[:, 1, None]).abs() < half)
    occupied = inside.sum(dim=1) >= SUPPORT_MIN_PTS
    if not bool(occupied.any()):
        return None
    cheb = (ct[:, None, :] - ct[occupied][None, :, :]).abs().amax(dim=-1).amin(dim=-1)
    kept = torch.nonzero(occupied | (cheb <= stride_m + 1e-6), as_tuple=True)[0].tolist()
    return [cand[i] for i in kept], [bool(occupied[i]) for i in kept]


def _window_cubes(layout, a, C, erode_m):
    """[N, 6] full and eroded window cubes (lo xyz, hi xyz) in metric units."""
    full, ero = [], []
    for d, l, z in layout:
        lo = torch.tensor([l - C / 2, d - C / 2, z])
        hi = lo + C
        full.append(torch.cat([lo, hi])[None] * a)
        ero.append(torch.cat([lo + erode_m, hi - erode_m])[None] * a)
    return torch.cat(full), torch.cat(ero)


# ----------------------------------------------------------------- cross-band outpainting

@dataclass
class BandPrior:
    """What a finished band hands to the next, farther one (metric units):
    its SS latents with their window cubes, and its SLat tokens."""

    ss_lat: torch.Tensor        # [N, 8, 16, 16, 16], one per window
    cubes: torch.Tensor         # [N, 6] window cubes (lo xyz, hi xyz)
    cubes_eroded: torch.Tensor  # [N, 6] the same minus the untrusted border
    tok_centers: torch.Tensor   # [M, 3]
    tok_feats: torch.Tensor     # [M, 32] normalized SLat feats
    tok_pitch: float            # token spacing
    tok_origin: torch.Tensor    # [3] corner of the band's token lattice

    def in_canonical_frame(self, a):
        """(ss prior, slat prior) as run_ss_stage / run_slat_stage expect
        them, for a band whose canonical frame is metric / a."""
        ss = dict(lat=self.ss_lat, full=self.cubes / a, boxes=self.cubes_eroded / a)
        slat = dict(centers=self.tok_centers / a, feats=self.tok_feats, pitch=self.tok_pitch / a,
                    origin=self.tok_origin / a, boxes=self.cubes_eroded / a)
        return ss, slat


def _band_prior(ss_latents, tokens, cubes, cubes_eroded, a, C):
    """Pack a band's latents (canonical frame, scale a) into a BandPrior."""
    toks, lay = tokens
    centers = []
    for (lc, _), (d, l, z) in zip(toks, lay):
        centers.append(((lc.float() + 0.5) * (C / 64) + torch.tensor([l - C / 2, d - C / 2, z])) * a)
    d0, l0, z0 = lay[0]
    return BandPrior(ss_lat=torch.cat([x.detach().float().cpu() for x in ss_latents]),
                     cubes=cubes, cubes_eroded=cubes_eroded,
                     tok_centers=torch.cat(centers), tok_feats=torch.cat([f for _, f in toks]),
                     tok_pitch=a * C / 64, tok_origin=torch.tensor([l0 - C / 2, d0 - C / 2, z0]) * a)


def _drop_faces_inside(mesh, T, cubes):
    """Remove the faces whose centroid, mapped through T, lies in any of the
    cubes [K, 6]. Returns (mesh, number of faces removed)."""
    from build_rome.representations import Mesh

    V = mesh.vertices.detach().double()
    T = torch.as_tensor(np.asarray(T, dtype=np.float64), device=V.device)
    Vw = V @ T[:3, :3].T + T[:3, 3]
    F = mesh.faces.long()
    cen = (Vw[F[:, 0]] + Vw[F[:, 1]] + Vw[F[:, 2]]) / 3.0
    inside = torch.zeros(len(F), dtype=torch.bool, device=V.device)
    for c in cubes.to(Vw):
        inside |= ((cen >= c[:3]) & (cen <= c[3:])).all(dim=1)
    return Mesh(mesh.vertices.detach(), mesh.faces[~inside]), int(inside.sum())


# ----------------------------------------------------------------- outputs

def scene_to_trimesh(meshes, transforms):
    """Per-band meshes (window-0-local) + transforms -> one trimesh.Scene,
    metric, glTF Y-up, camera at the origin looking along -Z."""
    import trimesh

    scene = trimesh.Scene()
    for k, (m, T) in enumerate(zip(meshes, transforms)):
        if m is None or m.faces.shape[0] == 0:
            continue
        tm = trimesh.Trimesh(vertices=m.vertices.detach().cpu().numpy(),
                             faces=m.faces.detach().cpu().numpy(), process=False)
        tm.remove_unreferenced_vertices()
        scene.add_geometry(tm, transform=B2G @ np.asarray(T, dtype=float), node_name=f"w{k:02d}")
    return scene


def export_scene_glb(meshes, transforms, path: str) -> None:
    """Metric scene mesh, camera at the origin; floor frame Z-up -> glTF Y-up."""
    scene_to_trimesh(meshes, transforms).export(path)
    print(f"[export] wrote {path}")


def compose_frame(panels, labels, sep_px: int = 6) -> np.ndarray:
    """Panels side by side with red separators and top-left labels."""
    H = panels[0].shape[0]
    red = np.zeros((H, sep_px, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    strips = []
    for i, p in enumerate(panels):
        strips += ([red] if i else []) + [p]
    img = Image.fromarray(np.concatenate(strips, axis=1))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, H // 20))
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for i, (p, label) in enumerate(zip(panels, labels)):
        x += sep_px if i else 0
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            draw.text((x + 10 + dx, 8 + dy), label, fill=(0, 0, 0), font=font)
        draw.text((x + 10, 8), label, fill=(255, 255, 255), font=font)
        x += p.shape[1]
    return np.array(img)


def plot_topview(pts_m, records, meshes, transforms, K, p_depth_m, path, title):
    """Top-down debug plot: cloud, generated mesh, band windows, frustum."""
    try:
        import matplotlib
    except ImportError:
        print(f"[bands] matplotlib not available; skipping {os.path.basename(path)}")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    P = pts_m.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 12))
    x_lo = min(min(l * r["a"] - r["a"] * r["chunk_m"] / 2 for _, l in r["layout_2d"]) for r in records) - 0.5
    ax.hexbin(P[:, 0], P[:, 1], gridsize=220, bins="log", cmap="Greys", mincnt=1, linewidths=0, zorder=1)
    V = []
    for m, T in zip(meshes, transforms):
        v = m.vertices.detach().float().cpu().numpy()
        v = v @ np.asarray(T, dtype=np.float32)[:3, :3].T + np.asarray(T)[:3, 3]
        if v.shape[0] > 300_000:
            v = v[np.random.default_rng(0).choice(v.shape[0], 300_000, replace=False)]
        V.append(v)
    V = np.concatenate(V)
    ax.hexbin(V[:, 0], V[:, 1], gridsize=220, bins="log", cmap="Blues", mincnt=1, alpha=0.45,
              linewidths=0, zorder=2)
    for r in records:
        a, col = r["a"], BAND_COLORS[r["band"] % len(BAND_COLORS)]
        w_m = a * r["chunk_m"]
        for (d, l), ev, k in zip(r["layout_2d"], r["evidence"], r["order"]):
            ax.add_patch(Rectangle((l * a - w_m / 2, d * a - w_m / 2), w_m, w_m, fill=False,
                                   lw=1.8 if ev else 1.2, ls="-" if ev else "--", ec=col, zorder=4))
            ax.text(l * a, d * a, f"{k}", ha="center", va="center", fontsize=9, color=col,
                    fontweight="bold", zorder=5)
        ax.axhline(r["front_m"], color=col, lw=0.6, ls=":", zorder=3)
        ax.text(x_lo, r["front_m"], f"  band {r['band']}  x{a:.3g}  ({w_m:.3g} m windows, "
                f"{w_m / 64 * 100:.1f} cm voxels, {len(r['layout_2d'])} cols)", color=col,
                fontsize=9, va="bottom", ha="left", zorder=5)
    ax.axhline(p_depth_m, color="k", lw=0.8, ls="--", zorder=3)
    ax.text(x_lo, p_depth_m, f"  p-depth {p_depth_m:.1f} m", va="bottom", fontsize=9)
    half_fov = math.atan(float(K[0, 2]) / float(K[0, 0]))
    dmax = max(p_depth_m * 1.15, max(r["far_m"] for r in records))
    for sgn in (-1, 1):
        ax.plot([0, sgn * dmax * math.tan(half_fov)], [0, dmax], color="k", lw=0.8, zorder=3)
    ax.plot(0, 0, marker="^", color="k", ms=9, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("depth (m)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-1.5, dmax + 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@dataclass
class SceneOutput:
    """Result of one ``ScenePipeline`` call."""

    stem: str
    meshes: list                    # per band, build_rome Mesh in the window-0-local frame
    transforms: list                # per band, [4, 4] window-local -> metric floor frame
    layout: dict                    # bands, windows, parameters (the layout_bands.json content)
    pack_path: str                  # MoGe-3 depth pack used for conditioning
    image: torch.Tensor             # [3, 512, 512] conditioning image in [0, 1]
    pts_m: torch.Tensor = field(repr=False)   # filtered MoGe-3 cloud, metric floor frame
    K: np.ndarray = field(repr=False)         # pixel intrinsics of the depth pack

    @property
    def mesh(self):
        """The scene as a ``trimesh.Scene`` (metric, glTF Y-up, camera at the origin)."""
        return scene_to_trimesh(self.meshes, self.transforms)

    def render_spiral(self, resolution: int = 1024, num_frames: int = 120) -> np.ndarray:
        """[T, H, 2W + 6, 3] uint8 frames: input photo next to an orbit render."""
        from build_rome.pipelines.render import build_spiral_cameras, merge_chunk_meshes, render_mesh_spiral

        scene = merge_chunk_meshes(self.meshes, self.transforms, 999.0)
        extr, intr = build_spiral_cameras(num_frames=num_frames, r=1.0, fov=60.0,
                                          base_elevation_deg=35.0, elevation_amp_deg=15.0)
        frames = render_mesh_spiral(scene, extr, intr, resolution, near_clip_frac=0.15)
        inp = (torch.nn.functional.interpolate(self.image[None], size=(resolution, resolution), mode="bilinear",
                                               align_corners=False)[0].clamp(0, 1) * 255)
        inp = inp.byte().permute(1, 2, 0).numpy()
        labels = ["input", f"SLat ({len(self.layout['bands'])} bands)"]
        return np.stack([compose_frame([inp, fr], labels) for fr in frames])

    def save(self, out_dir: str, spiral_video: bool = True, topview: bool = True) -> str:
        """Write scene.glb, layout_bands.json and (optionally) layout_topview.png,
        spiral.mp4 into ``out_dir``; returns ``out_dir``."""
        os.makedirs(out_dir, exist_ok=True)
        export_scene_glb(self.meshes, self.transforms, os.path.join(out_dir, "scene.glb"))
        with open(os.path.join(out_dir, "layout_bands.json"), "w") as f:
            json.dump(self.layout, f, indent=1)
        if topview:
            plot_topview(self.pts_m, self.layout["bands"], self.meshes, self.transforms, self.K,
                         self.layout["p_depth_m"], os.path.join(out_dir, "layout_topview.png"),
                         f"{self.stem}: {len(self.layout['bands'])} bands, gray = MoGe-3 cloud, blue = generated mesh")
        if spiral_video:
            from build_rome.pipelines.render import write_video_file

            write_video_file(self.render_spiral(), os.path.join(out_dir, "spiral.mp4"), fps=30)
        print(f"[done] {out_dir}")
        return out_dir


# ----------------------------------------------------------------- pipeline

def _resolve_weights(ss_ckpt: str, slat_ckpt: Optional[str]):
    """(ss path, slat path) from local files, a directory with the two deltas,
    or a Hugging Face repo id (``org/name`` or ``org/name@revision``)."""
    if slat_ckpt is not None:
        return ss_ckpt, slat_ckpt
    if os.path.isdir(ss_ckpt):
        return os.path.join(ss_ckpt, DELTA_FILES["ss"]), os.path.join(ss_ckpt, DELTA_FILES["slat"])
    if os.path.isfile(ss_ckpt):
        raise ValueError("pass both checkpoints: from_pretrained(ss_ckpt, slat_ckpt)")
    from huggingface_hub import hf_hub_download

    repo, _, revision = ss_ckpt.partition("@")
    return tuple(hf_hub_download(repo, DELTA_FILES[k], revision=revision or None) for k in ("ss", "slat"))


class ScenePipeline:
    """Single image -> metric 3D scene mesh. Build with ``from_pretrained``,
    then call with a photo (path or PIL image)."""

    def __init__(self, ss_models, slat_models, extractors, device, moge3_python: str, moge3_cache: str):
        self.ss_models = ss_models          # (denoiser, occupancy decoder)
        self.slat_models = slat_models      # (denoiser, shape decoder)
        self.extractor, self.slat_extractor = extractors
        self.device = device
        self.moge3_python = moge3_python
        self.moge3_cache = moge3_cache

    @classmethod
    def from_pretrained(cls, ss_ckpt: str, slat_ckpt: Optional[str] = None, *, device="cuda",
                        moge3_python: Optional[str] = None, moge3_cache: Optional[str] = None):
        """Load everything once.

        ss_ckpt / slat_ckpt: the fine-tuned deltas (``.safetensors``, combined
            with the public TRELLIS.2 base weights downloaded from Hugging
            Face) or full Lightning ``.ckpt`` files. With ``slat_ckpt`` omitted,
            ``ss_ckpt`` is a directory holding ``ss_delta.safetensors`` and
            ``slat_delta.safetensors`` or a Hugging Face repo id with them.
        moge3_python: python of an environment with MoGe-3 installed (its
            dependency pins can conflict with this environment); default: the
            current interpreter.
        moge3_cache: directory of MoGe-3 depth packs, keyed by image content."""
        from build_rome.modules.image_features import DinoV3FeatureExtractor

        device = torch.device(device)
        ss_path, slat_path = _resolve_weights(ss_ckpt, slat_ckpt)
        extractor = DinoV3FeatureExtractor(DINO, image_size=512)        # SS cross-attention
        slat_extractor = DinoV3FeatureExtractor(DINO, image_size=1024)  # SLat cross-attention
        extractor.to(device)
        slat_extractor.to(device)
        return cls(load_ss_models(ss_path, device), load_slat_models(slat_path, device),
                   (extractor, slat_extractor), device,
                   moge3_python or sys.executable,
                   moge3_cache or os.path.expanduser("~/.cache/image_to_scene/moge3"))

    def depth_pack(self, image_path: str) -> str:
        """Path of the MoGe-3 depth pack for a photo (built on first use)."""
        return ensure_moge3_pack(image_path, self.moge3_python, self.moge3_cache)

    @torch.no_grad()
    def __call__(self, image: Union[str, "Image.Image"], *, seed: int = 0, depth_pack: Optional[str] = None,
                 band_near_m: float = 9.0, band_growth: float = 2.0, band_p: float = 0.99,
                 band_min_coverage: float = 0.05, band_count: Optional[int] = None,
                 band_min_depth_m: Optional[float] = None, band_overlap_frac: float = 0.25,
                 band_simplify: float = 4.0, cross_band: bool = True,
                 max_inflated_voxels: int = 100_000) -> SceneOutput:
        """Generate the scene of one photo.

        image: path, or a PIL image (written to a temporary file for MoGe-3).
        depth_pack: a precomputed depth pack (.npz) instead of running MoGe-3.
        band_*: depth-band schedule (see the module docstring); cross_band:
            pin each band's overlap with the previous band to its latents;
            max_inflated_voxels: shape-decode memory cap (lower on OOM)."""
        from build_rome.pipelines import windows as mdv

        tmp = None
        if not isinstance(image, str):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.convert("RGB").save(tmp.name)
            image_path, stem = tmp.name, "scene"
        else:
            image_path, stem = image, os.path.splitext(os.path.basename(image))[0]
        try:
            img = load_image(image_path)
            pack_path = depth_pack or self.depth_pack(image_path)
            lift_kwargs = load_lift_kwargs(pack_path, self.device)
            lift_hi_feats = self.extractor(load_image(image_path, size=1024)[None])  # depth-lift source
        finally:
            if tmp is not None:
                os.unlink(tmp.name)

        C = mdv.CHUNK_SIZE
        stride_m = WINDOW_STRIDE_M
        base_overlap_m = C - stride_m
        pts_m = filtered_cloud(lift_kwargs)
        h99 = float(torch.quantile(pts_m[:, 2], 0.99))
        p_depth = float(torch.quantile(pts_m[:, 1], band_p))
        bands = band_schedule(
            pts_m, p_depth, band_near_m, band_growth, stride_m, WINDOW_DEPTH_START_M, C,
            base_overlap_m, n_pix=int(lift_kwargs["lift_depth"][0].numel()),
            min_coverage=band_min_coverage, band_count=band_count,
            min_depth_m=band_min_depth_m, overlap_frac=band_overlap_frac)
        print(f"[bands] cloud p{band_p:.2f} depth {p_depth:.2f} m, p99 height {h99:.2f} m; {len(bands)} bands:")
        for i, b in enumerate(bands):
            print(f"[bands]   band {i}: x{b['a']:.3g} ({b['a'] * C:.3g} m windows, voxel "
                  f"{b['a'] * C / 64 * 100:.1f} cm) depth {b['front_m']:.2f}..{b['far_m']:.2f} m, "
                  f"{b['coverage']:.1%} of image")

        erode_m = SS_AR_ERODE * C / 16
        meshes, transforms, records = [], [], []
        owned = []      # window cubes of the bands done so far: they keep their geometry
        prior = None    # the previous band's latents (cross-band outpainting)
        for bi, b in enumerate(bands):
            a = float(b["a"])
            lay = band_layout(pts_m / a, b["rows"], stride_m, C, GROUND_Z_OFF_M)
            if lay is None:
                print(f"[bands] band {bi} (x{a:g}): no window has support, skipped")
                continue
            layout, evidence = lay
            print(f"[bands] === band {bi}: x{a:g}, {len(layout)} windows "
                  f"({sum(evidence)} evidence + {len(layout) - sum(evidence)} ring) ===")
            out = self._run_band(img, lift_kwargs, lift_hi_feats, layout, evidence, a, seed + 1000 * bi,
                                 prior if cross_band else None, max_inflated_voxels)
            if out is None:
                print(f"[bands] band {bi} (x{a:g}) produced no geometry, skipped")
                continue
            mesh, T0, ss_latents, tokens = out
            T = np.diag([a, a, a, 1.0]).astype(np.float32) @ T0    # window-0-local -> metric
            n_dec = int(mesh.faces.shape[0])
            if band_simplify > 1 and a > 1.0 + 1e-6:
                t0 = time.time()
                mesh.simplify(target=int(n_dec / band_simplify))
                print(f"[bands] band {bi} simplify x{band_simplify:g}: {n_dec / 1e6:.1f}M -> "
                      f"{mesh.faces.shape[0] / 1e6:.2f}M faces in {time.time() - t0:.1f}s")
            cubes, cubes_eroded = _window_cubes(layout, a, C, erode_m)
            if cross_band:
                if owned:
                    mesh, n_dropped = _drop_faces_inside(mesh, T, torch.cat(owned))
                    print(f"[cross-band] band {bi}: dropped {n_dropped:,} faces inside nearer bands")
                owned.append(cubes)
                prior = _band_prior(ss_latents, tokens, cubes, cubes_eroded, a, C)
            meshes.append(mesh)
            transforms.append(T)
            rank = [0] * len(layout)
            for k, i in enumerate(window_order(layout, evidence)):
                rank[i] = k + 1
            records.append(dict(band=bi, a=a, chunk_m=C, front_m=b["front_m"], far_m=b["far_m"],
                                coverage=b["coverage"], layout_2d=[[float(d), float(l)] for d, l, _ in layout],
                                evidence=evidence, order=rank,
                                layout_canonical=[[float(c) for c in w] for w in layout],
                                n_faces_decoded=n_dec, n_faces=int(mesh.faces.shape[0])))
            torch.cuda.empty_cache()
        assert meshes, "no band produced a mesh"

        layout = dict(p_depth_m=p_depth, band_p=band_p, near_m=band_near_m, growth=band_growth,
                      overlap_m=base_overlap_m, stride_m=stride_m, p99_height_m=h99, bands=records)
        return SceneOutput(stem=stem, meshes=meshes, transforms=transforms, layout=layout, pack_path=pack_path,
                           image=img, pts_m=pts_m, K=lift_kwargs["lift_K"][0].cpu().numpy())

    def _run_band(self, img, lift_kwargs, lift_hi_feats, layout, evidence, a, seed, prior, max_inflated_voxels):
        """SS then SLat for one band in its canonical frame (evidence scaled by
        1/a). Returns (mesh, T0, ss_latents, tokens), or None if the band came
        out empty."""
        lk = apply_scene_scale({k: v.clone() for k, v in lift_kwargs.items()}, 1.0 / a)
        ss_prior = slat_prior = None
        if prior is not None:
            ss_prior, slat_prior = prior.in_canonical_frame(a)
        try:
            occs, ss_latents, world_coords, R = run_ss_stage(
                self.ss_models, img, self.extractor, self.device, lk, lift_hi_feats, layout, evidence, seed,
                prior=ss_prior)
            del occs
            torch.cuda.empty_cache()
            mesh, T0, tokens = run_slat_stage(
                self.slat_models, img, self.slat_extractor, self.device, lk, lift_hi_feats, layout,
                world_coords, R, seed, scene_scale=1.0 / a, prior=slat_prior,
                max_inflated_voxels=max_inflated_voxels)
        except AssertionError as e:
            # every window below MIN_VOXELS; the other bands still make a scene
            print(f"[bands] {e}")
            torch.cuda.empty_cache()
            return None
        return mesh, T0, ss_latents, tokens
