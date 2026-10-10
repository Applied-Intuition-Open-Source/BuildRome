# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

"""Spiral (orbit) rendering of decoded Sparse Structure voxels for validation.

Reuses the repo's own ``VoxelRenderer`` / ``Voxel`` and camera helpers. The
camera path is a full 360 azimuth sweep with the elevation oscillating up and
down (a TRELLIS-style turntable spiral), so a single video shows the object
from all sides and heights.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from ..representations import Mesh, Voxel
from ..utils.render_utils import get_renderer, render_frames, yaw_pitch_r_fov_to_extrinsics_intrinsics


def build_spiral_cameras(
    num_frames: int = 60,
    r: float = 2.0,
    fov: float = 40.0,
    base_elevation_deg: float = 20.0,
    elevation_amp_deg: float = 30.0,
    elevation_cycles: float = 1.0,
):
    """Azimuth 0->360 with sinusoidally varying elevation (spiral turntable)."""
    two_pi = 2.0 * np.pi
    yaws = (-torch.linspace(0, two_pi, num_frames) + np.pi / 2).tolist()
    theta = torch.linspace(0, two_pi * elevation_cycles, num_frames)
    base = np.deg2rad(base_elevation_deg)
    amp = np.deg2rad(elevation_amp_deg)
    pitchs = (base + amp * torch.sin(theta)).tolist()
    return yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)


@torch.no_grad()
def decode_latent_to_voxels(ss_dec, z: torch.Tensor) -> List[Voxel]:
    """Decode SS latents ``z`` [B, C, 16, 16, 16] into a list of colored Voxels."""
    decoder_dtype = next(ss_dec.parameters()).dtype
    occ = ss_dec(z.to(dtype=decoder_dtype)) > 0  # [B, 1, R, R, R]
    resolution = occ.shape[-1]
    voxels: List[Voxel] = []
    for i in range(occ.shape[0]):
        coords = torch.nonzero(occ[i, 0], as_tuple=False)  # [N, 3]
        if coords.shape[0] == 0:
            voxels.append(None)
            continue
        color = coords.float() / resolution  # position-based RGB
        voxels.append(
            Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1.0 / resolution,
                coords=coords.int(),
                attrs=color,
                layout={"color": slice(0, 3)},
            )
        )
    return voxels


@torch.no_grad()
def merge_chunk_voxels(voxels: List[Voxel], transforms, cutaway_z_m: float = None) -> Voxel:
    """Assemble per-chunk Voxels into one world-space scene Voxel.

    ``transforms`` are the 4x4 ``chunk_to_world`` sidecar matrices; chunk-local
    voxel centers live in [-0.5, 0.5]^3 (floor-anchored cubes, world z up). If
    ``cutaway_z_m`` is set, voxels above that world height are dropped — the
    "dollhouse" view that removes ceilings so an orbit camera can see inside.

    Overlapping chunks are deduped by re-voxelizing at the common world voxel
    size, and the merged scene is normalized to the unit cube centered at the
    origin so the existing spiral cameras work unchanged. Colors encode
    normalized world position. Returns None if nothing survives.
    """
    pts = []
    world_vs = None
    for v, T in zip(voxels, transforms):
        if v is None:
            continue
        T = torch.as_tensor(np.asarray(T), dtype=torch.float32, device=v.coords.device)
        local = v.origin[None, :] + (v.coords.float() + 0.5) * v.voxel_size
        pts.append(local @ T[:3, :3].T + T[:3, 3])
        if world_vs is None:
            world_vs = (T[:3, 0].norm() * v.voxel_size).item()  # uniform scale assumed
    if not pts:
        return None
    pts = torch.cat(pts, dim=0)
    if cutaway_z_m is not None:
        pts = pts[pts[:, 2] <= cutaway_z_m]
    if pts.shape[0] == 0:
        return None

    pmin, pmax = pts.min(dim=0).values, pts.max(dim=0).values
    coords = torch.unique(((pts - pmin) / world_vs).long(), dim=0)
    extent = (pmax - pmin).max().item() + world_vs
    color = ((coords.float() + 0.5) * world_vs / (pmax - pmin + 1e-6)).clamp(0, 1)
    return Voxel(
        origin=(-(pmax - pmin) / (2.0 * extent)).tolist(),
        voxel_size=world_vs / extent,
        coords=coords.int(),
        attrs=color,
        layout={"color": slice(0, 3)},
        device=coords.device,
    )


@torch.no_grad()
def merge_chunk_meshes(meshes: List[Mesh], transforms, cutaway_z_m: float = None) -> Mesh:
    """Assemble per-chunk decoded meshes into one world-space scene Mesh.

    Mesh counterpart of ``merge_chunk_voxels``: chunk-local vertices live in
    [-0.5, 0.5]^3, ``transforms`` are the 4x4 ``chunk_to_world`` sidecar
    matrices. If ``cutaway_z_m`` is set, faces with any vertex above that world
    height are dropped (dollhouse: ceiling removed). The merged mesh is
    normalized to the unit cube centered at the origin so the spiral cameras
    work unchanged. Returns None if nothing survives.
    """
    verts, faces = [], []
    v_offset = 0
    for m, T in zip(meshes, transforms):
        if m is None or m.faces.shape[0] == 0:
            continue
        T = torch.as_tensor(np.asarray(T), dtype=torch.float32, device=m.vertices.device)
        verts.append(m.vertices @ T[:3, :3].T + T[:3, 3])
        faces.append(m.faces.long() + v_offset)
        v_offset += verts[-1].shape[0]
    if not verts:
        return None
    verts, faces = torch.cat(verts), torch.cat(faces)
    if cutaway_z_m is not None:
        above = verts[:, 2] > cutaway_z_m
        faces = faces[~above[faces].any(dim=1)]
    if faces.shape[0] == 0:
        return None

    used = torch.unique(faces)
    pmin, pmax = verts[used].min(dim=0).values, verts[used].max(dim=0).values
    extent = (pmax - pmin).max().clamp(min=1e-6)
    verts = (verts - (pmin + pmax) / 2.0) / extent
    return Mesh(verts, faces.int())


@torch.no_grad()
def render_mesh_spiral(
    mesh: Mesh,
    extrinsics,
    intrinsics,
    resolution: int = 512,
    near_clip_frac: float = None,
) -> np.ndarray:
    """Render a Mesh's normal maps along the camera path. Returns [T, H, W, 3] uint8.

    ``near_clip_frac`` is the same per-frame camera-facing cutaway as in
    ``render_voxel_spiral``, applied to face centroids so the orbit always
    looks into an open cross-section instead of at the outside of a wall.
    """
    if mesh is None:
        return np.zeros((len(extrinsics), resolution, resolution, 3), dtype=np.uint8)
    renderer = get_renderer(mesh, resolution=resolution, near=0.1, far=10.0, chunk_size=5_000_000)
    centroids = mesh.vertices[mesh.faces.long()].mean(dim=1) if near_clip_frac is not None else None

    frames = []
    for extr, intr in zip(extrinsics, intrinsics):
        frame_mesh = mesh
        if near_clip_frac is not None:
            extr_t = torch.as_tensor(extr, dtype=torch.float32, device=mesh.vertices.device)
            cam = -extr_t[:3, :3].T @ extr_t[:3, 3]
            fwd = -cam[:2] / (cam[:2].norm() + 1e-8)
            keep = (centroids[:, :2] @ fwd) >= -near_clip_frac
            if not torch.any(keep):
                frames.append(np.zeros((resolution, resolution, 3), dtype=np.uint8))
                continue
            frame_mesh = Mesh(mesh.vertices, mesh.faces[keep])
        res = renderer.render(frame_mesh, extr, intr, return_types=["normal"])["normal"]
        frames.append(np.clip(res.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
    return np.stack(frames, axis=0)


@torch.no_grad()
def render_voxel_spiral(
    voxel: Voxel,
    extrinsics,
    intrinsics,
    resolution: int = 512,
    bg_color=(0, 0, 0),
    near_clip_frac: float = None,
) -> np.ndarray:
    """Render a single Voxel along the given camera path. Returns [T, H, W, 3] uint8.

    ``near_clip_frac`` enables a per-frame camera-facing cutaway: voxels on the
    near side of the scene (horizontal offset toward the camera greater than
    this fraction of the unit cube) are dropped each frame, so the orbit always
    looks into an open cross-section instead of at the outside of a wall.
    """
    if voxel is None:
        return np.zeros((len(extrinsics), resolution, resolution, 3), dtype=np.uint8)
    options = {"resolution": resolution, "bg_color": bg_color}
    if near_clip_frac is None:
        rets = render_frames(voxel, extrinsics, intrinsics, options=options, verbose=False)
        return np.stack(rets["color"], axis=0)  # [T, H, W, 3]

    pos = voxel.position  # [N, 3] object space (unit cube around the origin)
    frames = []
    for extr, intr in zip(extrinsics, intrinsics):
        extr_t = torch.as_tensor(extr, dtype=torch.float32, device=pos.device)
        cam = -extr_t[:3, :3].T @ extr_t[:3, 3]  # camera position in object space
        fwd = -cam[:2] / (cam[:2].norm() + 1e-8)  # horizontal view direction (toward origin)
        keep = (pos[:, :2] @ fwd) >= -near_clip_frac
        if not torch.any(keep):
            frames.append(np.zeros((resolution, resolution, 3), dtype=np.uint8))
            continue
        clipped = Voxel(
            origin=voxel.origin.tolist(),
            voxel_size=voxel.voxel_size,
            coords=voxel.coords[keep],
            attrs=voxel.attrs[keep],
            layout=voxel.layout,
            device=voxel.coords.device,
        )
        rets = render_frames(clipped, [extr], [intr], options=options, verbose=False)
        frames.append(rets["color"][0])
    return np.stack(frames, axis=0)


def compose_side_by_side(*videos: np.ndarray, gap: int = 4) -> np.ndarray:
    """Concatenate [T, H, W, 3] videos left-to-right with thin white separators."""
    T, H, _, _ = videos[0].shape
    sep = np.full((T, H, gap, 3), 255, dtype=np.uint8)
    panels = []
    for i, v in enumerate(videos):
        if i > 0:
            panels.append(sep)
        panels.append(v)
    return np.concatenate(panels, axis=2)


def image_panel(image_chw: torch.Tensor, num_frames: int, resolution: int) -> np.ndarray:
    """Turn a [3, H, W] float image in [0, 1] into a static [T, res, res, 3] uint8 video panel."""
    img = torch.nn.functional.interpolate(
        image_chw[None].float(), size=(resolution, resolution), mode="bilinear", align_corners=False
    )[0]
    img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    return np.repeat(img[None], num_frames, axis=0)


def write_video_file(frames_thwc: np.ndarray, path: str, fps: int = 15) -> str:
    """Encode [T, H, W, 3] uint8 frames to ``path`` (``.gif`` or ``.mp4``) via imageio.

    ``imageio`` (+ ``imageio-ffmpeg`` for mp4) is the only dependency.
    """
    import imageio.v2 as imageio

    frames = [np.ascontiguousarray(frames_thwc[i]) for i in range(len(frames_thwc))]
    if path.lower().endswith(".gif"):
        imageio.mimsave(path, frames, fps=fps, loop=0)
    else:
        # macro_block_size=1 stops ffmpeg from silently resizing to multiples of 16
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path
