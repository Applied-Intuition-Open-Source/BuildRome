# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# Portions of this file are derived from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2),
# Copyright (c) Microsoft Corporation, licensed under the MIT License, and were
# modified by The Build Rome Authors. See NOTICE and third_party/licenses/TRELLIS.2-MIT.txt.

from typing import Dict

import torch


class Voxel:
    def __init__(
        self,
        origin: list,
        voxel_size: float,
        coords: torch.Tensor = None,
        attrs: torch.Tensor = None,
        layout: Dict = {},
        device: torch.device = "cuda",
    ):
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.layout = layout
        self.device = device

    @property
    def position(self):
        return (self.coords + 0.5) * self.voxel_size + self.origin[None, :]

    def split_attrs(self):
        return {k: self.attrs[:, self.layout[k]] for k in self.layout}

    def save(self, path):
        # lazy import
        if "o_voxel" not in globals():
            import o_voxel
        o_voxel.io.write(
            path,
            self.coords,
            self.split_attrs(),
        )

    def load(self, path):
        # lazy import
        if "o_voxel" not in globals():
            import o_voxel
        coord, attrs = o_voxel.io.read(path)
        self.coords = coord.int().to(self.device)
        self.attrs = torch.cat([attrs[k] for k in attrs], dim=1).to(self.device)
        # build layout
        start = 0
        self.layout = {}
        for k in attrs:
            self.layout[k] = slice(start, start + attrs[k].shape[1])
            start += attrs[k].shape[1]
