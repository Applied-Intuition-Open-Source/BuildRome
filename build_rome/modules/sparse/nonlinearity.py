# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# Portions of this file are derived from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2),
# Copyright (c) Microsoft Corporation, licensed under the MIT License, and were
# modified by The Build Rome Authors. See NOTICE and third_party/licenses/TRELLIS.2-MIT.txt.

import torch.nn as nn

from . import VarLenTensor

__all__ = ["SparseReLU", "SparseSiLU", "SparseGELU", "SparseActivation"]


class SparseReLU(nn.ReLU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))


class SparseSiLU(nn.SiLU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))


class SparseGELU(nn.GELU):
    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))


class SparseActivation(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(self.activation(input.feats))
