# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# Portions of this file are derived from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2),
# Copyright (c) Microsoft Corporation, licensed under the MIT License, and were
# modified by The Build Rome Authors. See NOTICE and third_party/licenses/TRELLIS.2-MIT.txt.

import importlib

import torch.nn as nn

from .. import SparseTensor, config

_backends = {}


class SparseConv3d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None
    ):
        super(SparseConv3d, self).__init__()
        if config.CONV not in _backends:
            _backends[config.CONV] = importlib.import_module(f"..conv_{config.CONV}", __name__)
        _backends[config.CONV].sparse_conv3d_init(
            self, in_channels, out_channels, kernel_size, stride, dilation, padding, bias, indice_key
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _backends[config.CONV].sparse_conv3d_forward(self, x)


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        if config.CONV not in _backends:
            _backends[config.CONV] = importlib.import_module(f"..conv_{config.CONV}", __name__)
        _backends[config.CONV].sparse_inverse_conv3d_init(
            self, in_channels, out_channels, kernel_size, stride, dilation, bias, indice_key
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _backends[config.CONV].sparse_inverse_conv3d_forward(self, x)
