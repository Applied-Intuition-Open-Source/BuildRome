# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# Portions of this file are derived from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2),
# Copyright (c) Microsoft Corporation, licensed under the MIT License, and were
# modified by The Build Rome Authors. See NOTICE and third_party/licenses/TRELLIS.2-MIT.txt.

import torch
import torchsparse

from .. import SparseTensor


def sparse_conv3d_init(
    self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None
):
    self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias)


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    out = self.conv(x.data)
    new_shape = [x.shape[0], self.conv.out_channels]
    out = SparseTensor(
        out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None
    )
    out._spatial_cache = x._spatial_cache
    out._scale = tuple([s * stride for s, stride in zip(x._scale, self.conv.stride)])
    return out


def sparse_inverse_conv3d_init(
    self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None
):
    self.conv = torchsparse.nn.Conv3d(
        in_channels, out_channels, kernel_size, stride, 0, dilation, bias, transposed=True
    )


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    out = self.conv(x.data)
    new_shape = [x.shape[0], self.conv.out_channels]
    out = SparseTensor(
        out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None
    )
    out._spatial_cache = x._spatial_cache
    out._scale = tuple([s / stride for s, stride in zip(x._scale, self.conv.stride)])
    return out
