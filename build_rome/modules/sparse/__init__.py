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

from . import config

__attributes = {
    "VarLenTensor": "basic",
    "varlen_cat": "basic",
    "varlen_unbind": "basic",
    "SparseTensor": "basic",
    "sparse_cat": "basic",
    "sparse_unbind": "basic",
    "SparseGroupNorm": "norm",
    "SparseLayerNorm": "norm",
    "SparseGroupNorm32": "norm",
    "SparseLayerNorm32": "norm",
    "SparseReLU": "nonlinearity",
    "SparseSiLU": "nonlinearity",
    "SparseGELU": "nonlinearity",
    "SparseActivation": "nonlinearity",
    "SparseLinear": "linear",
    "sparse_scaled_dot_product_attention": "attention",
    "sparse_windowed_scaled_dot_product_self_attention": "attention",
    "sparse_windowed_scaled_dot_product_cross_attention": "attention",
    "SparseRotaryPositionEmbedder": "attention",
    "SparseMultiHeadAttention": "attention",
    "SparseConv3d": "conv",
    "SparseInverseConv3d": "conv",
    "SparseDownsample": "spatial",
    "SparseUpsample": "spatial",
    "SparseSpatial2Channel": "spatial",
    "SparseChannel2Spatial": "spatial",
}

__submodules = ["transformer", "conv"]

__all__ = list(__attributes.keys()) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == "__main__":
    import conv
    import transformer

    from .attention import *
    from .basic import *
    from .conv import *
    from .linear import *
    from .nonlinearity import *
    from .norm import *
    from .spatial import *
