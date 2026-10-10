# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# Portions of this file are derived from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2),
# Copyright (c) Microsoft Corporation, licensed under the MIT License, and were
# modified by The Build Rome Authors. See NOTICE and third_party/licenses/TRELLIS.2-MIT.txt.

from typing import *

CONV = "flex_gemm"
DEBUG = False
ATTN = "flash_attn"


def __from_env():
    import os

    global CONV
    global DEBUG
    global ATTN

    env_sparse_conv_backend = os.environ.get("SPARSE_CONV_BACKEND")
    env_sparse_debug = os.environ.get("SPARSE_DEBUG")
    env_sparse_attn_backend = os.environ.get("SPARSE_ATTN_BACKEND")
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get("ATTN_BACKEND")

    if env_sparse_conv_backend is not None and env_sparse_conv_backend in [
        "none",
        "spconv",
        "torchsparse",
        "flex_gemm",
    ]:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == "1"
    if env_sparse_attn_backend is not None and env_sparse_attn_backend in ["xformers", "flash_attn", "flash_attn_3"]:
        ATTN = env_sparse_attn_backend

    if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0":  # rank 0 only under DDP
        print(f"[SPARSE] Conv backend: {CONV}; Attention backend: {ATTN}")


__from_env()


def set_conv_backend(backend: Literal["none", "spconv", "torchsparse", "flex_gemm"]):
    global CONV
    CONV = backend


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


def set_attn_backend(backend: Literal["xformers", "flash_attn"]):
    global ATTN
    ATTN = backend
