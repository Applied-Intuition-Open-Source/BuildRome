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

BACKEND = "flash_attn"
DEBUG = False


def __from_env():
    import os

    global BACKEND
    global DEBUG

    env_attn_backend = os.environ.get("ATTN_BACKEND")
    env_attn_debug = os.environ.get("ATTN_DEBUG")

    if env_attn_backend is not None and env_attn_backend in ["xformers", "flash_attn", "flash_attn_3", "sdpa", "naive"]:
        BACKEND = env_attn_backend
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == "1"

    if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0":  # rank 0 only under DDP
        print(f"[ATTENTION] Using backend: {BACKEND}")


__from_env()


def set_backend(backend: Literal["xformers", "flash_attn"]):
    global BACKEND
    BACKEND = backend


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
