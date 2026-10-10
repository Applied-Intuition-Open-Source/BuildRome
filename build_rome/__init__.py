# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# Portions of this file are derived from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2),
# Copyright (c) Microsoft Corporation, licensed under the MIT License, and were
# modified by The Build Rome Authors. See NOTICE and third_party/licenses/TRELLIS.2-MIT.txt.

from . import models, modules, pipelines, renderers, representations, utils


def __getattr__(name):
    # The inference pipeline lives next to the package (inference/pipeline.py)
    # and pulls in the model loaders, so it is resolved lazily.
    if name in ("ScenePipeline", "SceneOutput"):
        from inference import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
