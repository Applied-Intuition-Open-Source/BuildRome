# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.

from build_rome import models

from ..modules.lora import apply_lora


def build_single_model(model_cfg):

    model = getattr(models, model_cfg.name)(**model_cfg.args)

    if hasattr(model_cfg, "lora") and model_cfg.lora is not None:
        model = apply_lora(model, model_cfg.lora)

    return model
