# Copyright 2026 The Build Rome Authors.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International
# License (CC BY-NC 4.0). See the LICENSE file in the root of this repository or
# https://creativecommons.org/licenses/by-nc/4.0/.
#
# Portions of this file are derived from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2),
# Copyright (c) Microsoft Corporation, licensed under the MIT License, and were
# modified by The Build Rome Authors. See NOTICE and third_party/licenses/TRELLIS.2-MIT.txt.

class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_rescale=0.0,
                         cfg_token_weight=None, **kwargs):
        if guidance_strength == 1:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        elif guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        else:
            pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
            pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            if cfg_token_weight is not None:
                # Per-token CFG (sparse chunks): tokens with corroborating
                # geometric evidence (weight 1) get the full strength, tokens
                # without (weight 0) fall back to the unguided prediction —
                # the image channel is only amplified where the depth pack
                # can vouch for it.
                w_tok = 1.0 + (guidance_strength - 1.0) * cfg_token_weight.to(pred_pos.feats.dtype)
                pred = pred_pos.replace(
                    pred_neg.feats + w_tok[:, None] * (pred_pos.feats - pred_neg.feats))
            else:
                pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg

            # CFG rescale
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)

            return pred
