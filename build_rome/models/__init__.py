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

__attributes = {
    # Sparse Structure
    "SparseStructureEncoder": "sparse_structure_vae",
    "SparseStructureDecoder": "sparse_structure_vae",
    "SparseStructureFlowModel": "sparse_structure_flow",
    "LiftedSSFlowModel": "lifted_ss_flow",
    # SLat Generation
    "SLatFlowModel": "structured_latent_flow",
    "ElasticSLatFlowModel": "structured_latent_flow",
    "LiftedSLatFlowModel": "lifted_slat_flow",
    # SC-VAEs
    "SparseUnetVaeEncoder": "sc_vaes.sparse_unet_vae",
    "SparseUnetVaeDecoder": "sc_vaes.sparse_unet_vae",
    "FlexiDualGridVaeEncoder": "sc_vaes.fdg_vae",
    "FlexiDualGridVaeDecoder": "sc_vaes.fdg_vae",
}

__submodules = []

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


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import json
    import os

    from safetensors.torch import load_file

    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        path_parts = path.split("/")
        repo_id = f"{path_parts[0]}/{path_parts[1]}"
        model_name = "/".join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, "r") as f:
        config = json.load(f)
    model = __getattr__(config["name"])(**config["args"], **kwargs)
    model.load_state_dict(load_file(model_file), strict=False)

    return model


# For Pylance
if __name__ == "__main__":
    from .sc_vaes.fdg_vae import FlexiDualGridVaeDecoder, FlexiDualGridVaeEncoder
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder, SparseUnetVaeEncoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .sparse_structure_vae import SparseStructureDecoder, SparseStructureEncoder
    from .structured_latent_flow import ElasticSLatFlowModel, SLatFlowModel
    from .lifted_slat_flow import LiftedSLatFlowModel
    from .lifted_ss_flow import LiftedSSFlowModel
