#!/usr/bin/env python

# Copyright 2025 Lola Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""VLM backbone registry for LoLA policies.

Centralizes all backbone-specific details (model class, default dims, token IDs,
FSDP wrap classes, forward-mode capabilities) so that lola / lola_v07 (and future
variants) can switch VLM backbones via ``LoLAConfig.vlm_backbone`` without
duplicating modeling or training-pipeline code.

Currently supported backbones:
- ``qwen3_5``      : Qwen3.5-4B / Qwen3.5-2B (original LoLA backbone)
- ``cosmos3_nano`` : NVIDIA Cosmos3-Nano Reasoner. Architecturally it is a
  Qwen3-VL (text: qwen3_vl_text, hidden 4096, 36 layers; vision:
  Qwen3VLVisionModel with deepstack injections at layers 8/16/24) with a
  Qwen2 tokenizer (eos = <|im_end|> = 151645). The unified checkpoint also
  carries the video-generator / sound / action towers; transformers>=5.14
  ships an offline weight-conversion that drops those on load.
"""

import logging
from dataclasses import dataclass
from typing import Callable

import torch

logger = logging.getLogger(__name__)


def _load_qwen3_5_model(path_or_name: str, dtype: torch.dtype, local_files_only: bool):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    kwargs = dict(
        torch_dtype=dtype,
        device_map=None,  # 不自动分配，让分布式策略管理
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if local_files_only:
        kwargs["local_files_only"] = True
    return Qwen3_5Model.from_pretrained(path_or_name, **kwargs)


def _load_cosmos3_nano_model(path_or_name: str, dtype: torch.dtype, local_files_only: bool):
    # Cosmos3OmniModel = Reasoner (vision tower + Qwen3-VL text LM) WITHOUT lm_head,
    # mirroring the Qwen3_5Model usage (vs Qwen3_5ForConditionalGeneration).
    # The unified checkpoint's Generator/sound/action towers are dropped on load via
    # Cosmos3OmniPreTrainedModel._keys_to_ignore_on_load_unexpected; lm_head.weight
    # shows up as an (ignored) unexpected key.
    from transformers.models.cosmos3_omni.modeling_cosmos3_omni import Cosmos3OmniModel

    kwargs = dict(
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if local_files_only:
        kwargs["local_files_only"] = True
    return Cosmos3OmniModel.from_pretrained(path_or_name, **kwargs)


@dataclass(frozen=True)
class VLMBackboneSpec:
    """Everything LoLA needs to know about a VLM backbone."""

    name: str
    default_model_name: str           # HF hub id; used when neither vlm_path nor vlm_model_name is set
    default_hidden_size: int          # VLM hidden size (drives LolaVLMFeatureExtractor concat dim)
    default_empty_token_id: int       # token appended as LoLA "empty token" (backbone eos)
    default_extract_layers: tuple     # hidden-state extraction layers (hidden_states index, 1-based)
    num_hidden_layers: int            # informational: total text layers before truncation
    supports_split_forward: bool      # whether _vlm_split_forward (Qwen3.5 internals) is available
    loader: Callable                  # (path_or_name, dtype, local_files_only) -> nn.Module

    def load_model(self, path_or_name: str, dtype: torch.dtype = torch.bfloat16,
                   local_files_only: bool = False):
        return self.loader(path_or_name, dtype=dtype, local_files_only=local_files_only)

    def get_fsdp_wrap_classes(self) -> tuple:
        """Transformer layer classes for FSDP transformer_auto_wrap_policy."""
        if self.name == "qwen3_5":
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DecoderLayer,
                Qwen3_5VisionBlock,
            )
            return (Qwen3_5DecoderLayer, Qwen3_5VisionBlock)
        if self.name == "cosmos3_nano":
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLTextDecoderLayer,
                Qwen3VLVisionBlock,
            )
            return (Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock)
        raise ValueError(f"No FSDP wrap classes registered for backbone '{self.name}'")


VLM_BACKBONES: dict[str, VLMBackboneSpec] = {
    "qwen3_5": VLMBackboneSpec(
        name="qwen3_5",
        default_model_name="Qwen/Qwen3.5-4B",
        default_hidden_size=2560,
        default_empty_token_id=248044,  # Qwen3.5 eos_token
        default_extract_layers=(8, 16, 24),
        num_hidden_layers=33,
        supports_split_forward=True,
        loader=_load_qwen3_5_model,
    ),
    "cosmos3_nano": VLMBackboneSpec(
        name="cosmos3_nano",
        default_model_name="nvidia/Cosmos3-Nano",
        default_hidden_size=4096,
        default_empty_token_id=151645,  # <|im_end|> (eos of the Qwen2-style chat template)
        # (8, 16, 24) matches the deepstack visual injection rhythm; truncating the
        # 36-layer text tower at layer 24 keeps all deepstack injection points.
        default_extract_layers=(8, 16, 24),
        num_hidden_layers=36,
        # _vlm_split_forward inlines Qwen3.5 vision internals (fast_pos_embed_interpolate
        # etc.) which do not exist in Qwen3VLVisionModel (deepstack_merger_list, different
        # pos-embed path). Use "hook" / "output_hidden_states" modes instead.
        supports_split_forward=False,
        loader=_load_cosmos3_nano_model,
    ),
}


def get_vlm_backbone(name: str) -> VLMBackboneSpec:
    if name not in VLM_BACKBONES:
        raise ValueError(
            f"Invalid vlm_backbone: '{name}'. Available: {sorted(VLM_BACKBONES.keys())}"
        )
    return VLM_BACKBONES[name]
