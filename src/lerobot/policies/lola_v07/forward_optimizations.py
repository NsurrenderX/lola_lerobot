"""Opt-in forward optimizations that preserve checkpoint parameter names."""

from types import MethodType

import torch
import torch.nn.functional as functional
from torch.utils._pytree import tree_flatten, tree_unflatten


def enable_batched_vision_sdpa(vlm):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionAttention,
        apply_rotary_pos_emb_vision,
    )

    def forward(attention, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        original = attention._lola_original_forward
        if (attention.config._attn_implementation != "sdpa"
            or position_embeddings is None
            or any(name != "output_hidden_states" for name in kwargs)
                or (attention.training and attention.attention_dropout != 0)):
            return original(hidden_states, cu_seqlens, position_embeddings, **kwargs)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if not lengths or min(lengths) <= 0:
            return original(hidden_states, cu_seqlens, position_embeddings)
        sequence_length = hidden_states.shape[0]
        query, key, value = (
            attention.qkv(hidden_states)
            .reshape(sequence_length, 3, attention.num_heads, -1)
            .permute(1, 0, 2, 3).unbind(0)
        )
        query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
        if len(set(lengths)) == 1:
            query, key, value = [
                tensor.reshape(len(lengths), lengths[0], attention.num_heads, -1).transpose(1, 2)
                for tensor in (query, key, value)
            ]
            output = functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False, scale=attention.scaling,
            ).transpose(1, 2).reshape(sequence_length, -1).contiguous()
        else:
            groups = {}
            for index, length in enumerate(lengths):
                groups.setdefault(length, []).append(index)
            parts = [tensor.split(lengths, dim=0) for tensor in (query, key, value)]
            outputs = [None] * len(lengths)
            for indices in groups.values():
                grouped_query, grouped_key, grouped_value = [
                    torch.stack([chunks[index] for index in indices]).transpose(1, 2)
                    for chunks in parts
                ]
                grouped_output = functional.scaled_dot_product_attention(
                    grouped_query, grouped_key, grouped_value,
                    dropout_p=0.0, is_causal=False, scale=attention.scaling,
                ).transpose(1, 2)
                for index, output in zip(indices, grouped_output.unbind(0)):
                    outputs[index] = output
            output = torch.cat(outputs, dim=0).reshape(sequence_length, -1).contiguous()
        return attention.proj(output)

    count = 0
    for attention in vlm.visual.modules():
        if isinstance(attention, Qwen3VLVisionAttention):
            if not hasattr(attention, "_lola_original_forward"):
                attention._lola_original_forward = attention.forward
                attention.forward = MethodType(forward, attention)
            count += 1
    if not count:
        raise ValueError("Batched vision SDPA requires a Qwen3-VL vision tower")
    return count


class DiTCUDAGraph:
    """Single-shape inference cache; clear after changing weights or module placement."""

    def __init__(self, module):
        if any(hasattr(parameter, "ds_id") for parameter in module.parameters()):
            raise ValueError("DiT CUDA Graph is not supported on ZeRO-managed parameters")
        self.module = module
        self.original = module.forward
        self.captures = 0
        self.clear()

    def clear(self):
        self.key = None
        self.graph = None
        self.static_inputs = None
        self.output = None

    def __call__(self, *args, **kwargs):
        if self.module.training or torch.is_grad_enabled():
            self.clear()
            return self.original(*args, **kwargs)
        flat, specification = tree_flatten((args, kwargs))
        tensors = [value for value in flat if isinstance(value, torch.Tensor)]
        devices = {value.device for value in tensors}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            return self.original(*args, **kwargs)
        device = tensors[0].device
        signature = tuple(
            (tuple(value.shape), value.dtype, value.device, tuple(value.stride()))
            if isinstance(value, torch.Tensor) else (type(value), value)
            for value in flat
        )
        key = (specification, signature, torch.is_autocast_enabled("cuda"),
               torch.get_autocast_dtype("cuda"))
        if key != self.key:
            self.clear()
            with torch.cuda.device(device):
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    self.static_inputs = [
                        value.clone() if isinstance(value, torch.Tensor) else value for value in flat
                    ]
                    static_args, static_kwargs = tree_unflatten(self.static_inputs, specification)
                    for warmup_step in range(3):
                        self.original(*static_args, **static_kwargs)
                stream.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.output = self.original(*static_args, **static_kwargs)
                torch.cuda.current_stream(device).wait_stream(stream)
            self.key = key
            self.captures += 1
        for destination, source in zip(self.static_inputs, flat):
            if isinstance(source, torch.Tensor):
                destination.copy_(source)
        self.graph.replay()
        return self.output.clone()