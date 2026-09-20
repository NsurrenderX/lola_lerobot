import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lerobot.policies.lola_v07.forward_optimizations import DiTCUDAGraph, enable_batched_vision_sdpa


class ForwardOptimizationTests(unittest.TestCase):
    def test_vision_outputs_and_gradients(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention

        config = Qwen3VLVisionConfig(hidden_size=64, num_heads=4)
        config._attn_implementation = "sdpa"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        original = Qwen3VLVisionAttention(config).to(device)
        candidate = copy.deepcopy(original)
        keys = set(candidate.state_dict())
        self.assertEqual(enable_batched_vision_sdpa(SimpleNamespace(visual=candidate)), 1)
        self.assertEqual(set(candidate.state_dict()), keys)
        for lengths in ([8, 8, 8], [8, 4, 8, 4, 4], [3, 8, 4]):
            hidden = torch.randn(sum(lengths), 64, device=device, requires_grad=True)
            other = hidden.detach().clone().requires_grad_()
            boundaries = torch.tensor([0] + lengths, device=device, dtype=torch.int32).cumsum(0)
            angle = torch.randn(sum(lengths), 16, device=device)
            embeddings = (angle.cos(), angle.sin())
            expected = original(hidden, boundaries, embeddings)
            with patch("lerobot.policies.lola_v07.forward_optimizations.functional.scaled_dot_product_attention",
                       wraps=torch.nn.functional.scaled_dot_product_attention) as sdpa:
                actual = candidate(other, boundaries, embeddings)
            self.assertEqual(sdpa.call_count, len(set(lengths)))
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            expected.square().sum().backward()
            actual.square().sum().backward()
            torch.testing.assert_close(other.grad, hidden.grad, rtol=1e-4, atol=1e-5)
            for baseline_parameter, parameter in zip(original.parameters(), candidate.parameters()):
                torch.testing.assert_close(parameter.grad, baseline_parameter.grad, rtol=1e-4, atol=1e-5)
            original.zero_grad()
            candidate.zero_grad()

    def test_vision_forwarded_metadata(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        config = Qwen3VLVisionConfig(
            hidden_size=64, intermediate_size=128, num_heads=4, depth=2,
            out_hidden_size=64, deepstack_visual_indexes=[], num_position_embeddings=16,
            patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
        )
        config._attn_implementation = "sdpa"
        original = Qwen3VLVisionModel(config).train()
        candidate = copy.deepcopy(original)
        enable_batched_vision_sdpa(SimpleNamespace(visual=candidate))
        grid = torch.tensor([[1, 4, 4], [1, 2, 2], [1, 4, 4], [1, 2, 2]])
        for output_hidden_states in (True, False):
            with self.subTest(output_hidden_states=output_hidden_states):
                inputs = torch.randn(40, 12, requires_grad=True)
                other = inputs.detach().clone().requires_grad_()
                expected = original(inputs, grid, output_hidden_states=output_hidden_states)
                with patch("lerobot.policies.lola_v07.forward_optimizations.functional.scaled_dot_product_attention",
                           wraps=torch.nn.functional.scaled_dot_product_attention) as sdpa:
                    actual = candidate(other, grid, output_hidden_states=output_hidden_states)
                self.assertEqual(sdpa.call_count, 4)
                torch.testing.assert_close(actual.pooler_output, expected.pooler_output, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(actual.hidden_states, expected.hidden_states, rtol=1e-5, atol=1e-6)
                expected.pooler_output.square().sum().backward()
                actual.pooler_output.square().sum().backward()
                torch.testing.assert_close(other.grad, inputs.grad, rtol=1e-4, atol=1e-5)
                for baseline_parameter, parameter in zip(original.parameters(), candidate.parameters()):
                    torch.testing.assert_close(parameter.grad, baseline_parameter.grad, rtol=1e-4, atol=1e-5)
                original.zero_grad(set_to_none=True)
                candidate.zero_grad(set_to_none=True)

    def test_cosmos_policy_wrapper_outputs_gradients_and_calls(self):
        from transformers.models.cosmos3_omni.configuration_cosmos3_omni import Cosmos3OmniConfig
        from transformers.models.cosmos3_omni.modeling_cosmos3_omni import Cosmos3OmniModel
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
        from lerobot.policies.lola_v07.modeling_lola_v07 import LoLAV07Policy

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        config = Cosmos3OmniConfig(
            vision_config=Qwen3VLVisionConfig(
                hidden_size=64, intermediate_size=128, num_heads=4, depth=2,
                out_hidden_size=64, deepstack_visual_indexes=[], num_position_embeddings=16,
                patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
            ),
            text_config=Qwen3VLTextConfig(
                vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, use_cache=False,
                rope_parameters={"rope_type": "default", "mrope_section": [2, 3, 3]},
            ),
            image_token_id=4, video_token_id=5, vision_start_token_id=9, vision_end_token_id=10,
        )
        config._attn_implementation = "sdpa"
        input_ids = torch.tensor([[1, 4, 4, 4, 4, 2, 4, 2, 4, 4, 4, 4, 2, 4, 3]], device=device)
        batch = dict(input_ids=input_ids, mm_token_type_ids=(input_ids == 4).long(),
                     image_grid_thw=torch.tensor([[1, 4, 4], [1, 2, 2], [1, 4, 4], [1, 2, 2]], device=device))
        for mode, retained in (("output_hidden_states", 0), ("output_hidden_states", 1), ("hook", None)):
            with self.subTest(mode=mode, retained=retained):
                torch.manual_seed(0)
                original = Cosmos3OmniModel(copy.deepcopy(config)).to(device).train()
                candidate = copy.deepcopy(original)
                keys = set(candidate.state_dict())
                self.assertEqual(enable_batched_vision_sdpa(candidate), 2)
                self.assertEqual(enable_batched_vision_sdpa(candidate), 2)
                self.assertEqual(set(candidate.state_dict()), keys)
                policies = []
                for model in (original, candidate):
                    policy = SimpleNamespace(
                        config=SimpleNamespace(train_vlm=True, vlm_extract_layers=[2],
                                               vision_gradient_checkpointing=True, vision_no_checkpoint_layers=retained),
                        vlm=model, _vlm_forward_mode=mode, _captured_hidden_states={},
                        _hook_handles=[], _in_vlm_forward=False,
                    )
                    if retained is not None:
                        LoLAV07Policy.enable_vlm_gradient_checkpointing(policy)
                    if mode == "hook":
                        LoLAV07Policy._register_vlm_hooks(policy)
                    policies.append(policy)
                inputs = torch.randn(40, 12, device=device, requires_grad=True)
                other = inputs.detach().clone().requires_grad_()
                weights = torch.randn(1, input_ids.shape[1], 64, device=device)
                outputs = []
                for policy, pixels, grouped in zip(policies, (inputs, other), (False, True)):
                    with patch("lerobot.policies.lola_v07.forward_optimizations.functional.scaled_dot_product_attention",
                               wraps=torch.nn.functional.scaled_dot_product_attention) as sdpa:
                        features, actual_ids = LoLAV07Policy.prepare_vlm_inputs(policy, dict(batch, pixel_values=pixels))
                        torch.testing.assert_close(actual_ids, input_ids, rtol=0, atol=0)
                        visual_calls = 4 if grouped else 8
                        self.assertEqual(sdpa.call_count, visual_calls + 2)
                        outputs.append(features[2])
                        (features[2] * weights).sum().backward()
                        recompute = 0 if retained is None else (2 - retained) * (2 if grouped else 4) + 2
                        self.assertEqual(sdpa.call_count, visual_calls + 2 + recompute)
                torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(other.grad, inputs.grad, rtol=1e-4, atol=1e-5)
                for (name, baseline_parameter), (other_name, parameter) in zip(
                        original.named_parameters(), candidate.named_parameters()):
                    self.assertEqual(name, other_name)
                    self.assertEqual(parameter.grad is None, baseline_parameter.grad is None, name)
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        torch.testing.assert_close(parameter.grad, baseline_parameter.grad, rtol=1e-4, atol=1e-5)
                for policy in policies:
                    LoLAV07Policy._remove_vlm_hooks(policy)

    def test_vision_unknown_kwargs_preserve_original_fallback(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention

        config = Qwen3VLVisionConfig(hidden_size=64, num_heads=4)
        config._attn_implementation = "sdpa"
        attention = Qwen3VLVisionAttention(config)
        enable_batched_vision_sdpa(SimpleNamespace(visual=attention))
        inputs = torch.randn(8, 64)
        boundaries = torch.tensor([0, 4, 8], dtype=torch.int32)
        angle = torch.randn(8, 16)
        embeddings = (angle.cos(), angle.sin())
        expected = torch.randn_like(inputs)
        for kwargs in (dict(attention_mask=torch.ones(4, 4)), dict(output_attentions=True), dict(custom_option=7)):
            with self.subTest(keys=tuple(kwargs)), \
                    patch.object(attention, "_lola_original_forward", return_value=expected) as original:
                actual = attention(inputs, boundaries, embeddings, output_hidden_states=False, **kwargs)
                self.assertIs(actual, expected)
                original.assert_called_once_with(inputs, boundaries, embeddings, output_hidden_states=False, **kwargs)

    def test_selective_vision_checkpointing(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        from lerobot.policies.lola_v07.configuration_lola_v07 import LoLAV07Config
        from lerobot.policies.lola_v07.modeling_lola_v07 import LoLAV07Policy

        config = Qwen3VLVisionConfig(
            hidden_size=64, intermediate_size=128, num_heads=4, depth=2,
            out_hidden_size=64, deepstack_visual_indexes=[], num_position_embeddings=16,
        )
        visual = Qwen3VLVisionModel(config)
        language = SimpleNamespace(gradient_checkpointing=False)

        def enable():
            visual.gradient_checkpointing_enable()
            language.gradient_checkpointing = True

        policy = SimpleNamespace(
            config=LoLAV07Config(),
            vlm=SimpleNamespace(visual=visual, language_model=language, gradient_checkpointing_enable=enable),
        )
        keys = set(visual.state_dict())
        for enabled in (True, False, False, True):
            policy.config.vision_gradient_checkpointing = enabled
            LoLAV07Policy.enable_vlm_gradient_checkpointing(policy)
            flags = [module.gradient_checkpointing for module in visual.modules()
                     if hasattr(module, "gradient_checkpointing")]
            self.assertTrue(flags)
            self.assertTrue(all(flag == enabled for flag in flags))
            self.assertTrue(language.gradient_checkpointing)
            self.assertEqual(set(visual.state_dict()), keys)

    def test_vision_recomputation_and_full_gradients(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        from lerobot.policies.lola_v07.modeling_lola_v07 import LoLAV07Policy

        config = Qwen3VLVisionConfig(
            hidden_size=64, intermediate_size=128, num_heads=4, depth=2,
            out_hidden_size=64, deepstack_visual_indexes=[], num_position_embeddings=16,
            patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
        )
        config._attn_implementation = "sdpa"
        original = Qwen3VLVisionModel(config).train()
        candidate = copy.deepcopy(original)
        original.gradient_checkpointing_enable()
        policy = SimpleNamespace(
            config=SimpleNamespace(vision_gradient_checkpointing=False),
            vlm=SimpleNamespace(visual=candidate, gradient_checkpointing_enable=candidate.gradient_checkpointing_enable),
        )
        LoLAV07Policy.enable_vlm_gradient_checkpointing(policy)
        enable_batched_vision_sdpa(policy.vlm)
        calls = {"original": 0, "candidate": 0}

        def hook(name):
            def count(module, arguments):
                calls[name] += 1
            return count

        original.blocks[0].register_forward_pre_hook(hook("original"))
        candidate.blocks[0].register_forward_pre_hook(hook("candidate"))
        inputs = torch.randn(20, 12, requires_grad=True)
        other = inputs.detach().clone().requires_grad_()
        grid = torch.tensor([[1, 4, 4], [1, 2, 2]])
        expected = original(inputs, grid).pooler_output
        actual = candidate(other, grid).pooler_output
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().sum().backward()
        actual.square().sum().backward()
        self.assertEqual(calls, {"original": 2, "candidate": 1})
        torch.testing.assert_close(other.grad, inputs.grad, rtol=1e-4, atol=1e-5)
        for baseline_parameter, parameter in zip(original.parameters(), candidate.parameters()):
            torch.testing.assert_close(parameter.grad, baseline_parameter.grad, rtol=1e-4, atol=1e-5)

        policy.config = SimpleNamespace(vision_gradient_checkpointing=True, vision_no_checkpoint_layers=1)
        LoLAV07Policy.enable_vlm_gradient_checkpointing(policy)
        calls["candidate"] = 0
        calls["last_candidate"] = 0
        candidate.blocks[-1].register_forward_pre_hook(hook("last_candidate"))
        candidate.zero_grad(set_to_none=True)
        partial = candidate(inputs.detach().clone().requires_grad_(), grid).pooler_output
        torch.testing.assert_close(partial, expected, rtol=1e-5, atol=1e-6)
        partial.square().sum().backward()
        self.assertEqual(calls["candidate"], 2)
        self.assertEqual(calls["last_candidate"], 1)
        for baseline_parameter, parameter in zip(original.parameters(), candidate.parameters()):
            torch.testing.assert_close(parameter.grad, baseline_parameter.grad, rtol=1e-4, atol=1e-5)
        policy.config.vision_no_checkpoint_layers = 3
        with self.assertRaisesRegex(ValueError, "between 0 and 2"):
            LoLAV07Policy.enable_vlm_gradient_checkpointing(policy)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_real_dit_graph(self):
        from lerobot.policies.lola.modeling_lola import LoLADiT
        from lerobot.policies.lola_v07.configuration_lola_v07 import LoLAV07Config

        config = LoLAV07Config(
            dit_hidden_size=64, dit_num_heads=4, dit_double_layers=1, dit_single_layers=1,
            action_bottleneck_dim=16, grip_bottleneck_dim=8,
            state_bottleneck_dim=16, state_grip_bottleneck_dim=8,
        )
        module = LoLADiT(config).to(device="cuda", dtype=torch.bfloat16).eval()
        graph = DiTCUDAGraph(module)
        arguments = dict(
            target_actions=torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16),
            hist_actions=torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16),
            vlm_features=torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16),
            empty_emb=torch.randn(1, 64, device="cuda", dtype=torch.bfloat16),
            timestep=torch.ones(1, device="cuda"), return_chunks=True,
            joint_attention_kwargs=dict(attention_mask=torch.ones(1, 16, device="cuda", dtype=torch.bool)),
        )
        with torch.no_grad():
            for timestamp in (1.0, 0.5):
                arguments["timestep"].fill_(timestamp)
                expected = module(**arguments)
                actual = graph(**arguments)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(graph.captures, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_graph_replay_and_training_fallback(self):
        module = torch.nn.Linear(16, 8).cuda().eval()
        graph = DiTCUDAGraph(module)
        with torch.no_grad():
            for batch in (2, 2, 3):
                inputs = torch.randn(batch, 16, device="cuda")
                torch.testing.assert_close(graph(inputs), module(inputs), rtol=0, atol=0)
            self.assertEqual(graph.captures, 2)
        module.train()
        graph(torch.randn(2, 16, device="cuda")).sum().backward()
        self.assertIsNone(graph.graph)
        self.assertIsNotNone(module.weight.grad)


if __name__ == "__main__":
    unittest.main()