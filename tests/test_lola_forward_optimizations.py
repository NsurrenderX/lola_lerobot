import copy
import unittest
from types import SimpleNamespace

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
        for lengths in ([8, 8, 8], [8, 4, 8]):
            hidden = torch.randn(sum(lengths), 64, device=device, requires_grad=True)
            other = hidden.detach().clone().requires_grad_()
            boundaries = torch.tensor([0] + lengths, device=device, dtype=torch.int32).cumsum(0)
            angle = torch.randn(sum(lengths), 16, device=device)
            embeddings = (angle.cos(), angle.sin())
            expected = original(hidden, boundaries, embeddings)
            actual = candidate(other, boundaries, embeddings)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            expected.square().sum().backward()
            actual.square().sum().backward()
            torch.testing.assert_close(other.grad, hidden.grad, rtol=1e-4, atol=1e-5)
            for baseline_parameter, parameter in zip(original.parameters(), candidate.parameters()):
                torch.testing.assert_close(parameter.grad, baseline_parameter.grad, rtol=1e-4, atol=1e-5)
            original.zero_grad()
            candidate.zero_grad()

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