import argparse
import ast
import fnmatch
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from lerobot.scripts.profile_lola_v07 import (
    BenchComplete, BenchRecorder, ProfileLoader, counter_deltas, parse_options, summarize,
    assess_memory_budget, localize_main, resolve_training_arguments, run_profile_child,
    stage_checkpoint_config, validate_checkpoint_tag, configure_profile_deepspeed,
    profile_deepspeed_initialization, profile_hpz_topology, configure_unfreeze_profile,
    profile_engine_lifecycle,
)


class ProfileBenchTests(unittest.TestCase):
    def test_unfreeze_profile_options(self):
        options, extra = parse_options([
            "--training-config", "config.json", "--output", "/tmp/unfreeze-profile",
            "--unfreeze-after", "590", "--", "--resume", "",
        ])
        self.assertEqual(options.unfreeze_after, 590)
        self.assertEqual(options.memory_budget_fraction, 0.9)
        self.assertEqual(extra, ["--resume", ""])
        defaults = dict(strategy="deepspeed", deepspeed_zero_stage=3, train_vlm=True,
                        vlm_unfreeze_v_loss_threshold=0.6, resume="")
        configure_unfreeze_profile(options, SimpleNamespace(**defaults))
        for changes in (dict(strategy="ddp"), dict(deepspeed_zero_stage=2),
                        dict(train_vlm=False), dict(vlm_unfreeze_v_loss_threshold=0),
                        dict(resume="/checkpoint")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                configure_unfreeze_profile(options, SimpleNamespace(**(defaults | changes)))
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            parse_options(["--training-config", "config.json", "--output", "/tmp/new",
                           "--unfreeze-after", "0"])

    def test_hpz_profile_config_isolated(self):
        from lerobot.scripts import train_lola_v07_azure as training

        original = training.get_deepspeed_config(batch_size=32, world_size=16, zero_stage=3,
                                                 allgather_bucket_size=5e8)
        original["zero_optimization"]["zero_hpz_partition_size"] = 4
        before = json.dumps(original, sort_keys=True)
        configs = []
        for size in (1, 8):
            options, arguments = parse_options([
                "--training-config", "config.json", "--output", "/tmp/hpz-profile",
                "--validated-optimizations", "--zero-hpz-partition-size", str(size),
            ])
            self.assertEqual(options.zero_hpz_partition_size, size)
            self.assertNotIn("--zero-hpz-partition-size", arguments)
            self.assertEqual(options.memory_budget_fraction, 0.9)
            config = configure_profile_deepspeed(original, size, 16, 8)
            self.assertEqual(config["zero_optimization"]["zero_hpz_partition_size"], size)
            self.assertEqual(config["train_batch_size"], 512)
            configs.append(config)
        self.assertEqual(json.dumps(original, sort_keys=True), before)
        self.assertEqual(configure_profile_deepspeed(original, None, 1, 1), original)
        configs[0]["zero_optimization"]["zero_hpz_partition_size"] = 8
        self.assertEqual(configs[0], configs[1])

    def test_hpz_profile_rejects_confounders(self):
        from lerobot.scripts import train_lola_v07_azure as training

        for overrides, message in (
                ({"stage": 2}, "BF16 ZeRO-3"),
                ({"mics_shard_size": 8}, "MiCS"),
                ({"offload_optimizer": {"device": "cpu"}}, "offload"),
                ({"offload_param": {"device": "nvme"}}, "offload"),
                *[({name: True}, "quantization") for name in (
                    "zero_quantized_weights", "zero_quantized_nontrainable_weights", "zero_quantized_gradients")]):
            config = training.get_deepspeed_config(zero_stage=3)
            config["zero_optimization"].update(overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                configure_profile_deepspeed(config, 8, 16, 8)
        config = training.get_deepspeed_config(zero_stage=3)
        for world_size, local_world_size in ((8, 8), (16, 4), (12, 8)):
            with self.subTest(world_size=world_size, local_world_size=local_world_size), \
                    self.assertRaisesRegex(ValueError, "multiple nodes with 8 ranks"):
                configure_profile_deepspeed(config, 8, world_size, local_world_size)

    def test_hpz_profile_actual_trainer_initialization(self):
        import deepspeed
        from lerobot.scripts import train_lola_v07_azure as training

        with tempfile.TemporaryDirectory() as folder:
            custom = Path(folder) / "custom.json"
            config = training.get_deepspeed_config(zero_stage=3)["zero_optimization"]
            config.update(zero_hpz_partition_size=4, stage3_max_live_parameters=12345)
            custom.write_text(json.dumps({"zero_optimization": config}))
            for size in (None, 1, 8):
                engine = SimpleNamespace()
                trainer = SimpleNamespace(
                    learning_rate=2.5e-5, weight_decay=0.01, gradient_clip_val=1.0, train_vlm=True,
                    batch_size=32, world_size=16, deepspeed_reduce_bucket_size=5e8,
                    deepspeed_allgather_bucket_size=5e8, deepspeed_zero_stage=3,
                    deepspeed_config_path=str(custom), policy=torch.nn.Linear(2, 2),
                    config=SimpleNamespace(), vlm_lr=1e-5, _vlm_delayed_unfreeze=False,
                    _configure_deepspeed_checkpointing=lambda: None,
                )
                with patch.object(deepspeed, "initialize", return_value=(engine, None, None, None)) as initialize, \
                        patch.object(training, "build_lola_v07_param_groups", return_value=[]), \
                        patch.dict(os.environ, WORLD_SIZE="16", LOCAL_WORLD_SIZE="8"), \
                        profile_deepspeed_initialization(SimpleNamespace(zero_hpz_partition_size=size)):
                    training.LoLAV07Trainer._setup_deepspeed(trainer)
                actual = initialize.call_args.kwargs["config"]
                self.assertEqual(actual["zero_optimization"]["zero_hpz_partition_size"], 4 if size is None else size)
                self.assertEqual(actual["zero_optimization"]["stage3_max_live_parameters"], 12345)
                self.assertEqual(actual["train_batch_size"], 512)
                self.assertEqual(engine._lola_profile_deepspeed_config, actual)
                self.assertIsNot(engine._lola_profile_deepspeed_config, actual)
                self.assertEqual(json.loads(custom.read_text())["zero_optimization"]["zero_hpz_partition_size"], 4)

    def test_hpz_profile_runtime_topology(self):
        for size in (1, 8):
            for invalid in (None, "size", "group", "host", "same-node", "local-rank"):
                if size == 1 and invalid in ("host", "same-node", "local-rank"):
                    continue
                records = [dict(rank=rank, hostname=f"node{rank // 8}", local_rank=rank % 8,
                                hpz_partition_size=size,
                                hpz_group_ranks=list(range(rank // 8 * 8, rank // 8 * 8 + 8)) if size == 8 else [])
                           for rank in range(16)]
                if invalid == "size":
                    records[0]["hpz_partition_size"] = 4
                elif invalid == "group":
                    records[0]["hpz_group_ranks"] = [0, 8]
                elif invalid == "host":
                    records[0]["hostname"] = "foreign"
                elif invalid == "local-rank":
                    records[0]["local_rank"] = 7
                elif invalid == "same-node":
                    for record in records:
                        record["hostname"] = "node0"
                policy = torch.nn.Linear(2, 2)
                next(policy.parameters()).ds_zero_param_process_group = object() if size == 8 else None
                trainer = SimpleNamespace(policy=policy, world_rank=0, local_rank=0, world_size=16,
                                          model=SimpleNamespace(optimizer=SimpleNamespace(zero_hpz_partition_size=size)))
                with self.subTest(size=size, invalid=invalid), \
                        patch("torch.distributed.get_process_group_ranks", return_value=list(range(8))), \
                        patch("torch.distributed.all_gather_object", side_effect=lambda output, value: output.__setitem__(slice(None), records)):
                    if invalid:
                        with self.assertRaises(ValueError):
                            profile_hpz_topology(trainer, size)
                    else:
                        self.assertEqual(profile_hpz_topology(trainer, size), records)
        self.assertIsNone(profile_hpz_topology(None, None))

    def test_hpz_profile_main_initialization(self):
        import deepspeed
        from lerobot.scripts import profile_lola_v07 as profile
        from lerobot.scripts import train_lola_v07_azure as training

        config = training.get_deepspeed_config(batch_size=32, world_size=16, zero_stage=3)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "training_config.json"
            source.write_text(json.dumps({"training_args": {"strategy": "deepspeed", "deepspeed_zero_stage": 3}}))
            for size in (1, 8):
                command = ["profile_lola_v07.py", "--training-config", str(source), "--output", str(Path(folder) / "output"),
                           "--validated-optimizations", "--zero-hpz-partition-size", str(size)]
                with patch("sys.argv", command), patch.dict(os.environ, WORLD_SIZE="16", LOCAL_WORLD_SIZE="8"), \
                        patch.object(deepspeed, "initialize", return_value=(SimpleNamespace(), None, None, None)) as initialize, \
                        patch.object(training, "main", side_effect=lambda: deepspeed.initialize(config=config)):
                    profile.main()
                    self.assertIs(deepspeed.initialize, initialize)
                self.assertEqual(initialize.call_args.kwargs["config"]["zero_optimization"]["zero_hpz_partition_size"], size)
                self.assertNotIn("zero_hpz_partition_size", config["zero_optimization"])

    def test_abc_grouping_and_zero_retention_options(self):
        override = Path(__file__).resolve().parents[1] / "deepspeed_lola_zero3_c.json"
        snapshot = dict(training_args=dict(
            vision_batched_sdpa=False, deepspeed_config=None,
            gradient_checkpointing=True, train_vlm=True,
        ))
        common = ["--strategy", "deepspeed", "--deepspeed_zero_stage", "3", "--batch_size", "32", "--seed", "0",
                  "--vision_no_checkpoint_layers", "12", "--deepspeed_reduce_bucket_size", "500000000",
                  "--deepspeed_allgather_bucket_size", "500000000"]
        configs = []
        for group in ("A", "B", "C"):
            before = [] if group == "A" else ["--vision-batched-sdpa"]
            after = ["--deepspeed_config", str(override)] if group == "C" else []
            options, trainer_arguments = parse_options([
                "--training-config", "original_training_config.json", "--output", f"/tmp/lola_fixed_{group}",
                "--warmup", "40", "--steps", "200", "--trace-steps", "0", "--memory-budget-fraction", "0.90",
                *before, "--", *common, *after,
            ])
            training, _, args = resolve_training_arguments(snapshot, trainer_arguments)
            self.assertEqual((options.warmup, options.steps, options.trace_steps), (40, 200, 0))
            self.assertEqual(options.memory_budget_fraction, 0.9)
            self.assertEqual(options.vision_batched_sdpa or args.vision_batched_sdpa, group != "A")
            self.assertEqual((args.batch_size, args.seed, args.vision_no_checkpoint_layers), (32, 0, 12))
            self.assertTrue(args.gradient_checkpointing)
            self.assertFalse(args.no_gradient_checkpointing or args.no_vision_gradient_checkpointing)
            config = training.get_deepspeed_config(
                batch_size=args.batch_size, world_size=16, zero_stage=args.deepspeed_zero_stage,
                reduce_bucket_size=args.deepspeed_reduce_bucket_size,
                allgather_bucket_size=args.deepspeed_allgather_bucket_size,
            )
            if args.deepspeed_config:
                config.update(json.loads(Path(args.deepspeed_config).read_text()))
            self.assertEqual(config["train_batch_size"], 512)
            self.assertEqual(config["zero_optimization"]["param_persistence_threshold"], 0)
            configs.append(config)
        self.assertEqual(configs[0], configs[1])
        for key in configs[1]:
            if key != "zero_optimization":
                self.assertEqual(configs[1][key], configs[2][key])
        previous, candidate = [config["zero_optimization"] for config in configs[1:]]
        self.assertEqual(previous.keys(), candidate.keys())
        changed = {key for key in previous if previous[key] != candidate[key]}
        self.assertEqual(changed, {"stage3_max_live_parameters", "stage3_max_reuse_distance"})
        for key in changed:
            self.assertEqual((previous[key], candidate[key]), (2000000000, 3000000000))

    def test_validated_profile_options(self):
        snapshot = {"training_args": {"vision_batched_sdpa": False, "vision_no_checkpoint_layers": 0}}
        common = ["--training-config", "config.json", "--output", "/tmp/new-profile", "--validated-optimizations"]
        options, arguments = parse_options(common)
        _, _, effective = resolve_training_arguments(snapshot, arguments)
        self.assertTrue(effective.vision_batched_sdpa)
        self.assertEqual(effective.vision_no_checkpoint_layers, 12)
        self.assertEqual(effective.deepspeed_reduce_bucket_size, 5e8)
        self.assertEqual(effective.deepspeed_allgather_bucket_size, 5e8)
        self.assertEqual(options.memory_budget_fraction, 0.9)
        options, arguments = parse_options([
            *common, "--memory-budget-fraction", "0.85", "--", "--no_vision_batched_sdpa",
            "--vision_no_checkpoint_layers", "0", "--no_vision_gradient_checkpointing",
        ])
        _, _, effective = resolve_training_arguments(snapshot, arguments)
        self.assertFalse(effective.vision_batched_sdpa)
        self.assertEqual(effective.vision_no_checkpoint_layers, 0)
        self.assertTrue(effective.no_vision_gradient_checkpointing)
        self.assertEqual(options.memory_budget_fraction, 0.85)

    def test_production_launcher_vision_options(self):
        launcher = Path(__file__).resolve().parents[1] / "src/lerobot/scripts/test_azure_v07c.sh"
        source = launcher.read_text()
        parser_source = source.split('\nif [[ ! "$RESUME_GPU_KEEPALIVE_BATCH_SIZE"', 1)[0]
        options_source = source[source.index('VISION_ARGS=()'):source.index('\nLAUNCH_ARGS+=(\n')]
        beginning = source.index('if [ "$GRADIENT_CHECKPOINTING" = false ]; then')
        ending = source.index('\n# V2:', beginning)
        script = parser_source + '\n' + options_source + '\ncmd=""\n' + source[beginning:ending]
        script += '\nprintf "%s\\n" "$cmd"\nprintf "%s\\n" "${LAUNCH_ARGS[*]}"\n'
        cases = [
            ([], ["--vision_no_checkpoint_layers", "12", "--vision_batched_sdpa"]),
            (["--no_vision_batched_sdpa", "--vision_no_checkpoint_layers", "0"],
             ["--vision_no_checkpoint_layers", "0", "--no_vision_batched_sdpa"]),
            (["--no_vision_gradient_checkpointing"],
             ["--no_vision_gradient_checkpointing", "--vision_no_checkpoint_layers", "0", "--vision_batched_sdpa"]),
            (["--no_vision_batched_sdpa", "--vision_batched_sdpa", "--vision_no_checkpoint_layers", "18"],
             ["--vision_no_checkpoint_layers", "18", "--vision_batched_sdpa"]),
        ]
        for arguments, expected in cases:
            result = subprocess.run(["bash", "-s", "--", *arguments], input=script,
                                    capture_output=True, text=True, check=True)
            command, snapshot = [line.split() for line in result.stdout.splitlines()]
            self.assertEqual(command, expected)
            self.assertEqual(snapshot[-len(expected):], expected)
            _, _, command_args = resolve_training_arguments({"training_args": {}}, command)
            _, _, snapshot_args = resolve_training_arguments({"training_args": {}}, snapshot)
            self.assertEqual(command_args.vision_batched_sdpa, snapshot_args.vision_batched_sdpa)
            self.assertEqual(command_args.vision_no_checkpoint_layers, snapshot_args.vision_no_checkpoint_layers)
            self.assertEqual(command_args.vision_batched_sdpa, expected[-1] == "--vision_batched_sdpa")
            from lerobot.scripts import train_lola_v07_azure as training
            from lerobot.configs.types import FeatureType, PolicyFeature

            features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
            metadata = SimpleNamespace(features={}, total_episodes=1, total_frames=1)
            with patch.object(training, "dataset_to_policy_features", return_value=features):
                config = training.build_lola_config(command_args, metadata)[0]
            self.assertEqual(config.vision_batched_sdpa, command_args.vision_batched_sdpa)
            self.assertEqual(config.vision_no_checkpoint_layers, command_args.vision_no_checkpoint_layers)
            self.assertEqual(config.vision_gradient_checkpointing, not command_args.no_vision_gradient_checkpointing)
        _, _, disabled = resolve_training_arguments(
            {"training_args": {"vision_batched_sdpa": True}}, ["--no_vision_batched_sdpa"])
        self.assertFalse(disabled.vision_batched_sdpa)
        result = subprocess.run(["bash", "-s"], input=parser_source +
                                '\nprintf "%s %s %s\\n" "$DEEPSPEED_REDUCE_BUCKET_SIZE" '
                                '"$DEEPSPEED_ALLGATHER_BUCKET_SIZE" "$DEEPSPEED_ZERO_STAGE"\n',
                                capture_output=True, text=True, check=True)
        self.assertEqual([float(value) for value in result.stdout.split()], [5e8, 5e8, 3])

    def test_launcher_cli_and_environment(self):
        launcher = Path(__file__).resolve().parents[1] / "src/lerobot/scripts/profile_azure_v07c.sh"
        environment = {key: value for key, value in os.environ.items() if key not in {
            "NNODES", "NPROC_PER_NODE", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT",
            "TRAINING_CONFIG", "PROFILE_OUTPUT", "PYTHON_BIN",
            "LOCALIZE_IO", "STORAGE_ACCOUNT", "STORAGE_CONTAINER", "MOUNT_PREFIX", "LOCAL_MIRROR", "AZCOPY_PATH",
        }}
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / "fake python"
            executable.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@"\n')
            executable.chmod(0o700)
            for rank in (0, 1):
                command = ["bash", str(launcher), "--nnodes", "2", "--nproc_per_node=8",
                           "--node_rank", str(rank), "--master_addr=10.0.0.1", "--master_port", "9901",
                           "--training-config", "/tmp/training config.json", "--output=/tmp/profile output",
                           "--python", str(executable), "--warmup", "10", "--steps", "50", "--validated-optimizations",
                           "--zero-hpz-partition-size", "8",
                           "--", "--resume", "/tmp/checkpoint tag", "--batch_size", "32"]
                for defaults in ({}, {"NODE_RANK": "99", "NNODES": "99", "MASTER_PORT": "1",
                                      "TRAINING_CONFIG": "/wrong", "PROFILE_OUTPUT": "/wrong"}):
                    result = subprocess.run(command, env=dict(environment, **defaults),
                                            capture_output=True, text=True, check=True)
                    actual = result.stdout.split("\0")[:-1]
                    self.assertEqual(actual[:8], ["-m", "torch.distributed.run", "--nnodes=2",
                                     "--nproc_per_node=8", f"--node_rank={rank}",
                                     "--master_addr=10.0.0.1", "--master_port=9901", "--max_restarts=0"])
                    self.assertEqual(actual[9:], ["--training-config", "/tmp/training config.json",
                                     "--output", "/tmp/profile output", "--warmup", "10", "--steps", "50", "--validated-optimizations",
                                     "--zero-hpz-partition-size", "8",
                                     "--", "--resume", "/tmp/checkpoint tag", "--batch_size", "32"])
            result = subprocess.run(["bash", str(launcher), "--batch_size", "32"],
                                    env=dict(environment, PYTHON_BIN=str(executable), NNODES="2", NPROC_PER_NODE="8",
                                             NODE_RANK="1", MASTER_ADDR="10.0.0.1", MASTER_PORT="9901",
                                             TRAINING_CONFIG="/tmp/config.json", PROFILE_OUTPUT="/tmp/output"),
                                    capture_output=True, text=True, check=True)
            actual = result.stdout.split("\0")[:-1]
            self.assertIn("--node_rank=1", actual)
            self.assertEqual(actual[-2:], ["--batch_size", "32"])
            result = subprocess.run([
                "bash", str(launcher), "--nnodes=2", "--nproc_per_node=8", "--node_rank=1",
                "--master_addr=host", "--training-config=/mnt/config.json", "--output=/mnt/profiles/new",
                "--python", str(executable), "--localize_io", "--storage_account=account",
                "--storage_container", "container", "--mount_prefix=/mnt", "--local_mirror=/scratch/mirror",
                "--azcopy_path", "/tmp/azcopy binary", "--unfreeze-after", "590",
                "--", "--dataset_root", "/mnt/dataset", "--resume", ""],
                env=environment, capture_output=True, text=True, check=True)
            actual = result.stdout.split("\0")[:-1]
            self.assertTrue(actual[0].endswith("profile_lola_v07.py"))
            self.assertEqual(actual[1], "localize")
            self.assertEqual(actual[actual.index("--azcopy_path") + 1], "/tmp/azcopy binary")
            self.assertEqual(actual[actual.index("--unfreeze-after") + 1], "590")
            self.assertEqual(actual[-5:], ["--", "--dataset_root", "/mnt/dataset", "--resume", ""])

    def test_summary_eval_launcher_options(self):
        root = Path(__file__).resolve().parents[1]
        evaluator = root.parent / "eval_on_lola_07_summary_torchrun.py"
        if not evaluator.is_file():
            self.skipTest("Adjacent CALVIN summary evaluator is not installed")
        tree = ast.parse(evaluator.read_text())
        parser_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_parser")
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=[parser_node], type_ignores=[]), str(evaluator), "exec"), namespace)
        parser = namespace["_build_parser"]()
        required = ["--training_config", "/tmp/training config.json", "--checkpoint_path", "/tmp/checkpoint"]
        original = parser.parse_args(required)
        self.assertIsNone(original.vision_batched_sdpa)
        self.assertFalse(original.dit_cuda_graph)
        launcher = root / "src/lerobot/scripts/eval_lola_v07_summary.sh"
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / "fake python"
            executable.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@"\n')
            executable.chmod(0o700)
            for switches, enabled in (([], True), (["--no_vision_batched_sdpa", "--no_dit_cuda_graph"], False)):
                result = subprocess.run([
                    "bash", str(launcher), "--python", str(executable), "--nproc_per_node", "2",
                    "--eval-script", str(evaluator), "--", *required, *switches,
                ], capture_output=True, text=True, check=True)
                command = result.stdout.split("\0")[:-1]
                self.assertIn("--nproc_per_node=2", command)
                actual = parser.parse_args(command[command.index(str(evaluator)) + 1:])
                self.assertEqual(actual.vision_batched_sdpa, enabled)
                self.assertEqual(actual.dit_cuda_graph, enabled)
                self.assertEqual(actual.num_inference_steps, original.num_inference_steps)
                self.assertEqual(actual.action_step, original.action_step)
                self.assertEqual(actual.training_config, original.training_config)
        contract_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                             and node.name == "_build_evaluation_contract")
        contract_return = next(node for node in contract_node.body if isinstance(node, ast.Return))
        optimization_node = next(value for key, value in zip(contract_return.value.keys, contract_return.value.values)
                                 if isinstance(key, ast.Constant) and key.value == "forward_optimizations")
        for enabled in (False, True):
            options = SimpleNamespace(dit_cuda_graph=enabled)
            config = SimpleNamespace(vision_batched_sdpa=enabled)
            actual = eval(compile(ast.Expression(optimization_node), str(evaluator), "eval"),
                          {"args": options, "config": config})
            self.assertEqual(actual, dict(vision_batched_sdpa=enabled, dit_cuda_graph=enabled))

    def test_launcher_rejects_missing_values(self):
        launcher = Path(__file__).resolve().parents[1] / "src/lerobot/scripts/profile_azure_v07c.sh"
        for arguments in (["--node_rank"], ["--nnodes", "--master_addr", "host"], ["--output="]):
            result = subprocess.run(["bash", str(launcher), *arguments], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("requires a", result.stderr)

    def test_checkpoint_tag_with_node_local_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            for rank in range(8, 16):
                (Path(folder) / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt").touch()
            validate_checkpoint_tag(folder, range(8, 16))
            with self.assertRaisesRegex(ValueError, "zero_pp_rank_0_"):
                validate_checkpoint_tag(folder, [0])
            with self.assertRaisesRegex(ValueError, "zero_pp_rank_16_"):
                validate_checkpoint_tag(folder, [16])

    def test_checkpoint_metadata_with_production_preflight(self):
        from lerobot.scripts import train_lola_v07_azure as training

        fingerprint = {key: "expected" for key in training.ARCH_FINGERPRINT_KEYS}
        trainer = SimpleNamespace(config=SimpleNamespace(is_segment_summary=True, **fingerprint))
        preflight = training.LoLAV07Trainer._assert_checkpoint_arch_compatible
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "blob/run/step_000010"
            local = Path(folder) / "mirror/run/step_000010"
            source.mkdir(parents=True)
            local.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "training_config.json"):
                preflight(trainer, str(local))
            with self.assertRaises(FileNotFoundError):
                stage_checkpoint_config(source, local)
            content = json.dumps(dict(lola_config=fingerprint)).encode() + b"\n"
            (source.parent / "training_config.json").write_bytes(content)
            stale = dict(fingerprint)
            changed_key = next(iter(stale))
            stale[changed_key] = "stale"
            (local.parent / "training_config.json").write_text(json.dumps(dict(lola_config=stale)))
            with patch.object(training, "_log"):
                receipt = stage_checkpoint_config(source, local)
                self.assertEqual(receipt["source"], str(source.parent / "training_config.json"))
                self.assertEqual((local / "training_config.json").read_bytes(), content)
                preflight(trainer, str(local))
                (source / "training_config.json").write_text(json.dumps(dict(lola_config=stale)))
                receipt = stage_checkpoint_config(source, local)
                self.assertEqual(receipt["source"], str(source / "training_config.json"))
                with self.assertRaisesRegex(RuntimeError, changed_key):
                    preflight(trainer, str(local))
            (source / "training_config.json").write_text('{"training_args": {}}')
            with self.assertRaisesRegex(ValueError, "lola_config"):
                stage_checkpoint_config(source, local)
            (source / "training_config.json").write_text("{malformed")
            with self.assertRaises(json.JSONDecodeError):
                stage_checkpoint_config(source, local)
            (source / "training_config.json").unlink()
            receipt = stage_checkpoint_config(source, source)
            self.assertEqual(receipt["source"], receipt["local"])
            self.assertFalse((source / "training_config.json").exists())
            self.assertEqual((source.parent / "training_config.json").read_bytes(), content)

    def test_localized_io_two_nodes_and_failures(self):
        from lerobot.scripts import download_azure_azcopy as transfers

        def resolve(snapshot, arguments):
            parser = argparse.ArgumentParser()
            for name in ("dataset_root", "vlm_path", "resume", "strategy"):
                parser.add_argument(f"--{name}")
            parser.add_argument("--deepspeed_zero_stage", type=int)
            parser.add_argument("--no_vision_gradient_checkpointing", action="store_true")
            parser.set_defaults(**snapshot["training_args"])
            return None, parser, parser.parse_args(arguments)

        for child_exit, fail_upload, missing_shard, missing_metadata, unfreeze in (
            (0, None, False, False, False), (7, None, False, False, False),
            (0, "rank008", False, False, False), (0, "upload_status.json", False, False, False),
            (0, None, True, False, False), (0, None, False, True, False),
            (0, None, False, False, True)):
            with self.subTest(child_exit=child_exit, fail_upload=fail_upload, missing_shard=missing_shard,
                      missing_metadata=missing_metadata), \
                    tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                mount = root / "mount"
                blob = root / "uploaded"
                dataset = mount / "dataset with spaces"
                vlm = mount / "model"
                checkpoint = mount / "checkpoints/run/step_000010"
                for path in (dataset / "meta", vlm, checkpoint):
                    path.mkdir(parents=True)
                (dataset / "meta/info.json").write_text("{}")
                (vlm / "config.json").write_text("{}")
                checkpoint_config = json.dumps(dict(lola_config=dict(history_tokenization_mode="segment_summary")))
                if not missing_metadata:
                    (checkpoint.parent / "training_config.json").write_text(checkpoint_config)
                for rank in range(16):
                    (checkpoint / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt").touch()
                    if not missing_shard or rank != 9:
                        (checkpoint / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").touch()
                config = mount / "training_config.json"
                config.write_text(json.dumps(dict(training_args=dict(
                    dataset_root="/stale/dataset", vlm_path=str(vlm), resume=str(checkpoint),
                    strategy="deepspeed", deepspeed_zero_stage=3,
                    train_vlm=True, vlm_unfreeze_v_loss_threshold=0.6))))
                downloaded = []
                launched = []

                def download(binary, source, destination, **kwargs):
                    relative = source.split("/container/", 1)[1]
                    origin = mount / relative
                    target = Path(destination)
                    downloaded.append((origin, kwargs.get("include_patterns")))
                    for path in origin.rglob("*"):
                        patterns = kwargs.get("include_patterns")
                        if path.is_file() and (not patterns or any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)):
                            output = target / path.relative_to(origin)
                            output.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(path, output)
                    return True

                def upload(binary, source, destination, **kwargs):
                    if node == 1 and Path(source).name == fail_upload:
                        return False
                    target = blob / destination.split("/container/", 1)[1]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if Path(source).is_dir():
                        shutil.copytree(source, target)
                    else:
                        shutil.copyfile(source, target)
                    return True

                def child(command, console):
                    launched.append(command)
                    options, arguments = parse_options(command[command.index("--training-config"):])
                    self.assertEqual(options.memory_budget_fraction, 0.9)
                    self.assertTrue(options.vision_batched_sdpa)
                    self.assertEqual(options.zero_hpz_partition_size, 8)
                    self.assertIn("--no_vision_gradient_checkpointing", arguments)
                    _, _, training = resolve(json.loads(options.training_config.read_text()), arguments)
                    self.assertFalse(Path(training.dataset_root).is_relative_to(mount))
                    self.assertTrue((Path(training.dataset_root) / "meta/info.json").is_file())
                    if unfreeze:
                        self.assertEqual(options.unfreeze_after, 2)
                        self.assertEqual(training.resume, "")
                        self.assertEqual(os.environ["LOLA_PROFILE_UNFREEZE_BLOB_BASE"],
                                         "https://account.blob.core.windows.net/container/profiles/run01/unfreeze_exchange")
                        self.assertEqual(os.environ["LOLA_AZCOPY_BIN"], "fake-azcopy")
                    else:
                        self.assertEqual((Path(training.resume) / "training_config.json").read_text(), checkpoint_config)
                    self.assertNotEqual(options.training_config.read_text(), checkpoint_config)
                    node = int(next(value.split("=", 1)[1] for value in command if value.startswith("--node_rank=")))
                    foreign = options.output / f"rank{8 if node == 0 else 0:03d}"
                    foreign.mkdir()
                    (foreign / "foreign.txt").touch()
                    for rank in range(node * 8, node * 8 + 8):
                        directory = options.output / f"rank{rank:03d}"
                        directory.mkdir()
                        (directory / "status.json").write_text(json.dumps(dict(rank=rank)))
                    console.write_text("fake torchrun output\n")
                    return child_exit

                for node in (0, 1):
                    local_tag = root / f"node{node}/checkpoints/run/step_000010"
                    if missing_metadata:
                        local_tag.mkdir(parents=True)
                        (local_tag / "training_config.json").write_text(checkpoint_config)
                    with patch("lerobot.scripts.profile_lola_v07.resolve_training_arguments", side_effect=resolve), \
                            patch.object(transfers, "install_azcopy", return_value="fake-azcopy"), \
                            patch.object(transfers, "download_with_fallback", side_effect=download), \
                            patch.object(transfers, "run_azcopy_transfer", side_effect=upload), \
                            patch("lerobot.scripts.profile_lola_v07.run_profile_child", side_effect=child), \
                            patch("builtins.print"), patch("traceback.print_exc"):
                        result = localize_main([
                            "--nnodes", "2", "--nproc_per_node", "8", "--node_rank", str(node),
                            "--master_addr", "host", "--master_port", "9901", "--storage_account", "account",
                            "--storage_container", "container", "--mount_prefix", str(mount),
                            "--local_mirror", str(root / f"node{node}"), "--training-config", str(config),
                            "--output", str(mount / "profiles/run01"), "--memory-budget-fraction", "0.9",
                            "--vision-batched-sdpa", "--zero-hpz-partition-size", "8",
                            *(["--unfreeze-after", "2"] if unfreeze else []),
                            "--", "--no_vision_gradient_checkpointing", "--dataset_root", str(dataset),
                            *(["--resume", ""] if unfreeze else [])])
                    expected = child_exit or (1 if missing_metadata or node == 1 and (fail_upload or missing_shard) else 0)
                    self.assertEqual(result, expected)
                    receipt_path = blob / f"profiles/run01/io_node{node:03d}/upload_status.json"
                    if node == 1 and fail_upload == "upload_status.json":
                        self.assertFalse(receipt_path.exists())
                        receipt_path = root / f"node{node}/profiles/run01/io_node{node:03d}/upload_status.json"
                    receipt = json.loads(receipt_path.read_text())
                    self.assertEqual(receipt["child_exit_code"], 1 if missing_metadata or missing_shard and node == 1 else child_exit)
                    self.assertEqual(receipt["upload_complete"], not (fail_upload == "rank008" and node == 1))
                    self.assertEqual(len(list(local_tag.glob("*model_states.pt"))), 0 if missing_metadata or unfreeze else 8)
                    if missing_metadata:
                        error_path = blob / f"profiles/run01/io_node{node:03d}/error.log"
                        self.assertIn("Missing checkpoint training_config.json", error_path.read_text())
                self.assertEqual(len(launched), 0 if missing_metadata else (1 if missing_shard else 2))
                self.assertEqual(len(downloaded), 0 if missing_metadata else 4 if unfreeze else 6)
                self.assertTrue((blob / "profiles/run01/source_training_config.json").is_file())
                self.assertFalse(list(blob.rglob("foreign.txt")))
                self.assertTrue((checkpoint / "zero_pp_rank_0_mp_rank_00_model_states.pt").is_file())
                if not missing_metadata:
                    self.assertEqual((checkpoint.parent / "training_config.json").read_text(), checkpoint_config)

    def test_localization_dry_run_and_path_safety(self):
        from lerobot.scripts import download_azure_azcopy as transfers
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mount = root / "mount"
            mount.mkdir()
            config = mount / "config.json"
            config.write_text('{"training_args": {}}')
            effective = SimpleNamespace(dataset_root=str(mount / "dataset"), vlm_path=str(mount / "vlm"),
                                        resume=str(mount / "checkpoints/run/step_1"),
                                        strategy="deepspeed", deepspeed_zero_stage=3)
            base = ["--nnodes=2", "--nproc_per_node=8", "--node_rank=1", "--master_addr=host",
                    "--master_port=9901", "--storage_account=account", "--storage_container=container",
                    "--mount_prefix", str(mount), "--local_mirror", str(root / "mirror"),
                    "--training-config", str(config), "--dry-run"]
            with patch("lerobot.scripts.profile_lola_v07.resolve_training_arguments", return_value=(None, None, effective)), \
                    patch.object(transfers, "install_azcopy") as install, \
                    patch("lerobot.scripts.profile_lola_v07.run_profile_child") as child, \
                    patch("builtins.print"), patch("sys.stderr"):
                self.assertEqual(localize_main([*base, "--output", str(mount / "profiles/new")]), 0)
                for output in (mount, root / "outside", mount / "dataset/profile",
                               mount / "checkpoints/run/profile"):
                    with self.assertRaises(SystemExit):
                        localize_main([*base, "--output", str(output)])
                install.assert_not_called()
                child.assert_not_called()
            self.assertFalse((root / "mirror").exists())

    def test_child_console_and_exit_code(self):
        import sys
        with tempfile.TemporaryDirectory() as folder, patch("builtins.print"):
            console = Path(folder) / "console.log"
            code = run_profile_child([sys.executable, "-c", "print('child output'); raise SystemExit(7)"], console)
            self.assertEqual(code, 7)
            self.assertEqual(console.read_text(), "child output\n")

    def test_counter_deltas(self):
        self.assertEqual(counter_deltas(
            {"num_alloc_retries": 4, "num_device_free": None},
            {"num_alloc_retries": 6, "num_device_free": None}), {"num_alloc_retries": 2})

    def test_trainer_passthrough(self):
        options, arguments = parse_options([
            "--training-config", "config.json", "--output", "results",
            "--steps", "3", "--trace-steps", "0", "--memory-budget-fraction", "0.9",
            "--", "--batch_size", "64"])
        self.assertEqual(arguments, ["--batch_size", "64"])
        self.assertEqual(options.steps, 3)
        self.assertEqual(options.memory_budget_fraction, 0.9)

    def test_memory_budget_estimate_and_invalid_limits(self):
        before = dict(total_bytes=100, used_bytes=70, non_torch_bytes=10)
        after = dict(total_bytes=100, used_bytes=75, non_torch_bytes=15)
        stats = {"reserved_bytes.all.peak": 80}
        budget = assess_memory_budget(before, after, stats, 0.9)
        self.assertEqual(budget["estimated_peak_fraction"], 0.95)
        self.assertTrue(budget["exceeded"])
        stats["reserved_bytes.all.peak"] = 75
        self.assertFalse(assess_memory_budget(before, after, stats, 0.9)["exceeded"])
        for value in ("0", "1", "-0.1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(SystemExit), patch("sys.stderr"):
                parse_options(["--training-config", "config.json", "--output", "results",
                               "--memory-budget-fraction", value])

    def test_memory_budget_collective_exit(self):
        for local_exceeds in (True, False):
            with self.subTest(local_exceeds=local_exceeds), tempfile.TemporaryDirectory() as folder:
                options = SimpleNamespace(output=Path(folder), warmup=0, steps=2, trace_steps=0,
                                          trace_ranks=(), memory_history=False, sync_phases=False,
                                          memory_budget_fraction=0.9)
                trainer = SimpleNamespace(world_rank=0, device="cpu", global_step=10)
                recorder = BenchRecorder(trainer, options)
                stats = {"num_alloc_retries": 0, "reserved_bytes.all.current": 60,
                         "reserved_bytes.all.peak": 90 if local_exceeds else 70}
                with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                        patch("lerobot.scripts.profile_lola_v07.memory_stats", return_value=stats), \
                        patch("torch.cuda.mem_get_info", return_value=(30, 100)), \
                        patch("torch.distributed.is_initialized", return_value=True), \
                        patch("torch.distributed.all_reduce", side_effect=lambda tensor, op: tensor.fill_(1)) as reduce:
                    iterator = iter(ProfileLoader([1, 2], recorder))
                    next(iterator)
                    trainer.global_step += 1
                    with self.assertRaisesRegex(RuntimeError, "Memory budget exceeded"):
                        next(iterator)
                    reduce.assert_called_once()
                recorder.journal.close()
                row = json.loads((recorder.output / "steps.jsonl").read_text())
                self.assertEqual(row["memory_budget"]["exceeded"], local_exceeds)
                self.assertTrue(row["memory_budget"]["any_rank_exceeded"])
                self.assertEqual(recorder.count, 1)

    def test_bounded_steps_and_skip(self):
        with tempfile.TemporaryDirectory() as folder:
            options = SimpleNamespace(output=Path(folder), warmup=1, steps=2, trace_steps=0,
                                      trace_ranks=(), memory_history=False, sync_phases=False)
            trainer = SimpleNamespace(world_rank=0, device="cpu", global_step=10)
            recorder = BenchRecorder(trainer, options)
            stats = {"num_alloc_retries": 0, "allocated_bytes.all.current": 1}
            with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                    patch("lerobot.scripts.profile_lola_v07.memory_stats", return_value=stats):
                loader = ProfileLoader([1, 2, 3, 4, 5], recorder)
                self.assertEqual(len(loader), 5)
                iterator = iter(loader)
                next(iterator)
                next(iterator)
                trainer.global_step += 1
                next(iterator)
                trainer.global_step += 1
                next(iterator)
                trainer.global_step += 1
                with self.assertRaises(BenchComplete):
                    next(iterator)
            recorder.journal.close()
            rows = [json.loads(line) for line in (recorder.output / "steps.jsonl").read_text().splitlines()]
            self.assertEqual([row["global_step"] for row in rows], [11, 12, 13])
            self.assertEqual([row["warmup"] for row in rows], [True, False, False])
            self.assertEqual(recorder.count, 3)

    def test_unfreeze_lifecycle_and_phases(self):
        import weakref

        class Engine:
            def __init__(self, trainable):
                self._lola_profile_deepspeed_config = dict(trainable=trainable)

            def backward(self, loss):
                pass

            def step(self):
                pass

        with tempfile.TemporaryDirectory() as folder:
            options, _ = parse_options(["--training-config", "config.json", "--output", folder,
                                        "--unfreeze-after", "2", "--warmup", "1", "--steps", "2", "--trace-steps", "0"])
            options.memory_budget_fraction = None
            visual = torch.nn.Module()
            block = torch.nn.Module()
            block.attn = torch.nn.Linear(1, 1)
            block.attn.config = SimpleNamespace(_attn_implementation="sdpa")
            block.attn.attention_dropout = 0
            block.attn._lola_original_forward = block.attn.forward
            block.gradient_checkpointing = False
            visual.blocks = torch.nn.ModuleList([block])
            visual.requires_grad_(False)
            trainer = SimpleNamespace(world_rank=0, device="cpu", global_step=0, model=Engine(False),
                                      policy=SimpleNamespace(vlm=SimpleNamespace(visual=visual, parameters=visual.parameters)),
                                      config=SimpleNamespace(vlm_unfreeze_v_loss_threshold=0.6),
                                      _vlm_delayed_unfreeze=True, _vlm_unfrozen=False,
                                      ckpt_dir="/must-not-write")
            old_engine = weakref.ref(trainer.model)

            def training_step(batch):
                self.assertEqual(trainer.config.vlm_unfreeze_v_loss_threshold, 0.6)
                trainer._pending_deepspeed_unfreeze = True
                return block.attn(torch.ones(1))

            def unfreeze():
                self.assertEqual(trainer.global_step, 2)
                self.assertEqual(trainer.ckpt_dir, str(Path(folder) / "unfreeze_checkpoint"))
                trainer.model = None
                self.assertIsNone(old_engine(), "Profile must not retain the destroyed engine")
                trainer.model = Engine(True)
                visual.requires_grad_(True)
                trainer._vlm_unfrozen = True
                trainer._vlm_delayed_unfreeze = False

            trainer.training_step = training_step
            trainer._unfreeze_vlm_deepspeed = unfreeze
            recorder = BenchRecorder(trainer, options)
            stats = {"num_alloc_retries": 0}
            with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                    patch("lerobot.scripts.profile_lola_v07.memory_stats", return_value=stats), \
                    profile_engine_lifecycle(trainer, recorder), self.assertRaises(BenchComplete):
                for batch in ProfileLoader(range(10), recorder):
                    trainer.training_step(batch)
                    trainer.model.backward(None)
                    trainer.model.step()
                    trainer.global_step += 1
                    if trainer._pending_deepspeed_unfreeze and not trainer._vlm_unfrozen:
                        trainer._pending_deepspeed_unfreeze = False
                        trainer._unfreeze_vlm_deepspeed()
            recorder.journal.close()
            rows = [json.loads(line) for line in (recorder.output / "steps.jsonl").read_text().splitlines()]
            self.assertEqual([row["phase"] for row in rows],
                             ["frozen", "unfreeze", "post_unfreeze_warmup", "measure", "measure"])
            self.assertEqual([row["warmup"] for row in rows], [True, True, True, False, False])
            self.assertTrue(all("backward" in row["host_stage_s"] for row in rows))
            self.assertTrue(all(row["vision_attention"]["calls"] == 1 for row in rows))
            receipt = json.loads((recorder.output / "unfreeze.json").read_text())
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["before"]["vlm_trainable_tensors"], 0)
            self.assertEqual(receipt["after"]["vlm_trainable_tensors"], 2)
            self.assertEqual(trainer.ckpt_dir, "/must-not-write")
            self.assertIs(trainer.training_step, training_step)
            (recorder.output / "status.json").write_text(json.dumps(dict(status="complete")))
            (recorder.output / "manifest.json").write_text(json.dumps(dict(
                rank=0, world_size=1, training_args=dict(batch_size=1),
                bench=dict(unfreeze_after=2, warmup=1, steps=2))))
            with patch("builtins.print"):
                summary = summarize(Path(folder))
            self.assertEqual(summary["per_rank"][0]["count"], 2)
            self.assertEqual(summary["unfreeze"]["global_step"], 2)
            receipt["status"] = "failed"
            (recorder.output / "unfreeze.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "incomplete production unfreeze"):
                summarize(Path(folder))

    def test_unfreeze_trace_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            options, _ = parse_options(["--training-config", "config.json", "--output", folder,
                                        "--unfreeze-after", "2", "--warmup", "1", "--steps", "3", "--trace-steps", "1"])
            options.memory_budget_fraction = None
            trainer = SimpleNamespace(world_rank=0, device="cpu", global_step=0)
            recorder = BenchRecorder(trainer, options)
            trace_events = []
            with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
                    patch("lerobot.scripts.profile_lola_v07.memory_stats", return_value={}), \
                    patch("torch.distributed.is_initialized", return_value=False), \
                    patch.object(recorder, "start_trace", side_effect=lambda: trace_events.append(("start", recorder.count))), \
                    patch.object(recorder, "stop_trace", side_effect=lambda: trace_events.append(("stop", recorder.count))), \
                    self.assertRaises(BenchComplete):
                for batch in ProfileLoader(range(10), recorder):
                    trainer.global_step += 1
                    if trainer.global_step == 2:
                        recorder.unfreeze_complete = True
            recorder.journal.close()
            self.assertEqual(trace_events, [("start", 3), ("stop", 4)])
            rows = [json.loads(line) for line in (recorder.output / "steps.jsonl").read_text().splitlines()]
            self.assertEqual([row["global_step"] for row in rows if row["traced"]], [4])

    def test_summary_excludes_trace_and_snapshot_following_step(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for rank in range(2):
                root = directory / f"rank{rank:03d}"
                root.mkdir()
                (root / "manifest.json").write_text(json.dumps(dict(
                    rank=rank, world_size=2, training_args=dict(batch_size=32))))
                (root / "status.json").write_text(json.dumps(dict(status="complete")))
                rows = [dict(global_step=step, warmup=step == 0, traced=step == 1,
                             snapshot_saved=rank == 0 and step == 2,
                             step_s=2.0, data_wait_s=0.5, counter_delta=dict(num_alloc_retries=0))
                        for step in range(5)]
                (root / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch("builtins.print"):
                result = summarize(directory)
            self.assertEqual(result["per_rank"][0]["count"], 2)
            self.assertEqual(result["per_rank"][1]["count"], 2)
            self.assertEqual(result["estimated_samples_per_second"], 25.6)
            manifest_path = directory / "rank001/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["training_args"]["batch_size"] = 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "training_args"):
                summarize(directory)
            manifest["training_args"]["batch_size"] = 32
            for key in ("deepspeed_config", "hpz_topology"):
                manifest[key] = {"different": True}
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, key):
                    summarize(directory)
                del manifest[key]
            manifest["training_args"]["batch_size"] = 32
            manifest_path.write_text(json.dumps(manifest))
            (directory / "rank001/status.json").write_text(json.dumps(dict(status="failed")))
            with self.assertRaises(ValueError):
                summarize(directory)

    def test_summary_memory_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            records = []
            for rank in range(2):
                root = directory / f"rank{rank:03d}"
                root.mkdir()
                (root / "manifest.json").write_text(json.dumps(dict(
                    rank=rank, world_size=2, training_args=dict(batch_size=32))))
                (root / "status.json").write_text(json.dumps(dict(status="complete")))
                row = dict(global_step=1, warmup=False, traced=False, snapshot_saved=False,
                           step_s=2.0, data_wait_s=0.0, counter_delta=dict(num_alloc_retries=0),
                           memory_budget=dict(limit_fraction=0.9, estimated_peak_fraction=0.8,
                                              exceeded=False, any_rank_exceeded=False))
                records.append(row)
                (root / "steps.jsonl").write_text(json.dumps(row) + "\n")
            with patch("builtins.print"):
                result = summarize(directory)
            self.assertEqual(result["memory_budget"], dict(limit_fraction=0.9, passed=True,
                                                           max_estimated_peak_fraction=0.8))
            target = directory / "rank001/steps.jsonl"
            records[1]["memory_budget"]["any_rank_exceeded"] = True
            target.write_text(json.dumps(records[1]) + "\n")
            with self.assertRaisesRegex(ValueError, "memory budget"):
                summarize(directory)
            del records[1]["memory_budget"]
            target.write_text(json.dumps(records[1]) + "\n")
            with self.assertRaisesRegex(ValueError, "memory budget"):
                summarize(directory)


def distributed_smoke(output, real_vision=False, unfreeze=False):
    import os
    from types import MethodType
    import deepspeed
    from lerobot.scripts import train_lola_v07_azure as training
    from lerobot.scripts.profile_lola_v07 import run_bench
    from lerobot.policies.lola_v07.configuration_lola_v07 import LoLAV07Config
    from lerobot.policies.lola_v07.modeling_lola_v07 import LoLAV07Policy
    from lerobot.policies.lola_v07.forward_optimizations import enable_batched_vision_sdpa

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", local_rank)

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm = torch.nn.Module()
            width = 64 if real_vision else 16
            if real_vision:
                from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
                from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

                vision_config = Qwen3VLVisionConfig(
                    hidden_size=width, intermediate_size=128, num_heads=4, depth=2,
                    out_hidden_size=width, deepstack_visual_indexes=[], num_position_embeddings=16,
                    patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
                )
                vision_config._attn_implementation = "sdpa"
                self.vlm.visual = Qwen3VLVisionModel(vision_config)
                self.config = SimpleNamespace(vision_gradient_checkpointing=True, vision_no_checkpoint_layers=1)
                self.vlm.gradient_checkpointing_enable = self.vlm.visual.gradient_checkpointing_enable
                LoLAV07Policy.enable_vlm_gradient_checkpointing(self)
                enable_batched_vision_sdpa(self.vlm)
            else:
                self.vlm.visual = torch.nn.Linear(width, width)
            self.vlm.language_model = torch.nn.Linear(width, width)
            self.model = torch.nn.Module()
            self.model.vlm_bridge = torch.nn.Linear(width, width)
            self.model.dit = torch.nn.Linear(width, width)
            self.model.action_encoder = torch.nn.Identity()
            self.model.arm_dit_to_latent = torch.nn.Identity()
            self.model.grip_dit_to_latent = torch.nn.Identity()
            self.model.state_encoder = None

        def enable_vlm_gradient_checkpointing(self):
            LoLAV07Policy.enable_vlm_gradient_checkpointing(self)

        def forward(self, batch):
            if real_vision:
                visual = self.vlm.visual(batch["input"], batch["grid"], output_hidden_states=True).pooler_output
            else:
                visual = self.vlm.visual(batch["input"])
            hidden = self.vlm.language_model(visual)
            loss = self.model.dit(self.model.vlm_bridge(hidden)).square().mean()
            return loss, {"loss": loss.item()}

    torch.manual_seed(42)
    policy = TinyPolicy().to(device)
    options = SimpleNamespace(output=output, training_config=Path(__file__),
                              warmup=1, steps=4, trace_steps=1, trace_ranks=(0,),
                              memory_history=True, snapshot_threshold=100.0,
                              sync_phases=False, trace_memory=False, vision_batched_sdpa=real_vision,
                              memory_budget_fraction=0.9, zero_hpz_partition_size=None,
                              unfreeze_after=2 if unfreeze else None)
    config = LoLAV07Config(train_vlm=unfreeze, gradient_checkpointing=True, ema_decay=0,
                          vision_batched_sdpa=real_vision, vision_no_checkpoint_layers=1 if real_vision else 0)
    trainer = training.LoLAV07Trainer(
        config, {}, dict(device=device, local_rank=local_rank, world_rank=rank,
                                 world_size=torch.distributed.get_world_size(), is_distributed=True),
        max_steps=100, strategy="deepspeed", batch_size=2, train_vlm=unfreeze, deepspeed_zero_stage=3,
        training_args=dict(seed=42, batch_size=2), log_every_n_steps=100,
    )
    trainer.policy = policy
    trainer.total_steps = 100
    policy.config = config
    with profile_deepspeed_initialization(options):
        if unfreeze:
            policy.vlm.requires_grad_(False)
            trainer._setup_deepspeed()
        else:
            optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
            trainer.model, trainer.optimizer, _, _ = deepspeed.initialize(
                model=policy, optimizer=optimizer, config=dict(
                    train_micro_batch_size_per_gpu=2, gradient_accumulation_steps=1,
                    zero_optimization=dict(stage=3, stage3_param_persistence_threshold=0),
                    zero_allow_untested_optimizer=True, bf16=dict(enabled=real_vision),
                    steps_per_print=1000,
                ),
            )
    trainer.preprocessor = lambda batch: {key: value.to(device) for key, value in batch.items()}

    def training_step(owner, batch, timing_dict=None):
        prepared = owner.preprocessor(batch)
        prepared["input"] = prepared["input"].to(next(owner.policy.parameters()).dtype)
        return owner.model(prepared)

    trainer.training_step = MethodType(training_step, trainer)
    loader = [dict(input=torch.randn(40, 12), grid=torch.tensor([[1, 4, 4], [1, 2, 2], [1, 4, 4], [1, 2, 2]]))
              if real_vision else {"input": torch.randn(2, 16)} for batch_index in range(16)]
    with patch("lerobot.policies.lola_v07.forward_optimizations.functional.scaled_dot_product_attention",
               wraps=torch.nn.functional.scaled_dot_product_attention) as sdpa, profile_deepspeed_initialization(options):
        run_bench(trainer, options, loader, 0, 0, training.LoLAV07Trainer.train)
    if real_vision:
        assert sdpa.call_count == 5 * 6 + (2 * 4 if unfreeze else 0), sdpa.call_count
    assert trainer.global_step == (7 if unfreeze else 5)
    torch.distributed.barrier()
    assert not list(output.rglob("*model_states.pt"))
    if rank == 0:
        result = summarize(output)
        assert all(row["count"] == 3 for row in result["per_rank"].values())
        assert result["memory_budget"]["passed"]
        assert (output / "rank000/trace.json").is_file()
        assert (output / "rank000/memory_final.pickle").is_file()
        if unfreeze:
            assert result["unfreeze"]["before"]["vlm_trainable_tensors"] == 0
            assert result["unfreeze"]["after"]["vlm_trainable_tensors"] > 0
            assert result["unfreeze"]["vision_fallback_calls"] == 0
            assert result["unfreeze"]["after"]["vision_gc"] == [True, False]
        print(f"PASS: real two-rank ZeRO-3 updates, vision={real_vision}, unfreeze={unfreeze}, 90% budget, trace, allocator snapshot, bounded exit, no retained checkpoint")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] in (["--distributed-smoke"], ["--distributed-vision-smoke"], ["--distributed-unfreeze-smoke"]):
        distributed_smoke(Path(sys.argv[2]), real_vision=sys.argv[1] != "--distributed-smoke",
                          unfreeze=sys.argv[1] == "--distributed-unfreeze-smoke")
    else:
        unittest.main()