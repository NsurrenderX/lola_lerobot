import argparse
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
    localize_main, run_profile_child, validate_checkpoint_tag,
)


class ProfileBenchTests(unittest.TestCase):
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
                           "--python", str(executable), "--warmup", "10", "--steps", "50",
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
                                     "--output", "/tmp/profile output", "--warmup", "10", "--steps", "50",
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
                "--azcopy_path", "/tmp/azcopy binary", "--", "--dataset_root", "/mnt/dataset"],
                env=environment, capture_output=True, text=True, check=True)
            actual = result.stdout.split("\0")[:-1]
            self.assertTrue(actual[0].endswith("profile_lola_v07.py"))
            self.assertEqual(actual[1], "localize")
            self.assertEqual(actual[actual.index("--azcopy_path") + 1], "/tmp/azcopy binary")
            self.assertEqual(actual[-3:], ["--", "--dataset_root", "/mnt/dataset"])

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

    def test_localized_io_two_nodes_and_failures(self):
        from lerobot.scripts import download_azure_azcopy as transfers

        def resolve(snapshot, arguments):
            parser = argparse.ArgumentParser()
            for name in ("dataset_root", "vlm_path", "resume", "strategy"):
                parser.add_argument(f"--{name}")
            parser.add_argument("--deepspeed_zero_stage", type=int)
            parser.set_defaults(**snapshot["training_args"])
            return None, parser, parser.parse_args(arguments)

        for child_exit, fail_upload, missing_shard in ((0, None, False), (7, None, False),
                                  (0, "rank008", False), (0, "upload_status.json", False),
                                  (0, None, True)):
            with self.subTest(child_exit=child_exit, fail_upload=fail_upload, missing_shard=missing_shard), \
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
                for rank in range(16):
                    (checkpoint / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt").touch()
                    if not missing_shard or rank != 9:
                        (checkpoint / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").touch()
                config = mount / "training_config.json"
                config.write_text(json.dumps(dict(training_args=dict(
                    dataset_root="/stale/dataset", vlm_path=str(vlm), resume=str(checkpoint),
                    strategy="deepspeed", deepspeed_zero_stage=3))))
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
                    _, _, training = resolve(json.loads(options.training_config.read_text()), arguments)
                    self.assertFalse(Path(training.dataset_root).is_relative_to(mount))
                    self.assertTrue((Path(training.dataset_root) / "meta/info.json").is_file())
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
                            "--output", str(mount / "profiles/run01"), "--", "--dataset_root", str(dataset)])
                    expected = child_exit or (1 if node == 1 and (fail_upload or missing_shard) else 0)
                    self.assertEqual(result, expected)
                    receipt_path = blob / f"profiles/run01/io_node{node:03d}/upload_status.json"
                    if node == 1 and fail_upload == "upload_status.json":
                        self.assertFalse(receipt_path.exists())
                        receipt_path = root / f"node{node}/profiles/run01/io_node{node:03d}/upload_status.json"
                    receipt = json.loads(receipt_path.read_text())
                    self.assertEqual(receipt["child_exit_code"], child_exit if not (missing_shard and node == 1) else 1)
                    self.assertEqual(receipt["upload_complete"], not (fail_upload == "rank008" and node == 1))
                    local_tag = root / f"node{node}/checkpoints/run/step_000010"
                    self.assertEqual(len(list(local_tag.glob("*model_states.pt"))), 8)
                self.assertEqual(len(launched), 1 if missing_shard else 2)
                self.assertEqual(len(downloaded), 6)
                self.assertTrue((blob / "profiles/run01/source_training_config.json").is_file())
                self.assertFalse(list(blob.rglob("foreign.txt")))
                self.assertTrue((checkpoint / "zero_pp_rank_0_mp_rank_00_model_states.pt").is_file())

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
            "--steps", "3", "--trace-steps", "0", "--", "--batch_size", "64"])
        self.assertEqual(arguments, ["--batch_size", "64"])
        self.assertEqual(options.steps, 3)

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
            manifest_path.write_text(json.dumps(manifest))
            (directory / "rank001/status.json").write_text(json.dumps(dict(status="failed")))
            with self.assertRaises(ValueError):
                summarize(directory)


def distributed_smoke(output):
    import os
    from types import MethodType
    import deepspeed
    from lerobot.scripts import train_lola_v07_azure as training
    from lerobot.scripts.profile_lola_v07 import run_bench
    from lerobot.policies.lola_v07.configuration_lola_v07 import LoLAV07Config

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", local_rank)

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm = torch.nn.Module()
            self.vlm.visual = torch.nn.Linear(16, 16)
            self.vlm.language_model = torch.nn.Linear(16, 16)
            self.model = torch.nn.Module()
            self.model.vlm_bridge = torch.nn.Linear(16, 16)
            self.model.dit = torch.nn.Linear(16, 16)

        def forward(self, batch):
            hidden = self.vlm.language_model(self.vlm.visual(batch["input"]))
            loss = self.model.dit(self.model.vlm_bridge(hidden)).square().mean()
            return loss, {"loss": loss.item()}

    torch.manual_seed(42)
    policy = TinyPolicy().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=policy, optimizer=optimizer, config=dict(
            train_micro_batch_size_per_gpu=2, gradient_accumulation_steps=1,
            zero_optimization=dict(stage=3, stage3_param_persistence_threshold=0),
            zero_allow_untested_optimizer=True, bf16=dict(enabled=False),
            steps_per_print=1000,
        ),
    )
    trainer = training.LoLAV07Trainer(
        LoLAV07Config(), {}, dict(device=device, local_rank=local_rank, world_rank=rank,
                                 world_size=torch.distributed.get_world_size(), is_distributed=True),
        max_steps=20, strategy="deepspeed", batch_size=2,
        training_args=dict(seed=42, batch_size=2), log_every_n_steps=100,
    )
    trainer.policy = policy
    trainer.model = engine
    trainer.optimizer = optimizer
    trainer.total_steps = 20
    trainer.preprocessor = lambda batch: {key: value.to(device) for key, value in batch.items()}

    def training_step(owner, batch, timing_dict=None):
        return owner.model(owner.preprocessor(batch))

    trainer.training_step = MethodType(training_step, trainer)
    options = SimpleNamespace(output=output, training_config=Path(__file__),
                              warmup=1, steps=4, trace_steps=1, trace_ranks=(0,),
                              memory_history=True, snapshot_threshold=100.0,
                              sync_phases=False, trace_memory=False, vision_batched_sdpa=False)
    loader = [{"input": torch.randn(2, 16)} for batch_index in range(16)]
    run_bench(trainer, options, loader, 0, 0, training.LoLAV07Trainer.train)
    assert trainer.global_step == 5
    torch.distributed.barrier()
    assert not list(output.rglob("*model_states.pt"))
    if rank == 0:
        result = summarize(output)
        assert all(row["count"] == 3 for row in result["per_rank"].values())
        assert (output / "rank000/trace.json").is_file()
        assert (output / "rank000/memory_final.pickle").is_file()
        print("PASS: real two-rank ZeRO-3 updates, trace, allocator snapshot, bounded exit, no checkpoint")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--distributed-smoke"]:
        distributed_smoke(Path(sys.argv[2]))
    else:
        unittest.main()