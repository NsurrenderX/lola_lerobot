"""CPU-only resume tests. Run: python tests/test_lola_resume_runtime.py."""

import argparse
import ast
import datetime
import importlib.util
import json
import os
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, SequentialSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "lerobot" / "scripts"))

from lerobot.datasets.sampler import SkipBatchSampler
from resume_gpu_keepalive import ResumeGPUKeepalive, _build_bert_workload


FAKE_WORKER = """
import os, signal, socket, sys
sys.path.insert(0, sys.argv[1])
from resume_gpu_keepalive import _arm_parent_death
_arm_parent_death(int(sys.argv[3]))
control = socket.socket(fileno=int(sys.argv[2]))
mode = sys.argv[4]
if mode == 'fail':
    sys.exit(3)
if mode == 'startup_hang':
    signal.pause()
if mode == 'ignore':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
control.sendall(b'{"event":"ready","peak_tensor_bytes":0}\\n')
if mode == 'ignore':
    signal.pause()
else:
    control.recv(4096)
"""


PARENT_HARNESS = """
import json, os, signal, subprocess, sys
sys.path.insert(0, sys.argv[1])
from resume_gpu_keepalive import ResumeGPUKeepalive
mode = sys.argv[2]
lifeline = int(sys.argv[4]) if len(sys.argv) > 4 else None
if lifeline is not None:
    original_popen = subprocess.Popen
    def pass_lifeline(*args, **kwargs):
        kwargs['pass_fds'] = (*kwargs.get('pass_fds', ()), lifeline)
        return original_popen(*args, **kwargs)
    subprocess.Popen = pass_lifeline
def command(self, control_fd):
    return [sys.executable, '-c', sys.argv[3], sys.argv[1], str(control_fd), str(os.getpid()), 'normal']
ResumeGPUKeepalive._command = command
process = None
try:
    with ResumeGPUKeepalive(0, 0, max_seconds=1 if mode == 'ttl' else 30,
                           startup_timeout=3, shutdown_timeout=0.3, log=lambda _: None) as manager:
        manager.start()
        process = manager._process
        if lifeline is not None:
            os.close(lifeline)
        print(json.dumps({'child_pid': process.pid}), flush=True)
        if mode == 'term':
            os.kill(os.getpid(), signal.SIGTERM)
        elif mode == 'crash':
            process.kill()
        signal.pause()
except (RuntimeError, SystemExit) as error:
    print(json.dumps({'error': str(error), 'child_exit': process.returncode}), flush=True)
"""


def fake_command(mode):
    def command(manager, control_fd):
        return [sys.executable, "-c", FAKE_WORKER,
                str(ROOT / "src" / "lerobot" / "scripts"), str(control_fd), str(os.getpid()), mode]
    return command


class RecordingDataset:
    def __init__(self, size):
        self.size = size
        self.reads = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.reads.append(index)
        return index


def trainer_methods(namespace):
    source = ROOT / "src" / "lerobot" / "scripts" / "train_lola_v07_azure.py"
    tree = ast.parse(source.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "LoLAV07Trainer")
    methods = [node for node in trainer.body if isinstance(node, ast.FunctionDef)
               and node.name in {"train", "_train", "_checkpoint_reasons", "_mark_checkpoint_saved"}]
    isolated = ast.Module(body=[ast.ClassDef(
        name="Trainer", bases=[], keywords=[], body=methods, decorator_list=[],
    )], type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated), str(source), "exec"), namespace)
    return namespace["Trainer"]


class TrainerResumeTests(unittest.TestCase):
    def run_training(self, fast, keepalive, restored=True, finished=False, empty=False):
        events = []

        class Guard:
            active = False

            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def start(self):
                self.active = True
                events.append("start")

            def stop(self):
                if self.active:
                    self.active = False
                    events.append("stop")

            def __exit__(self, *args):
                self.stop()

        namespace = dict(
            os=os, time=time, datetime=datetime, json=json, torch=torch,
            SkipBatchSampler=SkipBatchSampler, nullcontext=nullcontext,
            ResumeGPUKeepalive=Guard, _log=lambda _: None,
            dist=types.SimpleNamespace(barrier=lambda: events.append("barrier")),
        )
        trainer_type = trainer_methods(namespace)
        trainer = trainer_type()
        trainer.training_args = dict(resume_fast_skip=fast, resume_gpu_keepalive=keepalive)
        trainer.resume_checkpoint_loaded = restored
        trainer.dist_info = {"local_rank": 0}
        trainer.world_rank = 0
        trainer.world_size = 1
        trainer.is_distributed = True
        trainer.is_main_process = False
        trainer.config = types.SimpleNamespace(ema_decay=0)
        trainer._ema_state = None
        trainer._pending_deepspeed_unfreeze = False
        trainer.model = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
        trainer.bf16_optimizer = None
        trainer.strategy = "ddp"
        trainer.use_bf16 = True
        trainer.gradient_clip_val = 0
        trainer.max_steps = 3 if finished else None
        trainer.total_steps = 3 if finished else 10
        trainer.max_epochs = 2
        trainer.log_every_n_steps = 10000
        trainer.save_every_n_seconds = None
        trainer.interconnect_monitor = None
        trainer.use_wandb = False
        trained = []
        positions = []

        def training_step(batch, timing_dict):
            events.append("forward")
            trained.extend(batch.tolist())
            return trainer.model(torch.ones(1, 1)).sum(), {}

        def checkpoint_reasons(batch_idx, epoch):
            positions.append((epoch, batch_idx))
            return []

        trainer.training_step = training_step
        trainer._checkpoint_reasons = checkpoint_reasons
        dataset = RecordingDataset(10)
        sampler = BatchSampler(SequentialSampler(dataset), 2, drop_last=True)
        if fast:
            sampler = SkipBatchSampler(sampler)
        loader = DataLoader(dataset, batch_sampler=sampler)
        if empty:
            class EmptyLoader:
                batch_sampler = None

                def __len__(self):
                    return 5

                def __iter__(self):
                    return iter(())
            loader = EmptyLoader()
        with tempfile.TemporaryDirectory() as directory:
            trainer.resume_save_dir = directory
            if empty:
                with self.assertRaisesRegex(RuntimeError, "before a productive batch"):
                    trainer.train(loader, start_step=3)
            else:
                trainer.train(loader, start_step=3)
        return trainer, dataset, trained, positions, events

    def test_all_switch_combinations_preserve_steps_and_positions(self):
        for fast in (False, True):
            for keepalive in (False, True):
                with self.subTest(fast=fast, keepalive=keepalive):
                    trainer, dataset, trained, positions, events = self.run_training(fast, keepalive)
                    self.assertEqual(trained, list(range(6, 10)) + list(range(10)))
                    self.assertEqual(positions, [(1, 3), (1, 4)] + [(2, index) for index in range(5)])
                    self.assertEqual(trainer.global_step, 10)
                    self.assertEqual(trainer.scheduler.last_epoch, 7)
                    self.assertEqual(trainer._batches_per_epoch, 5)
                    self.assertEqual(len(dataset.reads), 14 if fast else 20)
                    if keepalive:
                        self.assertEqual(events[:5], ["start", "barrier", "stop", "barrier", "forward"])
                        self.assertEqual(events.count("start"), 1)
                    else:
                        self.assertEqual(set(events), {"forward"})

    def test_unrestored_and_finished_training_do_not_start_keepalive(self):
        for restored, finished in ((False, False), (True, True)):
            with self.subTest(restored=restored, finished=finished):
                *_, events = self.run_training(True, True, restored=restored, finished=finished)
                self.assertNotIn("start", events)

    def test_exhausted_loader_cleans_up_without_barrier(self):
        *_, events = self.run_training(False, True, empty=True)
        self.assertEqual(events, ["start", "stop"])


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = ROOT / "src" / "lerobot" / "scripts" / "test_azure_v07c.sh"
        text = script.read_text()
        cls.harness = (
            "set -e\n" + text[text.index("NNODES=1"):text.index("SCRIPTS_DIR=")]
            + text[text.index('if [ "$NNODES" -eq 1 ]; then'):text.index('echo "Running: $cmd"')]
            + "printf 'ARGS\\0'; printf '%s\\0' \"${LAUNCH_ARGS[@]}\"; printf 'CMD\\0%s\\0' \"$cmd\"\n"
        )
        source = ROOT / "src" / "lerobot" / "scripts" / "train_lola_v07_azure.py"
        tree = ast.parse(source.read_text())
        parser_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                               and node.name == "build_arg_parser")
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=[parser_function], type_ignores=[]), str(source), "exec"), namespace)
        cls.parser = namespace["build_arg_parser"]()

    def command(self, arguments):
        return subprocess.run(["bash", "-s", "--", *arguments], input=self.harness.encode(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)

    def test_all_flags_reach_single_and_multi_node_commands(self):
        for nodes in (1, 2):
            for fast in (False, True):
                for keepalive in (False, True):
                    for resume in ([], ["--resume"], ["--resume", "/checkpoints/run"]):
                        with self.subTest(nodes=nodes, fast=fast, keepalive=keepalive, resume=resume):
                            arguments = ["--nnodes", str(nodes), "--ckpt_dir", "/checkpoints/default",
                                         "--resume_gpu_keepalive_batch_size", "3",
                                         "--resume_gpu_keepalive_max_seconds", "900", *resume]
                            if fast:
                                arguments.append("--resume_fast_skip")
                            if keepalive:
                                arguments.append("--resume_gpu_keepalive")
                            result = self.command(arguments)
                            self.assertEqual(result.returncode, 0, result.stderr.decode())
                            raw_launch, raw_command = result.stdout.split(b"ARGS\0", 1)[1].split(b"CMD\0", 1)
                            launch = raw_launch.decode().strip("\0").split("\0")
                            command = shlex.split(raw_command.decode().strip("\0"))
                            options = command[command.index("src/lerobot/scripts/train_lola_v07_azure.py") + 1:]
                            parsed = self.parser.parse_args(options)
                            self.assertEqual(parsed.resume_fast_skip, fast)
                            self.assertEqual(parsed.resume_gpu_keepalive, keepalive)
                            self.assertEqual(parsed.resume_gpu_keepalive_batch_size, 3)
                            self.assertEqual(parsed.resume_gpu_keepalive_max_seconds, 900)
                            self.assertEqual("--resume_fast_skip" in launch, fast)
                            self.assertEqual("--resume_gpu_keepalive" in launch, keepalive)
                            self.assertEqual(launch[launch.index("--resume_gpu_keepalive_batch_size") + 1], "3")
                            if resume:
                                self.assertEqual(parsed.resume, resume[1] if len(resume) > 1 else "/checkpoints/default")
                            else:
                                self.assertIsNone(parsed.resume)

    def test_defaults_and_invalid_numeric_arguments(self):
        defaults = self.parser.parse_args([])
        self.assertFalse(defaults.resume_fast_skip)
        self.assertFalse(defaults.resume_gpu_keepalive)
        for flag in ("--resume_gpu_keepalive_batch_size", "--resume_gpu_keepalive_max_seconds"):
            for suffix in ([], ["0"], ["-1"], ["--resume"], ["nan"], ["1;exit 0"]):
                with self.subTest(flag=flag, suffix=suffix):
                    self.assertNotEqual(self.command([flag, *suffix]).returncode, 0)


class KeepaliveLifecycleTests(unittest.TestCase):
    def manager(self):
        return ResumeGPUKeepalive(0, 0, startup_timeout=3, shutdown_timeout=0.2, log=lambda _: None)

    def test_unstarted_context_does_not_spawn(self):
        with patch("resume_gpu_keepalive.subprocess.Popen") as popen:
            with self.manager():
                pass
            popen.assert_not_called()

    def test_exit_reaps_child_and_restores_handlers(self):
        handlers = {value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)}
        with patch.object(ResumeGPUKeepalive, "_command", fake_command("normal")):
            with self.manager() as manager:
                manager.start()
                process = manager._process
                self.assertIsNone(process.poll())
            self.assertEqual(process.returncode, 0)
            manager.stop()
        self.assertEqual(handlers, {value: signal.getsignal(value) for value in handlers})

    def test_exception_reaps_child(self):
        with patch.object(ResumeGPUKeepalive, "_command", fake_command("normal")):
            with self.assertRaisesRegex(ValueError, "loader failed"):
                with self.manager() as manager:
                    manager.start()
                    process = manager._process
                    raise ValueError("loader failed")
            self.assertIsNotNone(process.returncode)

    def test_startup_failure_is_not_silently_ignored(self):
        with patch.object(ResumeGPUKeepalive, "_command", fake_command("fail")):
            with self.manager() as manager:
                with self.assertRaisesRegex(RuntimeError, "first completed step"):
                    manager.start()
                self.assertIsNone(manager._process)

    def test_unresponsive_child_is_killed_and_reaped(self):
        with patch.object(ResumeGPUKeepalive, "_command", fake_command("ignore")):
            with self.manager() as manager:
                manager.start()
                process = manager._process
            self.assertEqual(process.returncode, -signal.SIGKILL)

    def test_command_uses_local_rank_not_global_rank(self):
        manager = ResumeGPUKeepalive(2, 10)
        command = manager._command(42)
        self.assertEqual(command[command.index("--local-rank") + 1], "2")
        self.assertEqual(command[command.index("--global-rank") + 1], "10")
        self.assertEqual(command[0], sys.executable)

    def test_startup_timeout_cleans_up(self):
        with patch.object(ResumeGPUKeepalive, "_command", fake_command("startup_hang")):
            with self.manager() as manager:
                manager.startup_timeout = 0.1
                with self.assertRaises(TimeoutError):
                    manager.start()
                self.assertIsNone(manager._process)

    def test_child_inherits_device_namespace_but_not_distributed_rank(self):
        original_popen = subprocess.Popen
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="2,5", WORLD_SIZE="8", LOCAL_RANK="1"):
            with patch.object(ResumeGPUKeepalive, "_command", fake_command("normal")):
                with patch("resume_gpu_keepalive.subprocess.Popen", wraps=original_popen) as popen:
                    with self.manager() as manager:
                        manager.start()
                    environment = popen.call_args.kwargs["env"]
                    self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2,5")
                    self.assertNotIn("WORLD_SIZE", environment)
                    self.assertNotIn("LOCAL_RANK", environment)
                    self.assertEqual(environment["HF_HUB_OFFLINE"], "1")

    def parent_command(self, mode):
        return [sys.executable, "-c", PARENT_HARNESS,
                str(ROOT / "src" / "lerobot" / "scripts"), mode, FAKE_WORKER]

    def test_sigterm_runtime_limit_and_child_crash_abort_cleanly(self):
        for mode, message in (("term", "143"), ("ttl", "runtime limit"), ("crash", "unexpectedly")):
            with self.subTest(mode=mode):
                result = subprocess.run(self.parent_command(mode), capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout.splitlines()[-1])
                self.assertIn(message, report["error"])
                self.assertIsNotNone(report["child_exit"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux parent-death signal required")
    def test_parent_sigkill_terminates_child(self):
        lifetime_read, lifetime_write = os.pipe()
        parent = subprocess.Popen(
            [*self.parent_command("wait"), str(lifetime_write)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(lifetime_write,),
        )
        os.close(lifetime_write)
        child_pid = None
        try:
            self.assertTrue(select.select([parent.stdout], [], [], 5)[0], "No child readiness report")
            child_pid = json.loads(parent.stdout.readline())["child_pid"]
            parent.kill()
            parent.wait(timeout=5)
            self.assertTrue(select.select([lifetime_read], [], [], 5)[0], "Child survived parent SIGKILL")
            self.assertEqual(os.read(lifetime_read, 1), b"")
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if child_pid is not None and not select.select([lifetime_read], [], [], 0)[0]:
                os.kill(child_pid, signal.SIGKILL)
            os.close(lifetime_read)
            parent.stdout.close()
            parent.stderr.close()


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Transformers required for CPU BERT smoke")
class BertWorkloadTests(unittest.TestCase):
    def test_real_small_bert_forward_backward_optimizer_on_cpu(self):
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                model, optimizer, token_ids, labels = _build_bert_workload(torch.device("cpu"), 2)
                self.assertEqual(model.config.num_hidden_layers, 4)
                self.assertEqual(tuple(token_ids.shape), (2, 128))
                before = model.classifier.weight.detach().clone()
                loss = model(input_ids=token_ids, labels=labels).loss
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                optimizer.step()
                self.assertFalse(torch.equal(before, model.classifier.weight))
        finally:
            torch.set_num_threads(threads)


class SkipBatchSamplerTests(unittest.TestCase):
    def test_production_offset_does_not_read_discarded_samples(self):
        dataset = RecordingDataset(2719 * 4)
        sampler = SkipBatchSampler(BatchSampler(SequentialSampler(dataset), 4, drop_last=True))
        loader = DataLoader(dataset, batch_sampler=sampler)
        sampler.set_epoch(1, 2385)
        self.assertEqual(sampler.full_length, 2719)
        self.assertEqual(len(loader), 334)
        batches = list(enumerate(loader, start=sampler.start_batch))
        self.assertEqual(batches[0][0], 2385)
        self.assertEqual(batches[-1][0], 2718)
        self.assertEqual(dataset.reads, list(range(2385 * 4, 2719 * 4)))
        sampler.set_epoch(2)
        self.assertEqual(len(loader), 2719)
        self.assertEqual(next(iter(loader)).tolist(), [0, 1, 2, 3])

    def test_distributed_suffix_matches_original_including_padding(self):
        for drop_last in (False, True):
            for rank in range(3):
                dataset = RecordingDataset(53)
                original = DistributedSampler(dataset, num_replicas=3, rank=rank, seed=17)
                original.set_epoch(4)
                expected = list(BatchSampler(original, 4, drop_last=drop_last))
                wrapped = SkipBatchSampler(BatchSampler(original, 4, drop_last=drop_last))
                for offset in (0, 1, len(expected) - 1, len(expected)):
                    with self.subTest(rank=rank, offset=offset, drop_last=drop_last):
                        wrapped.set_epoch(4, offset)
                        self.assertEqual(list(wrapped), expected[offset:])
                        self.assertEqual(len(wrapped), len(expected) - offset)
                wrapped.set_epoch(5)
                self.assertEqual(original.epoch, 5)
                self.assertEqual(len(wrapped), len(expected))

    def test_invalid_offsets(self):
        wrapped = SkipBatchSampler(BatchSampler(range(8), 2, drop_last=True))
        for offset in (-1, 5):
            with self.assertRaises(ValueError):
                wrapped.set_epoch(1, offset)


if __name__ == "__main__":
    unittest.main()