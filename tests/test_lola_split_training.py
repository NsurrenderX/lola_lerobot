from datetime import timedelta
from contextlib import contextmanager
import json
import multiprocessing
import os
import random
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

from lerobot.scripts import train_lola_v07_split as split


class RandomDataset(Dataset):
    def __len__(self):
        return 48

    def __getitem__(self, index):
        return torch.tensor([index, random.random(), np.random.rand(), torch.rand(()).item()], dtype=torch.float64)


def exchange_worker(directory, node, mode):
    root = Path(directory) / f'node{node}'
    torch.distributed.init_process_group('gloo', init_method=f'file://{directory}/rendezvous', rank=node,
                                         world_size=2, timeout=timedelta(seconds=15))
    try:
        actual_exchange = split.exchange_files
        def small_chunks(root, manifests, node_rank):
            actual_exchange(root, manifests, node_rank, chunk_bytes=7)
        with patch.object(split, 'exchange_files', small_chunks):
            continued = split.finish_frozen(root, 1 if mode == 'failure' and node == 1 else 0, node, 1)
        split.write_json(root / 'result.json', dict(continued=continued))
    except (RuntimeError, ValueError) as error:
        split.write_json(root / 'result.json', dict(error=str(error)))
    finally:
        torch.distributed.destroy_process_group()


def staging_worker(directory, node, port, release, entered, finished, mode):
    root = Path(directory) / f'node{node}'
    options = SimpleNamespace(master_addr='127.0.0.1', master_port=port, node_rank=node,
                              nnodes=2, nproc_per_node=8, exchange_timeout=15)
    plan = dict(arguments=dict(batch_size=16 if mode == 'mismatch' and node == 1 else 32))
    if node == 1 and not release.wait(timeout=15):
        raise RuntimeError('Test did not release the slow node')
    original_group = split.node_group
    @contextmanager
    def signal_group(options, offset):
        if node == 0:
            entered.set()
        with original_group(options, offset):
            yield
    try:
        with patch.object(split, 'node_group', signal_group):
            split.wait_for_staging(options, plan, root,
                                   'download failed' if mode == 'failure' and node == 1 else None)
        if mode == 'success':
            result = subprocess.run([
                sys.executable, '-m', 'torch.distributed.run', '--nnodes=2', '--nproc_per_node=1',
                f'--node_rank={node}', '--master_addr=127.0.0.1', f'--master_port={port}',
                '--rdzv_conf=timeout=15', '--max_restarts=0', str(Path(__file__).resolve()), '--rendezvous-smoke',
            ], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True, timeout=45)
            if result.returncode != 0:
                raise RuntimeError(f'Torchrun failed after staging gate: {result.stdout}\n{result.stderr}')
            if f'CPU_WORKER_READY rank={node} world_size=2' not in result.stdout:
                raise RuntimeError('Torchrun did not start the expected CPU worker')
            split.write_json(root / 'torchrun.json', dict(returncode=result.returncode, worker_rank=node))
        split.write_json(root / 'result.json', dict(launch=True))
    except (RuntimeError, ValueError) as error:
        split.write_json(root / 'result.json', dict(error=str(error)))
    finally:
        finished.set()


class SplitTrainingTests(unittest.TestCase):
    def test_staging_waits_for_all_nodes_and_rejects_failures(self):
        for mode in ('success', 'failure', 'mismatch'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with socket.socket() as listener:
                    listener.bind(('127.0.0.1', 0))
                    port = listener.getsockname()[1] - 4
                context = multiprocessing.get_context('spawn')
                release, entered = context.Event(), context.Event()
                finished = [context.Event(), context.Event()]
                processes = [context.Process(target=staging_worker,
                             args=(directory, node, port, release, entered, finished[node], mode)) for node in range(2)]
                for process in processes:
                    process.start()
                try:
                    self.assertTrue(entered.wait(timeout=15), 'Fast node did not reach staging gate')
                    self.assertFalse(finished[0].is_set(), 'Fast node launched before slow node was ready')
                    self.assertFalse((Path(directory) / 'node0/result.json').exists())
                    release.set()
                    for process in processes:
                        process.join(timeout=60)
                        self.assertFalse(process.is_alive(), 'Staging gate hung')
                        self.assertEqual(process.exitcode, 0)
                    for node in range(2):
                        root = Path(directory) / f'node{node}'
                        result = split.read_json(root / 'result.json')
                        receipt = split.read_json(root / 'staging_status.json')
                        self.assertEqual(len(receipt['nodes']), 2)
                        self.assertEqual(receipt['endpoint'], f'127.0.0.1:{port + 4}')
                        if mode == 'success':
                            self.assertEqual(result, dict(launch=True))
                            self.assertEqual(receipt['phase'], 'ready')
                            self.assertEqual(split.read_json(root / 'torchrun.json'), dict(returncode=0, worker_rank=node))
                        else:
                            self.assertIn('training not started', result['error'])
                            self.assertEqual(receipt['phase'], 'failed')
                            self.assertFalse((root / 'torchrun.json').exists())
                finally:
                    release.set()
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                            process.join()

    def test_shell_launcher_environment_and_passthrough(self):
        launcher = Path(split.__file__).with_name('train_azure_v07_split.sh')
        with tempfile.TemporaryDirectory() as directory:
            environment_root = Path(directory) / 'python environment'
            executable = environment_root / 'bin/python'
            executable.parent.mkdir(parents=True)
            (environment_root / 'lib').mkdir()
            executable.write_text(
                f'#!{sys.executable}\n'
                'import json, os, sys\n'
                'print(json.dumps(dict(arguments=sys.argv[1:], environment=dict(os.environ))))\n'
            )
            executable.chmod(0o755)
            environment = dict(os.environ, PYTHON_BIN=str(executable), PYTHONPATH='/existing/python',
                               LD_LIBRARY_PATH='/existing/lib', TOKENIZERS_PARALLELISM='true',
                               OMP_NUM_THREADS='12', PYTORCH_CUDA_ALLOC_CONF='old', WANDB_MODE='online')
            arguments = ['--output', '/path with spaces', '--profile', '--', '--resume', '', '--python', 'literal']
            for interpreter_args in ([], ['--python', str(executable)], [f'--python={executable}']):
                result = subprocess.run(['bash', str(launcher), *interpreter_args, *arguments],
                                        env=environment, cwd=directory, capture_output=True, text=True, check=True)
                payload = json.loads(result.stdout)
                self.assertEqual(payload['arguments'],
                                 ['-m', 'lerobot.scripts.train_lola_v07_split', 'run', *arguments])
                received = payload['environment']
                self.assertEqual(received['TOKENIZERS_PARALLELISM'], 'false')
                self.assertEqual(received['OMP_NUM_THREADS'], '4')
                self.assertEqual(received['PYTORCH_CUDA_ALLOC_CONF'], 'expandable_segments:True')
                self.assertEqual(received['WANDB_MODE'], 'disabled')
                paths = received['PYTHONPATH'].split(':')
                self.assertEqual(Path(paths[0]).resolve(), launcher.parents[2])
                self.assertEqual(Path(paths[1]).resolve(), launcher.parent)
                self.assertEqual(paths[2], '/existing/python')
                paths = received['LD_LIBRARY_PATH'].split(':')
                self.assertEqual(Path(paths[0]).resolve(), environment_root / 'lib')
                self.assertEqual(paths[1], '/existing/lib')
            for arguments in (['--python'], ['--python='], ['--python', '--profile']):
                result = subprocess.run(['bash', str(launcher), *arguments], env=environment,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('--python requires', result.stderr)

    def loader(self, workers=0):
        dataset = RandomDataset()
        return split.restartable_loader(dataset, seed=17, batch_size=3,
                                        sampler=DistributedSampler(dataset, num_replicas=2, rank=0),
                                        num_workers=workers)

    def test_data_resume_and_worker_count(self):
        original = self.loader()
        original.batch_sampler.set_epoch(1)
        expected = list(original)
        for workers in (0, 2):
            resumed = self.loader(workers)
            resumed.batch_sampler.set_epoch(1, start_batch=3)
            actual = list(resumed)
            self.assertEqual(len(actual), len(expected) - 3)
            self.assertTrue(all(torch.equal(left, right) for left, right in zip(expected[3:], actual)))
        original.batch_sampler.set_epoch(2)
        self.assertFalse(torch.equal(expected[0], next(iter(original))))

    def test_loader_does_not_consume_training_rng(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        loader = self.loader()
        loader.batch_sampler.set_epoch(3, start_batch=4)
        next(iter(loader))
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_manifest_integrity_and_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split.write_json(root / 'manifest.json', {'step': 100})
            self.assertEqual(split.read_json(root / 'manifest.json'), {'step': 100})
            files = {'manifest.json': {'size': (root / 'manifest.json').stat().st_size,
                                      'sha256': split.file_hash(root / 'manifest.json')}}
            split.checked_files(root, files)
            split.write_json(root / 'manifest.json', {'step': 101})
            with self.assertRaises(ValueError):
                split.checked_files(root, files)
            for name in ('../escape', '/absolute'):
                with self.assertRaises(ValueError):
                    split.safe_file(root, name)

    def test_real_two_node_exchange_and_failure_gate(self):
        for mode in ('success', 'failure', 'complete'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                rank_files = []
                for node in range(2):
                    root = parent / f'node{node}'
                    split.write_json(root / f'frozen_node{node:03d}.json',
                                     dict(outcome='complete' if mode == 'complete' else 'boundary', step=3))
                    shard = root / 'handoff/boundary' / f'rank{node}.pt'
                    shard.parent.mkdir(parents=True)
                    shard.write_bytes(bytes([node]) * 23)
                    rank_files.append({f'boundary/rank{node}.pt': dict(size=23, sha256=split.file_hash(shard))})
                for node in range(2):
                    (parent / f'node{node}/handoff/latest').write_text('boundary\n')
                pointer = parent / 'node0/handoff/latest'
                manifest = dict(step=3, contract=dict(world_size=2), rank_files=rank_files,
                                common_files={'latest': dict(size=pointer.stat().st_size,
                                                             sha256=split.file_hash(pointer))})
                for node in range(2):
                    split.write_json(parent / f'node{node}/handoff/manifest.json', manifest)
                context = multiprocessing.get_context('spawn')
                processes = [context.Process(target=exchange_worker, args=(directory, node, mode)) for node in range(2)]
                for process in processes:
                    process.start()
                try:
                    for process in processes:
                        process.join(timeout=35)
                        self.assertFalse(process.is_alive(), 'Gloo node gate hung')
                        self.assertEqual(process.exitcode, 0)
                finally:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                            process.join()
                for node in range(2):
                    root = parent / f'node{node}'
                    result = split.read_json(root / 'result.json')
                    if mode == 'success':
                        self.assertEqual(result, dict(continued=True))
                        for files in rank_files:
                            split.checked_files(root / 'handoff', files)
                        self.assertTrue((root / f'exchange_node{node:03d}.json').is_file())
                    else:
                        if mode == 'failure':
                            self.assertIn('Frozen-stage node failure', result['error'])
                        else:
                            self.assertEqual(result, dict(continued=False))
                        self.assertFalse((root / f'exchange_node{node:03d}.json').exists())

    def test_consumer_resets_optimizer_and_preserves_horizon(self):
        observations = {}
        class FakeTrainer:
            def __init__(self, **kwargs):
                observations['init'] = kwargs
                self.total_steps = 40
                self.learning_rate = 0.002
                self.config = SimpleNamespace(vlm_lr_mult=0.5)
                self.policy = torch.nn.Linear(2, 2)
                self.global_step = self.current_epoch = 0
                self.device = 'cpu'
                self.world_rank = self.local_rank = 0

            def _setup_deepspeed(self):
                observations['scheduler_horizon'] = self.total_steps
                def load_checkpoint(*args, **kwargs):
                    observations['load'] = kwargs
                    return '/boundary', {}
                self.model = SimpleNamespace(load_checkpoint=load_checkpoint)

            def train(self, loader, start_step=0, start_epoch=0):
                observations['train'] = (start_step, start_epoch, self.total_steps, self.vlm_lr)
                observations['rng'] = torch.rand(3)
                self.global_step = self.total_steps

        training = SimpleNamespace(LoLAV07Trainer=FakeTrainer, DataLoader=None)
        contract = dict(total_steps=40)
        manifest = dict(contract=contract, step=8, epoch=1, batches_per_epoch=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'handoff'
            split.write_json(root / 'manifest.json', manifest)
            state = split.rng_state('cpu')
            expected = torch.rand(3)
            torch.save(state, root / 'rng_rank000.pt')
            with patch.object(split, 'training_contract', return_value=contract), \
                    patch.object(torch.distributed, 'barrier'), patch.dict(os.environ, LOCAL_WORLD_SIZE='1'), \
                    split.split_training_context(training, SimpleNamespace(seed=0), root, 'trainable'):
                trainer = training.LoLAV07Trainer()
                trainer._setup_deepspeed()
                trainer.train(SimpleNamespace(batch_sampler=SimpleNamespace(full_length=8)))
                with self.assertRaises(RuntimeError):
                    trainer._unfreeze_vlm_deepspeed()
            self.assertEqual(observations['init'], dict(resume_vlm_unfrozen=True))
            self.assertEqual(observations['scheduler_horizon'], 32)
            self.assertEqual(observations['load'], dict(tag='boundary', load_optimizer_states=False,
                                                      load_lr_scheduler_states=False, load_module_strict=True))
            self.assertEqual(observations['train'], (8, 1, 40, 0.001))
            self.assertTrue(torch.equal(observations['rng'], expected))
            self.assertIs(training.LoLAV07Trainer, FakeTrainer)
            self.assertEqual(split.read_json(Path(directory) / 'trainable_node000.json')['outcome'], 'complete')

    def test_contract_rejects_drift_and_invalid_cursor(self):
        contract = dict(total_steps=40)
        manifest = dict(contract=contract, step=8, epoch=1, batches_per_epoch=8)
        split.validate_manifest(manifest, contract, 8)
        for changed in (dict(step=40), dict(epoch=2), dict(contract=dict(total_steps=41))):
            with self.assertRaises(ValueError):
                split.validate_manifest(manifest | changed, contract, 8)
        with self.assertRaises(ValueError):
            split.validate_manifest(manifest, contract, 9)

    def test_profile_protocol_and_forced_boundary(self):
        options = SimpleNamespace(profile=True, profile_frozen_steps=100, profile_warmup=40,
                                  profile_steps=120, profile_trace_steps=3, profile_trace_ranks='0,8',
                                  profile_memory_budget_fraction=0.90, nnodes=2, nproc_per_node=8)
        protocol = split.profile_protocol(options)
        self.assertEqual(protocol['measure_steps'], 120)
        class FakeTrainer:
            def __init__(self):
                self.global_step = 0
                self.total_steps = 1000
            def training_step(self):
                self._pending_deepspeed_unfreeze = True
                return 'loss'
            def _setup_deepspeed(self):
                pass
        training = SimpleNamespace(LoLAV07Trainer=FakeTrainer, DataLoader=None)
        with tempfile.TemporaryDirectory() as directory, \
                split.split_training_context(training, SimpleNamespace(split_profile=protocol), Path(directory), 'frozen'), \
                patch.object(split, 'export_boundary') as export:
            trainer = training.LoLAV07Trainer()
            trainer.training_step()
            self.assertFalse(trainer._pending_deepspeed_unfreeze)
            trainer.global_step = 99
            self.assertEqual(trainer.training_step(), 'loss')
            self.assertTrue(trainer._pending_deepspeed_unfreeze)
            trainer.global_step = 100
            trainer._unfreeze_vlm_deepspeed()
            export.assert_called_once_with(trainer, Path(directory))
            self.assertEqual(trainer._checkpoint_reasons(), [])
            self.assertIsNone(trainer.save_checkpoint())
            trainer.total_steps = 263
            with self.assertRaises(ValueError):
                trainer._setup_deepspeed()
        for name, value in (('profile_steps', 0), ('profile_frozen_steps', 0),
                            ('profile_trace_ranks', '0,16'), ('profile_memory_budget_fraction', 1.0)):
            with patch.object(options, name, value), self.assertRaises(ValueError):
                split.profile_protocol(options)
        options.profile = False
        self.assertIsNone(split.profile_protocol(options))

    def test_profile_consumer_uses_real_loader_and_propagates_failure(self):
        from lerobot.scripts import profile_lola_v07 as bench

        protocol = dict(frozen_steps=100, warmup=40, measure_steps=120, trace_steps=3,
                        trace_ranks=[0], memory_budget_fraction=0.9)
        contract = dict(total_steps=1000)
        manifest = dict(contract=contract, step=100, epoch=1, batches_per_epoch=200)
        class FakeTrainer:
            def __init__(self, **kwargs):
                self.global_step, self.current_epoch = 100, 1
                self.device = 'cpu'
                self.local_rank = self.world_rank = 0
            def train(self, *args, **kwargs):
                raise AssertionError('Profile bypassed recorder')
        training = SimpleNamespace(LoLAV07Trainer=FakeTrainer, DataLoader=None)
        loader = SimpleNamespace(batch_sampler=SimpleNamespace(full_length=200))
        state = dict(vlm_trainable_tensors=5, vlm_parameter_tensors=5)
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'handoff'
                split.write_json(root / 'manifest.json', manifest)
                torch.save(split.rng_state('cpu'), root / 'rng_rank000.pt')
                options = split.split_profile_options(Path(directory), protocol)
                def record(trainer, received_options, received_loader, start_step, start_epoch, original_train):
                    self.assertIs(received_options, options)
                    self.assertIs(received_loader, loader)
                    self.assertIs(original_train, FakeTrainer.train)
                    self.assertEqual((start_step, start_epoch), (100, 1))
                    self.assertIsNone(received_options.handoff_input)
                    self.assertEqual(received_options.steps - received_options.trace_steps, 120)
                    if fail:
                        raise RuntimeError('Memory budget exceeded')
                    trainer.global_step = 263
                with patch.object(split, 'training_contract', return_value=contract), \
                        patch.object(torch.distributed, 'barrier'), patch.dict(os.environ, LOCAL_WORLD_SIZE='1'), \
                        patch.object(bench, 'profile_policy_state', return_value=state), \
                        patch.object(bench, 'run_bench', side_effect=record) as run_bench, \
                        split.split_training_context(training, SimpleNamespace(split_profile=protocol),
                                                     root, 'trainable', options):
                    trainer = training.LoLAV07Trainer()
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'Memory budget exceeded'):
                            trainer.train(loader)
                    else:
                        trainer.train(loader)
                    run_bench.assert_called_once()
                receipt = Path(directory) / 'trainable_node000.json'
                if fail:
                    self.assertFalse(receipt.exists())
                else:
                    self.assertEqual(split.read_json(receipt), dict(outcome='complete', step=263))
                    provenance = split.read_json(Path(directory) / 'profile/split_rank000.json')
                    self.assertEqual(provenance['boundary_step'], 100)
                    self.assertEqual(provenance['handoff_manifest_sha256'], split.file_hash(root / 'manifest.json'))

    def test_boundary_export_includes_frozen_weights_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = {}
            def save_checkpoint(path, **kwargs):
                observations.update(kwargs)
                target = Path(path) / 'boundary'
                target.mkdir()
                torch.save(dict(frozen=torch.tensor([9.])), target / 'zero_pp_rank_0_mp_rank_00_model_states.pt')
                torch.save({}, target / 'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt')
                torch.rand(4)
            trainer = SimpleNamespace(model=SimpleNamespace(save_checkpoint=save_checkpoint), device='cpu',
                                      world_rank=0, local_rank=0, world_size=1, global_step=3,
                                      current_epoch=1, _batches_per_epoch=8)
            before = torch.get_rng_state()
            with patch.object(split, 'training_contract', return_value=dict(total_steps=40)), \
                    patch.object(torch.distributed, 'all_gather_object', side_effect=lambda records, files: records.__setitem__(0, files)), \
                    patch.object(torch.distributed, 'barrier'), self.assertRaises(split.BoundarySaved):
                split.export_boundary(trainer, root)
            self.assertIs(observations['exclude_frozen_parameters'], False)
            self.assertEqual(observations['client_state'], dict(step=3, epoch=1))
            manifest = split.read_json(root / 'manifest.json')
            self.assertEqual(len(manifest['rank_files'][0]), 3)
            split.checked_files(root, manifest['rank_files'][0])
            split.checked_files(root, manifest['common_files'])
            self.assertEqual((root / 'latest').read_text().strip(), 'boundary')
            saved_rng = torch.load(root / 'rng_rank000.pt', weights_only=False)
            self.assertTrue(torch.equal(before, saved_rng['cpu']))


if __name__ == '__main__':
    if sys.argv[1:] == ['--rendezvous-smoke']:
        torch.distributed.init_process_group('gloo', timeout=timedelta(seconds=15))
        print(f'CPU_WORKER_READY rank={torch.distributed.get_rank()} world_size={torch.distributed.get_world_size()}', flush=True)
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    else:
        unittest.main()