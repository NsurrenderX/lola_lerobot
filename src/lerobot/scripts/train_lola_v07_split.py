"""Opt-in node-local, fresh-process handoff for LoLA V07 training."""

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

from lerobot.datasets.sampler import SkipBatchSampler


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as output:
        json.dump(value, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def safe_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError(f'Unsafe relative file: {name}')
    target = Path(root) / relative
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f'File escapes root: {name}')
    return target


class SeededDataset(Dataset):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, position):
        epoch, index = position
        digest = hashlib.sha256(f'lola-split-v1:{self.seed}:{epoch}:{index}'.encode()).digest()
        seed = int.from_bytes(digest[:8], 'little') % (2**63 - 1)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                random.seed(seed)
                np.random.seed(seed % 2**32)
                torch.default_generator.manual_seed(seed)
                return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class RestartableBatchSampler(SkipBatchSampler):
    def __init__(self, batch_sampler):
        super().__init__(batch_sampler)
        self.epoch = 0

    def set_epoch(self, epoch, start_batch=0):
        self.epoch = epoch
        super().set_epoch(epoch, start_batch=start_batch)

    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.epoch, index) for index in batch]


def restartable_loader(dataset, *, seed, batch_size, sampler, shuffle=False, drop_last=True, **kwargs):
    if not isinstance(sampler, DistributedSampler) or not drop_last or shuffle:
        raise ValueError('Split training requires DistributedSampler and drop_last=True')
    if isinstance(dataset, torch.utils.data.IterableDataset):
        raise ValueError('Split training requires a map-style dataset')
    batch_sampler = RestartableBatchSampler(BatchSampler(sampler, batch_size, drop_last=True))
    return DataLoader(SeededDataset(dataset, seed), batch_sampler=batch_sampler,
                      generator=torch.Generator().manual_seed(seed + sampler.rank), **kwargs)


def checked_files(root, files):
    for name, expected in files.items():
        path = safe_file(root, name)
        if path.stat().st_size != expected['size'] or file_hash(path) != expected['sha256']:
            raise ValueError(f'Handoff file mismatch: {name}')


def exchange_files(root, manifests, node_rank, chunk_bytes=8 * 1024 * 1024):
    if chunk_bytes <= 0:
        raise ValueError('chunk_bytes must be positive')
    if len(manifests) != torch.distributed.get_world_size():
        raise ValueError('Exchange manifest count does not match node group')
    identities = set()
    for owner, files in enumerate(manifests):
        for name, expected in sorted(files.items()):
            if name in identities:
                raise ValueError(f'Duplicate shard ownership: {name}')
            identities.add(name)
            path = safe_file(root, name)
            if owner == node_rank:
                checked_files(root, {name: expected})
            elif path.exists():
                raise FileExistsError(f'Refusing stale destination: {path}')
            torch.distributed.barrier()
            temporary = path.with_suffix(path.suffix + '.partial')
            path.parent.mkdir(parents=True, exist_ok=True)
            if owner == node_rank:
                handle = path.open('rb')
            else:
                handle = temporary.open('xb')
            digest = hashlib.sha256()
            try:
                with handle:
                    remaining = expected['size']
                    while remaining:
                        length = min(chunk_bytes, remaining)
                        if owner == node_rank:
                            block = bytearray(handle.read(length))
                            if len(block) != length:
                                raise ValueError(f'Short source read: {name}')
                            tensor = torch.frombuffer(block, dtype=torch.uint8)
                        else:
                            tensor = torch.empty(length, dtype=torch.uint8)
                        torch.distributed.broadcast(tensor, src=owner)
                        content = tensor.numpy().tobytes()
                        digest.update(content)
                        if owner != node_rank:
                            handle.write(content)
                        remaining -= length
                    if owner != node_rank:
                        handle.flush()
                        os.fsync(handle.fileno())
                if digest.hexdigest() != expected['sha256']:
                    raise ValueError(f'Transfer hash mismatch: {name}')
                if owner != node_rank:
                    temporary.replace(path)
            finally:
                if owner != node_rank:
                    temporary.unlink(missing_ok=True)
            torch.distributed.barrier()


class BoundarySaved(Exception):
    pass


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def training_contract(trainer):
    import deepspeed
    import transformers
    from lerobot.datasets import lola_dataset, sampler
    from lerobot.policies.lola_v07 import modeling_lola_v07, forward_optimizations
    from lerobot.scripts import train_lola_v07_azure

    modules = (lola_dataset, sampler, modeling_lola_v07, forward_optimizations, train_lola_v07_azure)
    return dict(version=1, data_policy='lola-split-v1',
                args=trainer.training_args,
                config=json.loads(json.dumps(asdict(trainer.config), default=str)),
                stats=json.loads(json.dumps(trainer.dataset_stats, default=lambda value: value.tolist())),
                total_steps=trainer.total_steps, world_size=trainer.world_size,
                source_hashes={module.__name__: file_hash(module.__file__) for module in modules}
                | {'split': file_hash(__file__)},
                torch=torch.__version__, deepspeed=deepspeed.__version__, transformers=transformers.__version__)


def validate_manifest(manifest, contract, batches_per_epoch=None):
    if manifest['contract'] != contract:
        raise ValueError('Split training contract changed')
    if not 0 < manifest['step'] < contract['total_steps']:
        raise ValueError('Boundary must precede end of training')
    if batches_per_epoch is not None and manifest['batches_per_epoch'] != batches_per_epoch:
        raise ValueError('Dataset length changed at boundary')
    if manifest['epoch'] != (manifest['step'] - 1) // manifest['batches_per_epoch'] + 1:
        raise ValueError('Boundary epoch does not match completed steps')


def export_boundary(trainer, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    state = rng_state(trainer.device)
    trainer.model.save_checkpoint(str(root), tag='boundary', exclude_frozen_parameters=False,
                                  client_state=dict(step=trainer.global_step, epoch=trainer.current_epoch))
    pointer_content = b'boundary\n'
    if trainer.local_rank == 0:
        pointer = root / 'latest.partial'
        pointer.write_bytes(pointer_content)
        pointer.replace(root / 'latest')
    state_path = root / f'rng_rank{trainer.world_rank:03d}.pt'
    torch.save(state, state_path)
    paths = list((root / 'boundary').glob(f'*zero_pp_rank_{trainer.world_rank}_mp_rank_00_*states.pt'))
    if len(paths) != 2 or not any(path.name.endswith('model_states.pt') for path in paths) \
            or not any(path.name.endswith('optim_states.pt') for path in paths):
        raise ValueError('Expected exactly one model and optimizer shard per rank')
    files = {str(path.relative_to(root)): dict(size=path.stat().st_size, sha256=file_hash(path))
             for path in [*paths, state_path]}
    records = [None] * trainer.world_size
    torch.distributed.all_gather_object(records, files)
    manifest = dict(contract=training_contract(trainer), step=trainer.global_step,
                    epoch=trainer.current_epoch, batches_per_epoch=trainer._batches_per_epoch,
                    rank_files=records, common_files={'latest': dict(size=len(pointer_content),
                                              sha256=hashlib.sha256(pointer_content).hexdigest())})
    validate_manifest(manifest, manifest['contract'])
    if trainer.local_rank == 0:
        write_json(root / 'manifest.json', manifest)
    torch.distributed.barrier()
    raise BoundarySaved()


def validate_arguments(args):
    if args.strategy != 'deepspeed' or args.deepspeed_zero_stage != 3 or not args.train_vlm \
            or args.vlm_unfreeze_v_loss_threshold <= 0 or args.resume or args.ema_decay != 0 \
            or args.deepspeed_config or args.seed is None:
        raise ValueError('Split requires fresh delayed-unfreeze ZeRO3, train_vlm, seed, EMA=0, no custom DS config')
    if args.resume_gpu_keepalive:
        raise ValueError('Split training does not use resume GPU keepalive')


def profile_protocol(options):
    if not options.profile:
        return None
    protocol = dict(frozen_steps=options.profile_frozen_steps, warmup=options.profile_warmup,
                    measure_steps=options.profile_steps, trace_steps=options.profile_trace_steps,
                    trace_ranks=[int(rank) for rank in options.profile_trace_ranks.split(',')],
                    memory_budget_fraction=options.profile_memory_budget_fraction)
    if protocol['frozen_steps'] <= 0 or protocol['warmup'] < 0 or protocol['measure_steps'] <= 0 \
            or protocol['trace_steps'] < 0 or not 0 < protocol['memory_budget_fraction'] < 1:
        raise ValueError('Invalid split profile step counts or memory budget')
    ranks = protocol['trace_ranks']
    if len(ranks) != len(set(ranks)) or any(not 0 <= rank < options.nnodes * options.nproc_per_node for rank in ranks):
        raise ValueError('Invalid split profile trace ranks')
    return protocol


def split_profile_options(root, protocol):
    from lerobot.scripts.profile_lola_v07 import parse_options

    options, _ = parse_options([
        '--training-config', str(root / 'source_training_config.json'), '--output', str(root / 'profile'),
        '--warmup', str(protocol['warmup']),
        '--steps', str(protocol['measure_steps'] + protocol['trace_steps']),
        '--trace-steps', str(protocol['trace_steps']), '--trace-at-end',
        '--trace-ranks', ','.join(map(str, protocol['trace_ranks'])), '--zero-hpz-partition-size', '1',
        '--memory-budget-fraction', str(protocol['memory_budget_fraction'])])
    return options


@contextmanager
def split_training_context(training, args, root, stage, bench_options=None):
    manifest = read_json(root / 'manifest.json') if stage == 'trainable' else None
    original_trainer = training.LoLAV07Trainer
    protocol = getattr(args, 'split_profile', None)

    class SplitTrainer(original_trainer):
        def __init__(self, *positional, **kwargs):
            if manifest:
                kwargs['resume_vlm_unfrozen'] = True
            super().__init__(*positional, **kwargs)

        def training_step(self, *positional, **kwargs):
            result = super().training_step(*positional, **kwargs)
            if protocol and not manifest:
                self._pending_deepspeed_unfreeze = self.global_step + 1 == protocol['frozen_steps']
            return result

        def save_checkpoint(self, *positional, **kwargs):
            if not protocol:
                return super().save_checkpoint(*positional, **kwargs)

        def _checkpoint_reasons(self, *positional, **kwargs):
            return [] if protocol else super()._checkpoint_reasons(*positional, **kwargs)

        def _unfreeze_vlm_deepspeed(self):
            if manifest:
                raise RuntimeError('Trainable stage attempted a second unfreeze')
            if self.global_step >= self.total_steps:
                return
            export_boundary(self, root)

        def _setup_deepspeed(self):
            if protocol:
                required = sum(protocol[key] for key in ('frozen_steps', 'warmup', 'measure_steps', 'trace_steps'))
                if self.total_steps <= required:
                    raise ValueError('Original training horizon must exceed the bounded split profile')
            if not manifest:
                return super()._setup_deepspeed()
            validate_manifest(manifest, training_contract(self))
            full_horizon = self.total_steps
            self.total_steps -= manifest['step']
            self.vlm_lr = self.learning_rate * self.config.vlm_lr_mult
            try:
                super()._setup_deepspeed()
            finally:
                self.total_steps = full_horizon
            loaded, _ = self.model.load_checkpoint(str(root), tag='boundary', load_optimizer_states=False,
                                                  load_lr_scheduler_states=False, load_module_strict=True)
            if loaded is None:
                raise RuntimeError('Boundary weights were not loaded')
            for parameter in self.policy.parameters():
                parameter.grad = None
            self.global_step = manifest['step']
            self.current_epoch = manifest['epoch']
            self._pending_deepspeed_unfreeze = False

        def train(self, train_loader, start_step=0, start_epoch=0):
            if manifest:
                validate_manifest(manifest, training_contract(self), train_loader.batch_sampler.full_length)
                state = torch.load(root / f'rng_rank{self.world_rank:03d}.pt', map_location='cpu', weights_only=False)
                restore_rng(state, self.device)
                start_step, start_epoch = self.global_step, self.current_epoch
            try:
                if protocol and manifest:
                    from lerobot.scripts.profile_lola_v07 import profile_policy_state, run_bench

                    state = profile_policy_state(self)
                    if state['vlm_trainable_tensors'] != state['vlm_parameter_tensors']:
                        raise ValueError('Split profile consumer has frozen VLM tensors')
                    write_json(root.parent / 'profile' / f'split_rank{self.world_rank:03d}.json',
                               dict(protocol=protocol, policy_state=state, boundary_step=start_step,
                                    source_sha256=file_hash(__file__), data_policy='lola-split-v1',
                                    handoff_manifest_sha256=file_hash(root / 'manifest.json')))
                    result = run_bench(self, bench_options, train_loader, start_step, start_epoch,
                                       original_trainer.train)
                else:
                    result = super().train(train_loader, start_step=start_step, start_epoch=start_epoch)
            except BoundarySaved:
                if self.interconnect_monitor:
                    self.interconnect_monitor.close()
                if self.use_wandb:
                    training.wandb.finish()
                result = None
                outcome = 'boundary'
            else:
                outcome = 'complete'
            if self.local_rank == 0:
                write_json(root.parent / f'{stage}_node{self.world_rank // int(os.environ["LOCAL_WORLD_SIZE"]):03d}.json',
                           dict(outcome=outcome, step=self.global_step))
            torch.distributed.barrier()
            return result

    def loader(dataset, **kwargs):
        return restartable_loader(dataset, seed=args.seed, **kwargs)

    with patch.object(training, 'LoLAV07Trainer', SplitTrainer), patch.object(training, 'DataLoader', loader):
        yield


def run_stage(options):
    sys.path.insert(0, str(Path(__file__).parent))
    from lerobot.scripts import train_lola_v07_azure as training

    args = training.build_arg_parser().parse_args([])
    vars(args).update(read_json(options.arguments))
    validate_arguments(args)
    protocol = getattr(args, 'split_profile', None)
    bench_options = split_profile_options(options.root, protocol) if protocol else None
    if protocol:
        from lerobot.scripts.profile_lola_v07 import profile_deepspeed_initialization
    if options.stage == 'trainable':
        receipt = read_json(options.root / f'exchange_node{int(os.environ["RANK"]) // int(os.environ["LOCAL_WORLD_SIZE"]):03d}.json')
        if receipt != dict(manifest_sha256=file_hash(options.root / 'handoff/manifest.json'), complete=True):
            raise ValueError('Missing or stale local exchange receipt')
    with patch.object(training, 'build_arg_parser') as parser, \
            split_training_context(training, args, options.root / 'handoff', options.stage, bench_options), \
            profile_deepspeed_initialization(bench_options) if protocol else nullcontext():
        parser.return_value.parse_args.return_value = args
        training.main()


def gather_nodes(value):
    records = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(records, value)
    return records


@contextmanager
def node_group(options, port_offset):
    torch.distributed.init_process_group(
        'gloo', init_method=f'tcp://{options.master_addr}:{options.master_port + port_offset}',
        rank=options.node_rank, world_size=options.nnodes, timeout=timedelta(seconds=options.exchange_timeout))
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


def finish_frozen(root, code, node_rank, nproc_per_node):
    status_path = root / f'frozen_node{node_rank:03d}.json'
    status = read_json(status_path) if status_path.is_file() else dict(outcome='missing')
    statuses = gather_nodes(dict(code=code, status=status))
    if any(record['code'] != 0 for record in statuses):
        raise RuntimeError(f'Frozen-stage node failure: {statuses}')
    if len({json.dumps(record['status'], sort_keys=True) for record in statuses}) != 1:
        raise ValueError('Frozen-stage nodes disagree on outcome/step')
    if status['outcome'] == 'complete':
        return False
    if status['outcome'] != 'boundary':
        raise ValueError('Frozen-stage completion receipt missing')
    handoff = root / 'handoff'
    manifest = read_json(handoff / 'manifest.json')
    hashes = gather_nodes(file_hash(handoff / 'manifest.json'))
    if len(set(hashes)) != 1 or manifest['step'] != status['step']:
        raise ValueError('Nodes disagree on handoff manifest')
    if set(manifest['common_files']) != {'latest'}:
        raise ValueError('Boundary pointer missing from manifest')
    checked_files(handoff, manifest['common_files'])
    if (handoff / 'latest').read_text().strip() != 'boundary':
        raise ValueError('Boundary pointer has an unexpected tag')
    expected_world = len(statuses) * nproc_per_node
    if len(manifest['rank_files']) != expected_world or manifest['contract']['world_size'] != expected_world:
        raise ValueError('Handoff rank coverage changed')
    node_files = []
    for node in range(len(statuses)):
        files = {}
        for rank in range(node * nproc_per_node, (node + 1) * nproc_per_node):
            files.update(manifest['rank_files'][rank])
        node_files.append(files)
    exchange_files(handoff, node_files, node_rank)
    write_json(root / f'exchange_node{node_rank:03d}.json', dict(manifest_sha256=hashes[0], complete=True))
    torch.distributed.barrier()
    return True


def build_plan(options, overrides):
    from lerobot.scripts.profile_lola_v07 import resolve_training_arguments

    if options.nnodes != 2 or options.nproc_per_node < 1 or not 0 <= options.node_rank < options.nnodes:
        raise ValueError('This launcher requires two nodes and a positive local rank count')
    if not 1024 <= options.master_port <= 65531 or options.exchange_timeout <= 0:
        raise ValueError('Invalid master port or exchange timeout')
    mount, mirror, output = options.mount_prefix.resolve(), options.local_mirror.resolve(), options.output.resolve()
    if mirror.is_relative_to(mount) or mount.is_relative_to(mirror) \
            or output == mount or not output.is_relative_to(mount):
        raise ValueError('Use a new output below mount_prefix and a separate local mirror')
    _, _, args = resolve_training_arguments(read_json(options.training_config), overrides)
    args.resume_gpu_keepalive = False
    validate_arguments(args)
    root = mirror / output.relative_to(mount)
    inputs = {}
    for name in ('dataset_root', 'vlm_path'):
        value = getattr(args, name)
        if not value or not Path(value).is_absolute():
            raise ValueError(f'Absolute {name} required')
        source = Path(value).resolve()
        local = mirror / source.relative_to(mount) if source.is_relative_to(mount) else source
        if root.is_relative_to(local) or local.is_relative_to(root):
            raise ValueError(f'Output overlaps {name}')
        inputs[name] = dict(source=str(source), local=str(local))
        setattr(args, name, str(local))
    args.ckpt_dir = str(root / 'checkpoints')
    args.split_data_policy = 'lola-split-v1'
    protocol = profile_protocol(options)
    if protocol:
        args.split_profile = protocol
        args.disable_wandb = True
    stages = {}
    for stage, offset in (('frozen', 0), ('trainable', 1)):
        stages[stage] = [sys.executable, '-m', 'torch.distributed.run', f'--nnodes={options.nnodes}',
                         f'--nproc_per_node={options.nproc_per_node}', f'--node_rank={options.node_rank}',
                         f'--master_addr={options.master_addr}', f'--master_port={options.master_port + offset}',
                         '--max_restarts=0', '-m', 'lerobot.scripts.train_lola_v07_split', 'stage',
                         '--stage', stage, '--root', str(root), '--arguments', str(root / 'arguments.json')]
    return dict(root=str(root), output=str(output), inputs=inputs, arguments=vars(args), stages=stages)


def run_job(options, overrides):
    from lerobot.scripts import download_azure_azcopy as transfers
    from lerobot.scripts.profile_lola_v07 import run_profile_child

    plan = build_plan(options, overrides)
    print(json.dumps(plan, indent=2), flush=True)
    if options.dry_run:
        return 0
    root = Path(plan['root'])
    if options.output.exists():
        raise FileExistsError(f'Output already exists: {options.output}')
    root.mkdir(parents=True, exist_ok=False)
    node_io = root / f'io_node{options.node_rank:03d}'
    node_io.mkdir()
    write_json(node_io / 'plan.json', plan)
    write_json(root / 'arguments.json', plan['arguments'])
    write_json(root / 'source_training_config.json', read_json(options.training_config))
    azcopy = transfers.install_azcopy(str(options.local_mirror / 'bin/azcopy'))
    for name, paths in plan['inputs'].items():
        if paths['source'] != paths['local']:
            url = transfers.resolve_blob_ref(paths['source'], options.storage_account,
                                             options.storage_container, str(options.mount_prefix))
            if not transfers.download_with_fallback(azcopy, url, paths['local'], account=options.storage_account,
                                                    container=options.storage_container,
                                                    mount_prefix=str(options.mount_prefix), dir_transfer=True):
                raise RuntimeError(f'Failed to stage {name}')
        required = 'meta/info.json' if name == 'dataset_root' else 'config.json'
        if not (Path(paths['local']) / required).is_file():
            raise FileNotFoundError(f'Missing {name}/{required}')
    blob_output = transfers.resolve_blob_ref(str(options.output), options.storage_account,
                                             options.storage_container, str(options.mount_prefix))
    checkpoints = root / 'checkpoints'
    checkpoints.mkdir()
    watcher_command = [sys.executable, str(Path(__file__).with_name('checkpoint_upload_watcher.py')),
                       '--local_root', str(checkpoints), '--blob_base', f'{blob_output}/checkpoints',
                       '--keep_last', '2', '--drain_timeout', '7200', '--azcopy-path', azcopy]
    result = 1
    with (node_io / 'watcher.log').open('x') as log:
        watcher = subprocess.Popen(watcher_command, stdout=log, stderr=subprocess.STDOUT)
        try:
            frozen_code = run_profile_child(plan['stages']['frozen'], node_io / 'frozen.log')
            with node_group(options, 2):
                needs_trainable = finish_frozen(root, frozen_code, options.node_rank, options.nproc_per_node)
            if needs_trainable:
                code = run_profile_child(plan['stages']['trainable'], node_io / 'trainable.log')
                with node_group(options, 3):
                    status_path = root / f'trainable_node{options.node_rank:03d}.json'
                    status = read_json(status_path) if status_path.is_file() else dict(outcome='missing')
                    statuses = gather_nodes(dict(code=code, status=status))
                    if any(record['code'] or record['status']['outcome'] != 'complete' for record in statuses) \
                            or len({record['status'].get('step') for record in statuses}) != 1:
                        raise RuntimeError(f'Trainable-stage node failure: {statuses}')
            result = 0
        finally:
            (checkpoints / '_upload_drain').touch()
            try:
                watcher_code = watcher.wait(timeout=7500)
            except subprocess.TimeoutExpired:
                watcher.terminate()
                try:
                    watcher.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    watcher.kill()
                    watcher.wait()
                watcher_code = 1
            result = result or watcher_code
            if plan['arguments'].get('split_profile'):
                artifacts = [root / 'profile' / f'rank{rank:03d}'
                             for rank in range(options.node_rank * options.nproc_per_node,
                                               (options.node_rank + 1) * options.nproc_per_node)]
                artifacts.extend((root / 'profile').glob(f'split_rank*.json'))
                if options.node_rank == 0:
                    artifacts.extend([root / 'profile/runtime_config', root / 'source_training_config.json',
                                      root / 'arguments.json'])
                for artifact in artifacts:
                    if artifact.exists() and not transfers.run_azcopy_transfer(
                            azcopy, str(artifact), f'{blob_output}/{artifact.relative_to(root)}',
                            overwrite='true', max_retries=3):
                        result = result or 1
            write_json(node_io / 'exit.json', dict(exit_code=result, watcher_exit_code=watcher_code))
            if not transfers.run_azcopy_transfer(azcopy, str(node_io), f'{blob_output}/{node_io.name}',
                                                 overwrite='true', max_retries=3):
                result = result or 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('stage')
    stage.add_argument('--stage', choices=('frozen', 'trainable'), required=True)
    stage.add_argument('--root', type=Path, required=True)
    stage.add_argument('--arguments', type=Path, required=True)
    run = commands.add_parser('run')
    run.add_argument('--training-config', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--nnodes', type=int, default=2)
    run.add_argument('--nproc-per-node', type=int, default=8)
    run.add_argument('--node-rank', type=int, required=True)
    run.add_argument('--master-addr', required=True)
    run.add_argument('--master-port', type=int, default=9951)
    run.add_argument('--exchange-timeout', type=int, default=3600)
    run.add_argument('--mount-prefix', type=Path, default=Path('/mnt/wangxiaofa'))
    run.add_argument('--local-mirror', type=Path, default=Path('/scratch/lola_profile_mirror'))
    run.add_argument('--storage-account', default='azsussc')
    run.add_argument('--storage-container', default='v-wangxiaofa')
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--profile', action='store_true')
    run.add_argument('--profile-frozen-steps', type=int, default=100)
    run.add_argument('--profile-warmup', type=int, default=40)
    run.add_argument('--profile-steps', type=int, default=120)
    run.add_argument('--profile-trace-steps', type=int, default=3)
    run.add_argument('--profile-trace-ranks', default='0,8')
    run.add_argument('--profile-memory-budget-fraction', type=float, default=0.90)
    options, overrides = parser.parse_known_args()
    if overrides[:1] == ['--']:
        overrides = overrides[1:]
    if options.command == 'stage':
        if overrides:
            parser.error('stage does not accept training overrides')
        run_stage(options)
        return 0
    return run_job(options, overrides)


if __name__ == '__main__':
    sys.exit(main())
