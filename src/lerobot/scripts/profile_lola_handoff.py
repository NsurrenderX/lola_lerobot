"""Trusted, profile-only frozen-boundary handoff and bounded batch replay."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch


class HandoffComplete(Exception):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    from lerobot.scripts import profile_lola_v07, train_lola_v07_azure
    from lerobot.policies.lola_v07 import modeling_lola_v07, forward_optimizations

    modules = (profile_lola_v07, train_lola_v07_azure, modeling_lola_v07, forward_optimizations)
    return {module.__name__: file_hash(module.__file__) for module in modules} | {
        "handoff": file_hash(__file__)}


def capture_rng(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None)


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return deepcopy(value)


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def engine_state(trainer):
    optimizer = trainer.model.optimizer
    state = dict(
        parameters={name: dict(shape=list(getattr(parameter, "ds_shape", parameter.shape)),
                               sha256=tensor_hash(getattr(parameter, "ds_tensor", parameter)))
                    for name, parameter in trainer.policy.named_parameters()},
        buffers={name: tensor_hash(value) for name, value in trainer.policy.named_buffers()},
        fp32=[tensor_hash(value) for value in optimizer.fp32_partitioned_groups_flat],
        optimizer_state=[{key: tensor_hash(value) if isinstance(value, torch.Tensor) else value
                  for key, value in optimizer.optimizer.state.get(parameter, {}).items()}
                 for group in optimizer.optimizer.param_groups for parameter in group["params"]],
        learning_rates=[group["lr"] for group in optimizer.optimizer.param_groups],
        scheduler=trainer.scheduler.state_dict(),
        deepspeed=trainer.model._lola_profile_deepspeed_config,
    )
    return json.loads(json.dumps(state))


def handoff_contract(trainer):
    import deepspeed
    import transformers

    return dict(config=json.loads(json.dumps(asdict(trainer.config), default=str)),
                normalization_stats=json.loads(json.dumps(trainer.dataset_stats, default=lambda value: value.tolist())),
                total_steps=trainer.total_steps, batch_size=trainer.batch_size,
                learning_rate=trainer.learning_rate, warmup_ratio=trainer.warmup_ratio,
                weight_decay=trainer.weight_decay, gradient_clip_val=trainer.gradient_clip_val,
                world_size=trainer.world_size, source_hashes=source_hashes(),
                torch=torch.__version__, deepspeed=deepspeed.__version__, transformers=transformers.__version__)


def export_handoff(trainer, recorder, checkpoint=None):
    root = recorder.options.output / "handoff"
    root.mkdir(exist_ok=True)
    replay_dir = root / "replay"
    replay_dir.mkdir(exist_ok=True)
    target = root / "boundary"
    if checkpoint is None:
        trainer.model.save_checkpoint(str(root), tag="boundary", exclude_frozen_parameters=False)
    else:
        target.mkdir(exist_ok=True)
        for path in Path(checkpoint).glob(f"*zero_pp_rank_{trainer.world_rank}_mp_rank_00_*.pt"):
            os.link(path, target / path.name)
    shards = sorted(target.glob(f"*zero_pp_rank_{trainer.world_rank}_mp_rank_00_*.pt"))
    if len(shards) != 2 or not any(path.name.endswith("optim_states.pt") for path in shards):
        raise RuntimeError("Handoff requires model and optimizer shards for every rank")
    if trainer.local_rank == 0:
        (root / "latest").write_text("boundary")
    bank = []
    for _ in range(recorder.options.replay_batches):
        try:
            bank.append(cpu_copy(next(recorder.source_iterator)))
        except StopIteration as error:
            raise RuntimeError("Not enough batches after the frozen boundary for replay") from error
    payload = dict(batches=bank, rng=capture_rng(trainer.device))
    replay_path = replay_dir / f"rank{trainer.world_rank:03d}.pt"
    torch.save(payload, replay_path)
    local = {str(path.relative_to(root)): file_hash(path) for path in [*shards, replay_path]}
    if trainer.world_rank == 0:
        local["latest"] = file_hash(root / "latest")
    records = [local]
    if torch.distributed.is_initialized():
        records = [None] * trainer.world_size
        torch.distributed.all_gather_object(records, local)
    files = {name: digest for record in records for name, digest in record.items()}
    manifest = dict(version=1, step=trainer.global_step, epoch=trainer.current_epoch,
                    batches_per_epoch=trainer._batches_per_epoch, contract=handoff_contract(trainer),
                    warmup=recorder.options.warmup, steps=recorder.options.steps,
                    trace_steps=recorder.options.trace_steps, trace_at_end=recorder.options.trace_at_end,
                    replay_batches=len(bank), files=files,
                    mode="stop" if recorder.options.stop_at_handoff else "paired",
                    policy="weights-only; reset all Adam state; OneCycleLR over remaining steps")
    if trainer.local_rank == 0:
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    recorder.replay = payload
    recorder.replay_rng_pending = True
    recorder.handoff_manifest = manifest
    return manifest


def validate_handoff(root, ranks, world_size):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["version"] != 1 or manifest["contract"]["world_size"] != world_size:
        raise ValueError("Handoff version/world size mismatch")
    if manifest["contract"]["source_hashes"] != source_hashes():
        raise ValueError("Handoff source code changed; use the exact producer revision")
    if manifest["contract"]["torch"] != torch.__version__:
        raise ValueError("Handoff PyTorch version changed")
    required = {f"boundary/zero_pp_rank_{rank}_mp_rank_00_model_states.pt" for rank in range(world_size)}
    required.add("latest")
    required.update(f"replay/rank{rank:03d}.pt" for rank in ranks)
    if manifest["mode"] == "paired":
        required.update(f"replay/rank{rank:03d}_state.json" for rank in ranks)
    if not required.issubset(manifest["files"]):
        raise ValueError("Incomplete handoff manifest")
    for rank in range(world_size):
        if not any(name.startswith("boundary/") and f"zero_pp_rank_{rank}_mp_rank_00_" in name
                   and name.endswith("optim_states.pt") for name in manifest["files"]):
            raise ValueError("Missing handoff optimizer shard")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in ("boundary", "replay", "latest"):
            raise ValueError("Unsafe handoff path")
        if relative.parts[0] == "replay" and relative.stem.split("_")[0] not in {f"rank{rank:03d}" for rank in ranks}:
            continue
        if file_hash(root / relative) != expected:
            raise ValueError(f"Handoff hash mismatch: {name}")
    if (root / "latest").read_text().strip() != "boundary":
        raise ValueError("Unexpected handoff checkpoint tag")
    return manifest


def validate_producer(root, nodes, ranks_per_node):
    root = Path(root)
    for node in range(nodes):
        receipt = json.loads((root / f"io_node{node:03d}" / "upload_status.json").read_text())
        if receipt["child_exit_code"] != 0 or not receipt["upload_complete"]:
            raise ValueError("Handoff producer failed or did not finish uploading")
        for rank in range(node * ranks_per_node, (node + 1) * ranks_per_node):
            status = json.loads((root / f"rank{rank:03d}" / "status.json").read_text())
            if status["status"] != "complete":
                raise ValueError("Handoff producer rank did not complete")


def verify_contract(trainer, manifest):
    if handoff_contract(trainer) != manifest["contract"]:
        raise ValueError("Handoff training contract differs from producer")
    if not 0 < manifest["step"] < trainer.total_steps:
        raise ValueError("Invalid handoff step/horizon")


@contextmanager
def consumer_context(training, options, manifest):
    original_init = training.LoLAV07Trainer.__init__
    original_setup = training.LoLAV07Trainer._setup_deepspeed

    def initialize(trainer, *args, **kwargs):
        kwargs["resume_vlm_unfrozen"] = True
        original_init(trainer, *args, **kwargs)

    def setup(trainer):
        verify_contract(trainer, manifest)
        total_steps = trainer.total_steps
        trainer.total_steps = total_steps - manifest["step"]
        trainer.vlm_lr = trainer.learning_rate * trainer.config.vlm_lr_mult
        try:
            original_setup(trainer)
        finally:
            trainer.total_steps = total_steps
        load_path, _ = trainer.model.load_checkpoint(
            str(options.handoff_input), tag="boundary", load_optimizer_states=False,
            load_lr_scheduler_states=False, load_module_strict=False)
        if load_path is None:
            raise RuntimeError("Frozen handoff weights could not be restored")
        for parameter in trainer.policy.parameters():
            parameter.grad = None
        trainer.global_step = manifest["step"]
        trainer.current_epoch = manifest["epoch"]
        trainer._vlm_unfrozen = True
        trainer._vlm_delayed_unfreeze = False
        trainer._pending_deepspeed_unfreeze = False

    with patch.object(training.LoLAV07Trainer, "__init__", initialize), \
            patch.object(training.LoLAV07Trainer, "_setup_deepspeed", setup):
        yield


class ReplayLoader:
    def __init__(self, loader, manifest):
        from lerobot.scripts.train_lola_v07_azure import SkipBatchSampler

        self.batch_sampler = SkipBatchSampler(loader.batch_sampler)
        self.length = manifest["batches_per_epoch"]
        if len(loader) != self.length:
            raise ValueError("Handoff dataset length changed")

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(())


def record_trainable_state(trainer, recorder, reference=None):
    state = engine_state(trainer)
    if reference is not None and state != reference:
        raise ValueError("Fresh engine differs from A at the trainable boundary")
    path = recorder.output / "trainable_state.json"
    path.write_text(json.dumps(state, indent=2))
    return state


def publish_trainable_state(trainer, recorder):
    state = record_trainable_state(trainer, recorder)
    root = recorder.options.output / "handoff"
    path = root / "replay" / f"rank{trainer.world_rank:03d}_state.json"
    path.write_text(json.dumps(state, indent=2))
    local = {str(path.relative_to(root)): file_hash(path)}
    records = [local]
    if torch.distributed.is_initialized():
        records = [None] * trainer.world_size
        torch.distributed.all_gather_object(records, local)
    recorder.handoff_manifest["files"].update({name: digest for record in records for name, digest in record.items()})
    if trainer.local_rank == 0:
        (root / "manifest.json").write_text(json.dumps(recorder.handoff_manifest, indent=2))
    if torch.distributed.is_initialized():
        torch.distributed.barrier()