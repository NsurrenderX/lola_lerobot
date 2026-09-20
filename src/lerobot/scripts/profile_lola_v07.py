"""Bounded, checkpoint-read-only profiling of the production DeepSpeed trainer."""

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import torch


class BenchComplete(Exception):
    pass


def parse_options(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--trace-steps", type=int, default=3)
    parser.add_argument("--trace-ranks", default="0,8")
    parser.add_argument("--sync-phases", action="store_true")
    parser.add_argument("--trace-memory", action="store_true")
    parser.add_argument("--memory-history", action="store_true")
    parser.add_argument("--snapshot-threshold", type=float, default=20.0)
    parser.add_argument("--memory-budget-fraction", type=float, default=None)
    parser.add_argument("--vision-batched-sdpa", action="store_true")
    parser.add_argument("--validated-optimizations", action="store_true",
                        help="Use grouped vision SDPA, retain 12 vision blocks, and check a 90%% memory budget")
    parser.add_argument("--zero-hpz-partition-size", type=int, choices=(1, 8), default=None,
                        help="Profile-only hpZ control: 1 for baseline, 8 for node-local parameter caching")
    parser.add_argument("--dry-run", action="store_true")
    options, trainer_arguments = parser.parse_known_args(arguments)
    if trainer_arguments[:1] == ["--"]:
        trainer_arguments = trainer_arguments[1:]
    if options.validated_optimizations:
        trainer_arguments = ["--vision_batched_sdpa", "--vision_no_checkpoint_layers", "12",
                             "--deepspeed_reduce_bucket_size", "500000000",
                             "--deepspeed_allgather_bucket_size", "500000000", *trainer_arguments]
        if options.memory_budget_fraction is None:
            options.memory_budget_fraction = 0.90
    if options.steps <= 0 or options.warmup < 0 or not 0 <= options.trace_steps <= options.steps:
        parser.error("Require steps > 0, warmup >= 0, and 0 <= trace-steps <= steps")
    if options.memory_budget_fraction is not None and not 0 < options.memory_budget_fraction < 1:
        parser.error("memory-budget-fraction must be between 0 and 1")
    options.trace_ranks = tuple(int(rank) for rank in options.trace_ranks.split(",") if rank)
    return options, trainer_arguments


def configure_profile_deepspeed(config, partition_size, world_size, local_world_size):
    result = deepcopy(config)
    if partition_size is None:
        return result
    if partition_size not in (1, 8):
        raise ValueError("hpZ profile partition size must be 1 or 8")
    zero = result["zero_optimization"]
    if zero["stage"] != 3 or not result.get("bf16", {}).get("enabled"):
        raise ValueError("hpZ comparison requires BF16 ZeRO-3")
    if any(zero.get(name, False) for name in (
            "zero_quantized_weights", "zero_quantized_nontrainable_weights", "zero_quantized_gradients")):
        raise ValueError("hpZ comparison requires all ZeRO quantization options disabled")
    if zero.get("mics_shard_size", -1) > 0:
        raise ValueError("hpZ comparison must not enable MiCS")
    if any(zero.get(name, {}).get("device", "none") != "none"
           for name in ("offload_param", "offload_optimizer") if zero.get(name)):
        raise ValueError("hpZ comparison requires no parameter or optimizer offload")
    if partition_size > 1 and (local_world_size != partition_size or world_size <= local_world_size
                                or world_size % local_world_size):
        raise ValueError("hpZ8 comparison requires multiple nodes with 8 ranks per node")
    zero["zero_hpz_partition_size"] = partition_size
    return result


@contextmanager
def profile_deepspeed_initialization(options):
    import deepspeed

    original_initialize = deepspeed.initialize

    def initialize(*args, **kwargs):
        config = configure_profile_deepspeed(
            kwargs["config"], options.zero_hpz_partition_size,
            int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
        kwargs["config"] = config
        receipt = deepcopy(config)
        result = original_initialize(*args, **kwargs)
        result[0]._lola_profile_deepspeed_config = receipt
        return result

    with patch.object(deepspeed, "initialize", initialize):
        yield


def profile_hpz_topology(trainer, partition_size):
    if partition_size is None:
        return None
    parameter = next(trainer.policy.parameters())
    group = getattr(parameter, "ds_zero_param_process_group", None)
    ranks = torch.distributed.get_process_group_ranks(group) if group is not None else []
    local = dict(rank=trainer.world_rank, hostname=socket.gethostname(), local_rank=trainer.local_rank,
                 hpz_group_ranks=ranks, hpz_partition_size=trainer.model.optimizer.zero_hpz_partition_size)
    topology = [None] * trainer.world_size
    torch.distributed.all_gather_object(topology, local)
    for entry in topology:
        if entry["hpz_partition_size"] != partition_size:
            raise ValueError("DeepSpeed did not apply the requested hpZ partition size")
        start = entry["rank"] // partition_size * partition_size
        expected = list(range(start, start + partition_size)) if partition_size > 1 else []
        if entry["hpz_group_ranks"] != expected:
            raise ValueError("Unexpected hpZ rank group")
        if partition_size > 1:
            members = [topology[rank] for rank in expected]
            if len({member["hostname"] for member in members}) != 1 or \
                    sorted(member["local_rank"] for member in members) != list(range(partition_size)):
                raise ValueError("hpZ group must contain all 8 ranks on one node")
    if partition_size > 1 and len({entry["hostname"] for entry in topology}) != trainer.world_size // partition_size:
        raise ValueError("hpZ comparison requires distinct physical nodes")
    return topology


def validate_checkpoint_tag(directory, ranks):
    for rank in ranks:
        shard = Path(directory) / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
        if not shard.is_file():
            raise ValueError(f"Use an explicit checkpoint tag directory; missing {shard.name} in {directory}")


def stage_checkpoint_config(source_tag, local_tag):
    source_tag, local_tag = Path(source_tag).resolve(), Path(local_tag).resolve()
    source = next((path for path in (source_tag / "training_config.json",
                                     source_tag.parent / "training_config.json") if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"Missing checkpoint training_config.json in {source_tag} or its parent")
    content = source.read_bytes()
    snapshot = json.loads(content)
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("lola_config"), dict) or not snapshot["lola_config"]:
        raise ValueError(f"Checkpoint configuration has no nonempty lola_config: {source}")
    destination = source
    if source_tag != local_tag:
        local_tag.mkdir(parents=True, exist_ok=True)
        destination = local_tag / "training_config.json"
        destination.write_bytes(content)
    return dict(source=str(source), local=str(destination), sha256=hashlib.sha256(content).hexdigest())


def resolve_training_arguments(snapshot, trainer_arguments):
    sys.path.insert(0, str(Path(__file__).parent))
    from lerobot.scripts import train_lola_v07_azure as training

    parser = training.build_arg_parser()
    parser.set_defaults(**snapshot["training_args"])
    return training, parser, parser.parse_args(trainer_arguments)


def run_profile_child(command, console_path):
    with console_path.open("x", buffering=1) as console:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", bufsize=1, start_new_session=True) as process:
            def forward_signal(signum, frame):
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signum)
                    except ProcessLookupError:
                        pass

            handlers = {signum: signal.signal(signum, forward_signal)
                        for signum in (signal.SIGTERM, signal.SIGINT)}
            try:
                for line in process.stdout:
                    console.write(line)
                    print(line, end="", flush=True)
                code = process.wait()
                return code if code >= 0 else 128 - code
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)


def localize_main(arguments=None):
    parser = argparse.ArgumentParser(description="Stage node-local profile IO and drain uploads on exit")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--node_rank", type=int, required=True)
    parser.add_argument("--master_addr", required=True)
    parser.add_argument("--master_port", type=int, required=True)
    parser.add_argument("--storage_account", required=True)
    parser.add_argument("--storage_container", required=True)
    parser.add_argument("--mount_prefix", type=Path, default=Path("/mnt/wangxiaofa"))
    parser.add_argument("--local_mirror", type=Path, default=Path("/scratch/lola_profile_mirror"))
    parser.add_argument("--azcopy_path", type=Path)
    settings, remaining = parser.parse_known_args(arguments)
    options, trainer_arguments = parse_options(remaining)
    if settings.nnodes <= 0 or settings.nproc_per_node <= 0 or not 0 <= settings.node_rank < settings.nnodes:
        parser.error("Invalid node topology")
    if not settings.mount_prefix.is_absolute() or not settings.local_mirror.is_absolute():
        parser.error("mount_prefix and local_mirror must be absolute paths")
    mount = settings.mount_prefix.resolve()
    mirror = settings.local_mirror.resolve()
    if mirror.is_relative_to(mount) or mount.is_relative_to(mirror):
        parser.error("local_mirror and mount_prefix must not overlap")
    requested_output = options.output.resolve()
    if requested_output == mount or not requested_output.is_relative_to(mount):
        parser.error("With --localize_io, --output must be a new directory below --mount_prefix")

    def local_path(value):
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f"Use absolute input paths for localized IO: {value}")
        path = path.resolve()
        return mirror / path.relative_to(mount) if path.is_relative_to(mount) else path

    source_config = options.training_config.read_text()
    snapshot = json.loads(source_config)
    _, _, effective = resolve_training_arguments(snapshot, trainer_arguments)
    if effective.strategy != "deepspeed" or effective.deepspeed_zero_stage != 3:
        parser.error("Localized profile requires DeepSpeed ZeRO-3")
    if not effective.dataset_root or not effective.vlm_path:
        parser.error("Localized profile requires dataset_root and vlm_path in config or CLI")
    output = local_path(requested_output)
    sources = {name: getattr(effective, name) for name in ("dataset_root", "vlm_path", "resume")
               if getattr(effective, name)}
    inputs = {name: local_path(value) for name, value in sources.items()}
    for name, path in inputs.items():
        protected = path.parent if name == "resume" else path
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            parser.error(f"Profile output must not overlap the {name} input")
    start_rank = settings.node_rank * settings.nproc_per_node
    ranks = list(range(start_rank, start_rank + settings.nproc_per_node))
    config_path = output / "source_training_config.json"
    options.training_config = config_path
    options.output = output
    command = [sys.executable, "-m", "torch.distributed.run",
               f"--nnodes={settings.nnodes}", f"--nproc_per_node={settings.nproc_per_node}",
               f"--node_rank={settings.node_rank}", f"--master_addr={settings.master_addr}",
               f"--master_port={settings.master_port}", "--max_restarts=0", str(Path(__file__).resolve())]
    for name, value in vars(options).items():
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, ",".join(map(str, value)) if isinstance(value, tuple) else str(value)])
    command.extend(["--", *trainer_arguments])
    for name, path in inputs.items():
        command.extend([f"--{name}", str(path)])
    plan = dict(node_rank=settings.node_rank, ranks=ranks, blob_output=str(requested_output),
                local_output=str(output), inputs={name: dict(source=sources[name], local=str(path))
                                                 for name, path in inputs.items()}, command=command)
    print(json.dumps(plan, indent=2), flush=True)
    if options.dry_run:
        return 0

    from lerobot.scripts import download_azure_azcopy as transfers

    node_output = output / f"io_node{settings.node_rank:03d}"
    node_output.mkdir(parents=True, exist_ok=False)
    config_path.write_text(source_config)
    (node_output / "plan.json").write_text(json.dumps(plan, indent=2))
    azcopy = None
    child_code = 1
    status = dict(node_rank=settings.node_rank, phase="staging", started_at=time.time())
    status_path = node_output / "status.json"
    status_path.write_text(json.dumps(status, indent=2))
    try:
        azcopy = transfers.install_azcopy(str(settings.azcopy_path or mirror / "bin/azcopy"))
        if "resume" in inputs:
            checkpoint_config = stage_checkpoint_config(sources["resume"], inputs["resume"])
            (node_output / "checkpoint_config.json").write_text(json.dumps(checkpoint_config, indent=2))
            print(f"[localize] checkpoint config: {checkpoint_config['source']} -> {checkpoint_config['local']}", flush=True)
        for name, path in inputs.items():
            source = Path(sources[name]).resolve()
            if source.is_relative_to(mount):
                patterns = [pattern for rank in ranks for pattern in
                            (f"*zero_pp_rank_{rank}_mp_rank_00_*", f"ema_rank_{rank}.pt")] if name == "resume" else None
                extra = ["--include-pattern=" + ";".join(patterns)] if patterns else []
                url = transfers.resolve_blob_ref(str(source), settings.storage_account,
                                                 settings.storage_container, str(mount))
                if not transfers.download_with_fallback(
                        azcopy, url, str(path), account=settings.storage_account,
                        container=settings.storage_container, mount_prefix=str(mount),
                        dir_transfer=True, extra_copy_args=extra, include_patterns=patterns):
                    raise RuntimeError(f"Failed to localize {name}")
            required_file = {"dataset_root": "meta/info.json", "vlm_path": "config.json"}.get(name)
            if required_file and not (path / required_file).is_file():
                raise FileNotFoundError(f"Missing {name}: {path / required_file}")
            if name == "resume":
                validate_checkpoint_tag(path, ranks)
                for rank in ranks:
                    if not any(path.glob(f"*zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")):
                        raise FileNotFoundError(f"Missing optimizer shard for rank {rank} in {path}")
        status["phase"] = "running"
        status_path.write_text(json.dumps(status, indent=2))
        child_code = run_profile_child(command, node_output / "console.log")
    except Exception:
        (node_output / "error.log").write_text(traceback.format_exc())
        traceback.print_exc()
    status.update(phase="exited", child_exit_code=child_code, finished_at=time.time())
    status_path.write_text(json.dumps(status, indent=2))
    if azcopy is None:
        return child_code or 1

    artifacts = [output / f"rank{rank:03d}" for rank in ranks]
    if settings.node_rank == 0:
        artifacts.extend([config_path, output / "runtime_config"])
    artifacts.append(node_output)
    uploaded, failed = [], []
    blob_output = transfers.resolve_blob_ref(str(requested_output), settings.storage_account,
                                             settings.storage_container, str(mount))
    for path in artifacts:
        if not path.exists():
            continue
        try:
            success = transfers.run_azcopy_transfer(
                azcopy, str(path), f"{blob_output}/{path.name}", overwrite="true", max_retries=3)
        except Exception:
            traceback.print_exc()
            success = False
        (uploaded if success else failed).append(path.name)
    receipt = node_output / "upload_status.json"
    receipt.write_text(json.dumps(dict(node_rank=settings.node_rank, child_exit_code=child_code,
                                      upload_complete=not failed, uploaded=uploaded, failed=failed,
                                      finished_at=time.time()), indent=2))
    try:
        receipt_ok = transfers.run_azcopy_transfer(
            azcopy, str(receipt), f"{blob_output}/{node_output.name}/{receipt.name}",
            overwrite="true", max_retries=3)
    except Exception:
        traceback.print_exc()
        receipt_ok = False
    if failed or not receipt_ok:
        print(f"Profile upload failed; local artifacts retained at {output}", file=sys.stderr)
        return child_code or 1
    return child_code


def memory_stats(device):
    stats = torch.cuda.memory_stats(device)
    keys = ("allocated_bytes.all.current", "reserved_bytes.all.current",
            "allocated_bytes.all.peak", "reserved_bytes.all.peak",
            "active_bytes.all.current", "inactive_split_bytes.all.current",
            "inactive_split_bytes.all.peak", "num_alloc_retries", "num_ooms",
            "num_sync_all_streams", "num_device_alloc", "num_device_free")
    return {key: stats.get(key) for key in keys}


def counter_deltas(before, after):
    return {key: after[key] - before[key]
            for key in after if key.startswith("num_")
            and before.get(key) is not None and after[key] is not None}


def device_memory_sample(device, stats):
    free, total = torch.cuda.mem_get_info(device)
    used = total - free
    return dict(total_bytes=total, used_bytes=used,
                non_torch_bytes=max(0, used - stats["reserved_bytes.all.current"]))


def assess_memory_budget(before, after, stats, limit):
    if before["total_bytes"] != after["total_bytes"]:
        raise ValueError("Device memory capacity changed during a step")
    estimated_peak = max(before["used_bytes"], after["used_bytes"],
                         stats["reserved_bytes.all.peak"] + max(before["non_torch_bytes"], after["non_torch_bytes"]))
    fraction = estimated_peak / after["total_bytes"]
    return dict(limit_fraction=limit, estimated_peak_fraction=fraction,
                estimated_peak_bytes=estimated_peak, exceeded=fraction > limit,
                device_before=before, device_after=after)


class BenchRecorder:
    def __init__(self, trainer, options):
        self.trainer = trainer
        self.engine = getattr(trainer, "model", None)
        self.options = options
        self.rank = trainer.world_rank
        self.device = trainer.device
        self.output = options.output / f"rank{self.rank:03d}"
        self.output.mkdir(parents=True, exist_ok=False)
        self.journal = (self.output / "steps.jsonl").open("x", buffering=1)
        self.count = 0
        self.profiler = None
        self.tracing = False
        self.trace_started = False
        self.snapshot_count = 0
        self.host_times = {}
        self.shapes = {}
        self.active = False
        self.memory_budget_fraction = getattr(options, "memory_budget_fraction", None)

    @contextmanager
    def stage(self, name):
        if not self.active:
            yield
            return
        synchronized = self.options.sync_phases and name in ("forward", "backward", "optimizer")
        if synchronized:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.profiler.record_function(f"bench/{name}"):
            yield
        if synchronized:
            torch.cuda.synchronize(self.device)
        self.host_times[name] = self.host_times.get(name, 0.0) + time.perf_counter() - started

    def wrap(self, function, name, shapes=False):
        def wrapped(*args, **kwargs):
            with self.stage(name):
                output = function(*args, **kwargs)
            if shapes and self.active:
                self.shapes = {key: list(value.shape) for key, value in output.items()
                               if isinstance(value, torch.Tensor)}
            return output
        return wrapped

    def start_trace(self):
        if self.trace_started:
            return
        self.trace_started = True
        if self.rank not in self.options.trace_ranks or not self.options.trace_steps:
            return
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=self.options.trace_memory, profile_memory=self.options.trace_memory,
            with_stack=False,
        )
        self.profiler.start()
        self.tracing = True

    def stop_trace(self):
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler.export_chrome_trace(str(self.output / "trace.json"))
            self.profiler = None
            self.tracing = False

    def batches(self, loader):
        iterator = iter(loader)
        while self.count < self.options.warmup + self.options.steps:
            if getattr(self.trainer, "model", None) is not self.engine:
                raise RuntimeError("DeepSpeed engine changed during bench; use a stable frozen/unfrozen checkpoint")
            if self.count == self.options.warmup:
                self.start_trace()
            started = time.perf_counter()
            with torch.profiler.record_function("bench/data_wait"):
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
            data_seconds = time.perf_counter() - started
            before_step = self.trainer.global_step
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            before_memory = memory_stats(self.device)
            device_before = device_memory_sample(self.device, before_memory) if self.memory_budget_fraction else None
            self.host_times = {}
            self.shapes = {}
            self.active = True
            started_ns = time.time_ns()
            started = time.perf_counter()
            yield batch
            torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            self.active = False
            if self.trainer.global_step == before_step:
                continue
            after_memory = memory_stats(self.device)
            deltas = counter_deltas(before_memory, after_memory)
            budget = None
            if self.memory_budget_fraction:
                budget = assess_memory_budget(device_before, device_memory_sample(self.device, after_memory),
                                              after_memory, self.memory_budget_fraction)
                exceeded = torch.tensor(int(budget["exceeded"]), device=self.device)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(exceeded, op=torch.distributed.ReduceOp.MAX)
                budget["any_rank_exceeded"] = bool(exceeded.item())
            snapshot_needed = (self.options.memory_history and self.rank in self.options.trace_ranks
                               and self.snapshot_count < 2
                               and (elapsed >= self.options.snapshot_threshold or deltas.get("num_alloc_retries", 0)))
            row = dict(rank=self.rank, bench_step=self.count, global_step=self.trainer.global_step,
                       warmup=self.count < self.options.warmup,
                       traced=self.options.warmup <= self.count < self.options.warmup + self.options.trace_steps,
                       profiler_active=self.tracing,
                       sync_phases=self.options.sync_phases, wall_start_ns=started_ns,
                       step_s=elapsed, data_wait_s=data_seconds,
                       host_stage_s=self.host_times, shapes=self.shapes,
                       memory_before=before_memory, memory_after=after_memory, counter_delta=deltas,
                       snapshot_saved=bool(snapshot_needed))
            if budget is not None:
                row["memory_budget"] = budget
            self.journal.write(json.dumps(row) + "\n")
            if snapshot_needed:
                torch.cuda.memory._dump_snapshot(str(self.output / f"memory_step{self.count}.pickle"))
                self.snapshot_count += 1
            self.count += 1
            if budget is not None and budget["any_rank_exceeded"]:
                raise RuntimeError(f"Memory budget exceeded on at least one rank; limit={self.memory_budget_fraction:.1%}, "
                                   f"rank {self.rank} estimated peak={budget['estimated_peak_fraction']:.1%}")
            if self.profiler is not None:
                self.profiler.step()
            if self.options.trace_steps and self.count == self.options.warmup + self.options.trace_steps:
                self.stop_trace()
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
        raise BenchComplete()


class ProfileLoader:
    def __init__(self, loader, recorder):
        self.loader = loader
        self.recorder = recorder

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        return self.recorder.batches(self.loader)


def run_bench(trainer, options, loader, start_step, start_epoch, original_train):
    if trainer.strategy != "deepspeed":
        raise ValueError("This bench requires the production DeepSpeed training path")
    hpz_topology = profile_hpz_topology(trainer, options.zero_hpz_partition_size)
    recorder = BenchRecorder(trainer, options)
    trainer.resume_save_dir = None
    trainer.ckpt_dir = str(options.output / "runtime_config")
    trainer.use_wandb = False
    trainer.save_every_n_steps = trainer.save_every_n_epochs = trainer.save_every_n_seconds = None
    source = Path(__file__)
    from lerobot.scripts import train_lola_v07_azure as training
    from lerobot.policies.lola_v07 import forward_optimizations
    import deepspeed
    manifest = dict(rank=trainer.world_rank, world_size=trainer.world_size,
                    hostname=socket.gethostname(), device=torch.cuda.get_device_name(trainer.device),
                    torch=torch.__version__, cuda=torch.version.cuda, deepspeed=deepspeed.__version__,
                    allocator_backend=torch.cuda.memory.get_allocator_backend(),
                    allocator_config=os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
                    bench=vars(options), training_args=trainer.training_args,
                    deepspeed_config=trainer.model._lola_profile_deepspeed_config,
                    hpz_topology=hpz_topology,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    trainer_sha256=hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest(),
                    optimizations_sha256=hashlib.sha256(Path(forward_optimizations.__file__).read_bytes()).hexdigest(),
                    policy_sha256=hashlib.sha256(Path(sys.modules[trainer.policy.__module__].__file__).read_bytes()).hexdigest(),
                    training_config_sha256=hashlib.sha256(options.training_config.read_bytes()).hexdigest(),
                    start_step=start_step, checkpoint_writes=False)
    (recorder.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    memory_history = options.memory_history and trainer.world_rank in options.trace_ranks
    status = "failed"
    try:
        if memory_history:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        with ExitStack() as stack:
            for owner, attribute, label in (
                (trainer, "training_step", "forward"), (trainer.model, "backward", "backward"),
                (trainer.model, "step", "optimizer"), (trainer.policy, "forward", "policy"),
                (trainer.policy.vlm.visual, "forward", "vision"),
                (trainer.policy.vlm.language_model, "forward", "language"),
                (trainer.policy.model.vlm_bridge, "forward", "bridge"),
                (trainer.policy.model.dit, "forward", "dit"),
            ):
                stack.enter_context(patch.object(owner, attribute, recorder.wrap(getattr(owner, attribute), label)))
            stack.enter_context(patch.object(trainer, "preprocessor", recorder.wrap(trainer.preprocessor, "preprocess", shapes=True)))
            stack.enter_context(patch.object(trainer, "save_checkpoint", lambda *args, **kwargs: None))
            stack.enter_context(patch.object(trainer, "_checkpoint_reasons", lambda *args, **kwargs: []))
            try:
                original_train(trainer, ProfileLoader(loader, recorder), start_step, start_epoch)
            except BenchComplete:
                pass
        if recorder.count != options.warmup + options.steps:
            raise RuntimeError(f"Training ended early: {recorder.count} productive steps")
        status = "complete"
    finally:
        recorder.stop_trace()
        if memory_history:
            torch.cuda.memory._dump_snapshot(str(recorder.output / "memory_final.pickle"))
            torch.cuda.memory._record_memory_history(enabled=None)
        recorder.journal.close()
        (recorder.output / "status.json").write_text(json.dumps(dict(status=status, steps=recorder.count)))
        if trainer.interconnect_monitor:
            trainer.interconnect_monitor.close()


def summarize(directory):
    manifests = [json.loads(path.read_text()) for path in sorted(directory.glob("rank*/manifest.json"))]
    if not manifests:
        raise ValueError("No rank manifests found")
    world_size = manifests[0]["world_size"]
    if {entry["rank"] for entry in manifests} != set(range(world_size)):
        raise ValueError("Missing ranks; copy both nodes' rank directories before summarizing")
    for key in ("world_size", "source_sha256", "trainer_sha256", "policy_sha256",
                "optimizations_sha256", "training_config_sha256", "start_step", "training_args", "bench",
                "deepspeed_config", "hpz_topology"):
        if any(entry.get(key) != manifests[0].get(key) for entry in manifests):
            raise ValueError(f"Rank manifests disagree on {key}")
    per_rank = {}
    by_step = {}
    rows_by_rank = {}
    excluded_steps = set()
    for manifest in manifests:
        rank = manifest["rank"]
        root = directory / f"rank{rank:03d}"
        if json.loads((root / "status.json").read_text())["status"] != "complete":
            raise ValueError(f"Rank {rank} did not complete")
        rows = [json.loads(line) for line in (root / "steps.jsonl").read_text().splitlines()]
        if len({row["global_step"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate global steps on rank {rank}")
        rows_by_rank[rank] = rows
        excluded_steps.update(row["global_step"] + 1 for row in rows if row.get("snapshot_saved"))
    for rank, rows in rows_by_rank.items():
        stable = [row for row in rows if not row["warmup"] and not row["traced"]
                  and row["global_step"] not in excluded_steps]
        if not stable:
            raise ValueError("No untraced post-warmup steps")
        durations = sorted(row["step_s"] + row["data_wait_s"] for row in stable)
        per_rank[rank] = dict(count=len(stable), mean_s=statistics.mean(durations),
                              median_s=statistics.median(durations),
                              p95_s=durations[min(len(durations) - 1, int(0.95 * len(durations)))],
                              max_s=max(durations),
                              alloc_retries=sum(row["counter_delta"].get("num_alloc_retries", 0) for row in rows))
        for row in stable:
            by_step.setdefault(row["global_step"], []).append(row["step_s"] + row["data_wait_s"])
    if any(len(values) != world_size for values in by_step.values()):
        raise ValueError("Rank step sets differ")
    rank_max_mean = statistics.mean(max(values) for values in by_step.values())
    result = dict(per_rank=per_rank, mean_rank_max_step_plus_data_s=rank_max_mean,
                  estimated_samples_per_second=world_size * manifests[0]["training_args"]["batch_size"] / rank_max_mean,
                  caveat="Rank-max step+data estimate; boundary synchronization and instrumentation perturb training. Host stages overlap and are not additive.")
    budgets = [row["memory_budget"] for rows in rows_by_rank.values() for row in rows if "memory_budget" in row]
    if budgets:
        if len(budgets) != sum(map(len, rows_by_rank.values())) or any(
                budget["exceeded"] or budget["any_rank_exceeded"] for budget in budgets):
            raise ValueError("Missing or exceeded memory budget records")
        if len({budget["limit_fraction"] for budget in budgets}) != 1:
            raise ValueError("Memory budget limits differ")
        result["memory_budget"] = dict(limit_fraction=budgets[0]["limit_fraction"], passed=True,
                                       max_estimated_peak_fraction=max(budget["estimated_peak_fraction"] for budget in budgets))
    (directory / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    if sys.argv[1:2] == ["localize"]:
        raise SystemExit(localize_main(sys.argv[2:]))
    if sys.argv[1:2] == ["summarize"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("directory", type=Path)
        summarize(parser.parse_args(sys.argv[2:]).directory)
        return
    options, trainer_arguments = parse_options()
    snapshot = json.loads(options.training_config.read_text())
    training, parser, arguments = resolve_training_arguments(snapshot, trainer_arguments)
    arguments.disable_wandb = True
    arguments.resume_gpu_keepalive = False
    arguments.resume_fast_skip = True
    arguments.ckpt_dir = str(options.output / "runtime_config")
    if arguments.strategy != "deepspeed":
        parser.error("Profile bench requires --strategy deepspeed")
    if arguments.resume:
        try:
            validate_checkpoint_tag(arguments.resume, [int(os.environ.get("RANK", "0"))])
        except ValueError as error:
            parser.error(str(error))
    if arguments.resume and (options.output.resolve() == Path(arguments.resume).resolve().parent
                             or Path(arguments.resume).resolve().parent in options.output.resolve().parents):
        parser.error("Bench output must be outside the source checkpoint run directory")
    if options.dry_run:
        print(json.dumps(dict(bench=vars(options), training=vars(arguments), checkpoint_writes=False), indent=2, default=str))
        return

    original_train = training.LoLAV07Trainer.train
    original_build_config = training.build_lola_config

    def build_config(*args, **kwargs):
        result = original_build_config(*args, **kwargs)
        result[0].vision_batched_sdpa = options.vision_batched_sdpa or result[0].vision_batched_sdpa
        return result

    def train(trainer, loader, start_step=0, start_epoch=0):
        return run_bench(trainer, options, loader, start_step, start_epoch, original_train)

    with patch.object(parser, "parse_args", return_value=arguments), \
            patch.object(training, "build_arg_parser", return_value=parser), \
            patch.object(training, "build_lola_config", build_config), \
            patch.object(training, "_resolve_inplace_save_dir", return_value=None), \
            patch.object(training.LoLAV07Trainer, "train", train), \
            profile_deepspeed_initialization(options):
        training.main()


if __name__ == "__main__":
    main()