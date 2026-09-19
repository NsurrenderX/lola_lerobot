# LoLA V07 Forward Optimization and Distributed Profile Bench

## Scope

The bench imports the production trainer and calls its existing setup, data,
forward, backward, clipping, optimizer, EMA and resume paths. It performs real
training updates in memory, but disables all checkpoint saves, including the
final save. It never writes resume history into the source checkpoint run.
Use a separate diagnostic job, not a live training process.

The saved training configuration supplies defaults; arguments after `--`
override the production trainer arguments. The original training horizon is
retained for learning-rate and warmup semantics. `--steps` counts productive
bench steps after `--warmup`; it does not replace the trainer's `max_steps`.
A checkpoint near the end of its training horizon can end the bench early;
that is reported as a failure, not a complete measurement.

## Local Results, 2026-09-19

Same-checkpoint alternating A/B on two RTX A6000 GPUs (VLM on GPU0, remaining
model on GPU1), PyTorch 2.11.0+cu126 and Transformers 5.14.1. Checkpoint
`step_032929`, its saved configuration and CALVIN v4 data were used. Each
variant has two warmup and six measured iterations, without a profiler.

| Path | Mean | Median |
| --- | --- | --- |
| Training forward, B32, original | 1.7314 s | 1.7307 s |
| Training forward, B32, batched vision SDPA | 1.4575 s | 1.3922 s |
| Inference, B1, original | 645.7 ms | 644.9 ms |
| Inference, loop/mask only | 641.8 ms | 637.6 ms |
| Inference, loop/mask + batched vision SDPA | 714.8 ms | 647.1 ms |
| Inference, loop/mask + batched vision SDPA + DiT graph | 183.7 ms | 182.5 ms |

Training forward mean latency decreased 15.8% (18.8% more forwards/second).
Full inference mean latency decreased 71.6% (3.52x speed). The graph captured
once and replayed with new inputs. Vision-only inference had one 1.088-second
outlier; it remains in the mean. There is no demonstrated standalone B1 vision
speedup, and the loop/mask-only difference is small enough to be timing noise.

All compared training scalar metrics and full action tensors were bitwise
equal to the original path in this run. Coverage is three replayed training
batches and one inference sample with three noise seeds, each repeated;
this is not full-model gradient parity or task-success validation. Focused
vision output/gradient tests and a real reduced-size BF16 LoLADiT graph replay
test also passed. The original ten Euler steps and FP32 precision boundaries
were retained. No weights were updated by this forward-only A/B.

Training numbers exclude backward, optimizer, ZeRO communication and steady
DataLoader wait. The results do not establish full-step or 16-A100 speedup.
Compare against the original path in this same A/B, not a previous session's
743-ms baseline. Inputs are preloaded; preprocessing is included in the timed
forward/inference calls, but robot-side action consumption is not.

Raw A/B records, report and reproduction driver are under
`/data_16T/lola_util/profile_local_20260918/` (`forward_ab_01`, `forward_ab.py`).
The successful two-GPU ZeRO-3 integration artifacts are in `bench_smoke_01`.
The smoke exercised five actual updates per rank, selected-rank CUDA/NCCL
tracing, allocator history/final snapshot and bounded exit without checkpoint
writes. Cross-node A100 execution has not been performed locally.

## Two Nodes, Sixteen A100 GPUs

Use identical code, dependencies, configuration and checkpoint on both nodes.
Restore a 16-partition checkpoint with 16 ranks. Activate the same environment
and preserve the working RDMA/NCCL variables from the production job. This
launcher does not force network devices or disable RDMA. The bench loads the
real dataset, not synthetic inputs. `dataset_root` is inherited from the saved
configuration unless explicitly overridden. Moving the configuration JSON does
not relocate any paths stored inside it.

For AMLT, use a single job command that runs once on each node. No preceding
`export` command is required. Enable localized IO to stage blob inputs onto each
node and upload results before exit. Replace the account/container names and
checkpoint/configuration paths with the actual cluster values:

```yaml
- >-
  bash src/lerobot/scripts/profile_azure_v07c.sh
  --nnodes $NODES --nproc_per_node $GPUS
  --node_rank $$AZUREML_CR_NODE_RANK
  --master_addr $$AZ_BATCHAI_JOB_MASTER_NODE_IP --master_port 9901
  --python /home/aiscuser/.conda/envs/lerobot/bin/python
  --localize_io --storage_account YOUR_ACCOUNT --storage_container YOUR_CONTAINER
  --mount_prefix /mnt/wangxiaofa --local_mirror /scratch/lola_profile_mirror
  --training-config /mnt/wangxiaofa/checkpoints/lola07/ACTUAL_RUN/training_config.json
  --output /mnt/wangxiaofa/profiles/lola_baseline_01
  --warmup 10 --steps 50 --trace-steps 3 --trace-ranks 0,8
  --
  --dataset_root /mnt/wangxiaofa/robot_dataset/lerobot-format-v30/calvin_task_ABC_D_training_v4
  --vlm_path /mnt/wangxiaofa/utils/Cosmos3-Nano
  --resume /mnt/wangxiaofa/checkpoints/lola07/ACTUAL_RUN/step_032929
  --strategy deepspeed --deepspeed_zero_stage 3 --batch_size 32
```

Use the dataset version matching the chosen checkpoint/configuration: the local
`step_032929` configuration uses v4, while the separate `log_5_1` run uses v5_1.
The explicit dataset path above must be adjusted accordingly, not blindly reused.

Keep the `$NODES`, `$GPUS` and `$$AZUREML_*` substitution syntax used by the
existing AMLT job. In an ordinary shell, use concrete values or single-dollar
environment expansion instead: `$$` there expands to the shell PID. The
`--python` path must name the same environment as production training; it is
optional when the correct Python is already on PATH. The launcher adds that
interpreter's adjacent `lib` directory to its child process library path.

For direct shell execution on each node, use the same CLI with concrete
values, changing only `--node_rank` from 0 to 1. This non-localizing example
assumes inputs have already been staged and requires manual output collection:

```bash
bash src/lerobot/scripts/profile_azure_v07c.sh \
  --nnodes 2 --nproc_per_node 8 --node_rank 0 \
  --master_addr 10.0.0.1 --master_port 9901 \
  --no_localize_io \
  --training-config /scratch/mirror/checkpoints/run/training_config.json \
  --output /scratch/profiles/lola_baseline_01 \
  --warmup 10 --steps 50 --trace-steps 3 --trace-ranks 0,8 \
  -- \
  --resume /scratch/mirror/checkpoints/run/step_032929 \
  --dataset_root /scratch/mirror/datasets/calvin_training \
  --vlm_path /scratch/mirror/models/Cosmos3-Nano \
  --batch_size 32
```

The default topology is two nodes with eight processes each. CLI values
override environment defaults; the earlier `NNODES`, `NPROC_PER_NODE`,
`NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `TRAINING_CONFIG`, `PROFILE_OUTPUT`
and `PYTHON_BIN` variables remain supported. Launcher options accept either
`--option value` or `--option=value`. Put topology, configuration/output paths
and profile options before `--`, and trainer overrides after it.
Each output rank directory must be new; existing rank directories
are rejected. The output may be shared storage or node-local storage with the
same path. With node-local storage, collect all sixteen rank directories
before summarizing. Never place the output inside the source checkpoint run.

For a no-GPU configuration check, run the Python entrypoint directly with
`--dry-run` and the same arguments. This validates CLI/configuration only,
not checkpoint compatibility, data decoding or multi-node connectivity.

### Localized IO and Persistence

`--localize_io` is opt-in; without it the launcher retains direct-path behavior.
Put its settings before `--`. This mode requires ZeRO-3, mounted filesystem
paths and Azure Managed Identity access via the existing AzCopy helper. It does
not accept blob URLs in place of paths. `--azcopy_path` can select an existing
AzCopy executable; otherwise the helper installs it under the local mirror.

- The configuration JSON is read once from its supplied path and copied to
  `source_training_config.json` under the local output. CLI overrides still win.
- Dataset and VLM directories below `--mount_prefix` are downloaded in full.
  Absolute input paths outside that prefix are treated as already local and
  validated, not guessed or silently remapped. Only `dataset_repo_id` without
  an explicit/inherited `dataset_root` is rejected in localized mode.
- An explicit resume tag is required when resuming. Each node downloads only
  its own model, optimizer and optional EMA shards. Missing model or optimizer
  shards abort before torchrun; a failed resume never falls back to fresh training.
- Checkpoint architecture metadata is staged separately from the profile's CLI
  configuration. The launcher reads `training_config.json` from the SOURCE tag,
  or its parent run directory if absent in the tag, and copies its unchanged
  bytes into the localized tag before downloading shards. The source must contain
  a nonempty `lola_config`; missing/malformed metadata aborts before torchrun,
  even if a stale local copy exists. The trainer's architecture check remains
  enabled. `io_nodeNNN/checkpoint_config.json` records source, local path and SHA256.
  Already-local checkpoints are validated without modifying their files.
- `--output` is the final blob-mounted destination, and must be a new run
  directory below `--mount_prefix`, outside the input directories. For example,
  `/mnt/wangxiaofa/profiles/run01` maps to
  `/scratch/lola_profile_mirror/profiles/run01` during the measurement.
- Use the SAME absolute `--local_mirror` path on both nodes, backed by node-local
  storage, with capacity for the dataset, VLM, local checkpoint shards and traces.
  Downloads and final uploads are outside the measured steps. No GPU keepalive
  workload or checkpoint upload watcher is started.
- After torchrun exits, each node uploads its own rank directories. Node zero
  also uploads the source configuration and runtime configuration. Console logs,
  path/command plans and exit status are saved under `io_node000`, `io_node001`.
  Partial diagnostics are uploaded even after a nonzero training exit.
- Uploads use the existing AzCopy retry helper, then upload a per-node
  `upload_status.json` receipt LAST. Require both receipts with
  `upload_complete: true` and `child_exit_code: 0` before summarizing. A failed
  artifact or receipt upload makes the job exit nonzero; local files are retained.
  AzCopy reports successful transfers; no independent remote hash readback is done.
- Use a fresh blob output name for every run. Remote name uniqueness is the
  caller's responsibility; uploads overwrite files within that destination.
  SIGTERM/SIGINT received while training are forwarded to the child before the
  exit upload. SIGKILL, node loss or an insufficient termination grace period can
  still lose local artifacts; there is no periodic cloud upload during measurement.

Adding `--dry-run` before `--` to the localized launcher prints the effective
paths and generated torchrun command, reading only the small configuration JSON.
It does not install AzCopy, download data, create output directories or run GPU
work. It does not prove that remote data, permissions or checkpoint shards exist.

## Measurements

Each rank writes:

- `manifest.json`: runtime, device, allocator backend/config, effective training
  arguments, source/config hashes and original start step.
- `steps.jsonl`: data wait, synchronized step duration, nested host-stage
  durations, processed tensor shapes, memory current/peak and allocator
  counter deltas. The step includes EMA, regular logging and periodic GC.
- `status.json`: complete/failed and productive step count.
- `trace.json`: only on selected ranks, for the first `--trace-steps` steps
  after warmup. Includes CUDA kernels, NCCL work and named module ranges.

Only step boundaries synchronize by default, never each module. Host-stage
durations are not GPU compute durations, and nested stages are not additive.
`data_wait_s` measures obtaining the next batch, not worker-side decode CPU time.
The first loader iterator creation is setup, outside this timer. Step-boundary
synchronization changes the production pipeline; treat these as diagnostic
measurements and retain a production wall-time baseline.

All ranks mark the trace window as perturbed, including ranks without a
profiler. After trace export, one all-rank barrier keeps export delay out of
the next timed step. There is no added per-step distributed barrier.
Cross-node wall timestamps require synchronized clocks; compare durations
and global step IDs when that cannot be guaranteed.

```bash
python src/lerobot/scripts/profile_lola_v07.py summarize /mnt/wangxiaofa/profiles/lola_baseline_01
```

Summary rejects incomplete or missing ranks and mismatched step sets. It
excludes warmup, trace windows and steps immediately following a snapshot on
any rank, and reports per-rank distributions plus a
rank-max step+data throughput estimate, not an uninstrumented production rate.

## Long-Tail and Allocator Investigation

For intermittent 30-second spikes, use a longer run, for example `--steps 500`.
Counters are recorded on every rank at every productive step, even outside
the short trace window. Correlate `num_alloc_retries`, `num_device_free`,
`num_sync_all_streams`, allocated/reserved bytes, shape changes and stage
times. Missing counters are null, not zero. A reserved-minus-allocated gap
alone does not prove fragmentation. `inactive_split_bytes` is backend-dependent
and may not characterize expandable segments or cudaMallocAsync adequately.

Optional diagnostic flags:

- `--memory-history`: bounded PyTorch allocator history (100,000 entries) on
  trace ranks; save at most two spike/retry snapshots and one final snapshot.
- `--snapshot-threshold 20`: seconds for a spike snapshot trigger.
- `--trace-memory`: enable memory/shape recording during the trace window;
  this adds overhead and may alter tensor lifetimes.
- `--sync-phases`: synchronize forward/backward/optimizer separately. Run as a
  separate diagnostic, since it destroys normal phase overlap.
- `--trace-steps 0`: counters and timings without a CUDA trace.

Memory history uses version-dependent private PyTorch APIs. Snapshot/export
cost is outside the local timed step but can stall peer ranks; the summary
excludes the following step on every rank when it occurs.
PyTorch snapshots do not include all NCCL/driver allocations. Do not change
allocator settings in the first baseline; compare settings in separate runs.
The launcher defaults to `expandable_segments:True` only when no
`PYTORCH_CUDA_ALLOC_CONF` value is supplied.

## Forward Optimizations

The V07 Euler loop now uses a fixed iteration count and constructs its
unchanging special-token mask once per action chunk. FP32 time updates,
precision-isolated encoders/projections and the number of denoising steps
are unchanged.

`LoLAV07Config.vision_batched_sdpa` defaults to false. It batches equal-length
image segments into one SDPA call per vision layer, retaining the original
image boundaries. Unequal lengths, nonzero training dropout, other attention
backends and unsupported call options fall back to the original implementation.
It requires the Qwen3-VL vision tower and does not install Flash Attention.
For a distributed A/B, add `--vision-batched-sdpa` before `--` and use a new
output directory, the same checkpoint, seed and data configuration.

`DiTCUDAGraph` is an explicit inference-only helper. It captures one shape,
copies all new inputs before replay, recaptures on shape changes and falls
back to the original forward when gradients or training are enabled. ZeRO
parameters are rejected. Clear the cache after changing weights or placement.
It must not be enabled in the distributed training bench. First-call capture
and warmup must be excluded from steady-state latency measurements.

```python
from lerobot.policies.lola_v07.forward_optimizations import DiTCUDAGraph

policy.eval()
graph = DiTCUDAGraph(policy.model.dit)
policy.model.dit.forward = graph
```

Restore `policy.model.dit.forward = graph.original` and call `graph.clear()`
to disable it. Capture is lazy on the first no-grad CUDA forward. Do not
share a graph instance between concurrent callers.

## Local Verification

The focused tests require the project environment and `PYTHONPATH=src`.
Run `tests/test_lola_forward_optimizations.py` for vision output/gradient and
real BF16 DiT graph tests, and `tests/test_lola_profile_bench.py` for recorder,
summary, launcher and mocked two-node IO tests (CLI/config precedence, node-local
shards, output isolation, training/upload failures, dry-run and path protection).
The IO tests use temporary files and simulated transfers, not Azure or GPUs.
The latter also has a small actual ZeRO-3 integration mode:

```bash
PYTHONPATH=src:src/lerobot/scripts \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/test_lola_profile_bench.py --distributed-smoke /tmp/lola-smoke-new
```

The smoke runs five tiny-model updates per rank, exports a rank-zero trace
and allocator snapshot, checks bounded exit and verifies no model checkpoint
was written. It tests the real training loop and recorder, not full LoLA
checkpoint restore, A100 performance or cross-node networking.

Numeric comparisons are required on the deployed hardware/software. Local
action agreement is not a robot task-success evaluation. Do not infer an
A100 training speedup from an A6000 split-model forward-only benchmark.