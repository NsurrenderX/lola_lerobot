# LoLA V07 Forward Optimization and Distributed Profile Bench

## Scope

The bench imports the production trainer and calls its existing setup, data,
forward, backward, clipping, optimizer, EMA and resume paths. It performs real
training updates in memory, but disables normal training checkpoint saves,
including the final save. By default it writes no model checkpoints. The opt-in
unfreeze diagnostic permits the production method's temporary ZeRO3 roundtrip
inside the new profile output, never in a source training run.
It never writes resume history into the source checkpoint run.
Use a separate diagnostic job, not a live training process.

The saved training configuration supplies defaults; arguments after `--`
override the production trainer arguments. The original training horizon is
retained for learning-rate and warmup semantics. `--steps` counts productive
bench steps after `--warmup`; it does not replace the trainer's `max_steps`.
A checkpoint near the end of its training horizon can end the bench early;
that is reported as a failure, not a complete measurement.

## Integration Decision, 2026-09-20

The selected training recipe is corrected grouped vision SDPA plus retention
of the last 12 vision blocks. Keep global/language checkpointing enabled,
DiT checkpointing disabled, ZeRO3 live/reuse at 2e9/2e9, parameter persistence
at zero, and allgather/reduce/prefetch buckets at 5e8. Retention stores
activations; it does not release trainable vision weights.

The target workload is two nodes, sixteen A100 40 GB GPUs, BF16, batch 32 per
rank (global 512), the recorded Cosmos3-Nano/CALVIN shapes and checkpoint.
Other batch sizes, resolutions or hardware must be rechecked against the
90% target. The measured 74.12% estimate is not a hard memory cap or a bound
on initialization/checkpoint-saving peaks.

### Results and Disposition

Cluster results use all 200 measured steps after 40 warmup steps, without
traces. The time metric is the mean per-step maximum across ranks of
step plus data wait. Slow samples and periodic cleanup remain included.

| Comparison | Mean step time | Recorded peak estimate | Decision |
| --- | ---: | ---: | --- |
| A07: grouping off, retain 12 | 5.657391 s | 75.30% | Reference |
| B08: corrected grouping on, retain 12 | 4.979472 s | 74.12% | Adopt B recipe |
| C09: grouping on, intended live/reuse 3e9 | 4.989423 s | 79.76% | Do not adopt |
| Grouping on, retain 18 | 5.188514 s | 77.69% | Do not adopt |
| Grouping on, retain 24 | 6.849667 s | 80.96% | Do not adopt |

B versus A reduced mean time by 11.98% and increased estimated throughput by
13.61%; all four 50-step windows favored B. This is one job per setting, not
independent repeated-job proof. C had no overall gain and used more memory;
its external JSON bytes were not archived by the cluster runner. V24's
post-hoc favorable windows showed only 0.83-1.55% and do not replace its poor
full-run result. The newly stalled V12 job is excluded; B08 is the V12 reference.

Local cache repeats used the same physical A6000 GPU0, restored module weights,
synthetic inputs and seeds, four warmup pairs and 18 measured AB/BA pairs per
mode. Both GPUs were empty before and after repeat 02, not continuously monitored.
Positive percentages below mean lower mean elapsed time.

| Cache candidate | Mode | Run 01 reduction | Run 02 reduction | Decision |
| --- | --- | ---: | ---: | --- |
| Vision metadata | B32 forward | 3.44% | -0.41% | Hold |
| Vision metadata | B32 forward/backward | -4.42% | -3.85% | Hold |
| Vision metadata | B1 no-grad forward | 5.19% | 6.91% | Keep experimental, no shared-path integration |
| DiT RoPE | B32 forward | 3.86% | 1.63% | Defer small gain |
| DiT RoPE | B32 forward/backward | 4.89% | 1.22% | Defer small gain |
| DiT RoPE | B1 no-grad forward | 0.31% | 2.14% | No consistently greater-than-2% gain |

Full-size BF16 outputs and active gradients matched exactly in both local runs.
Vision candidate forward/backward spikes repeated at iteration 5, case 2:
832.599 ms and 866.640 ms. They cannot be dismissed as the other GPU's load.
The useful B1 metadata result does not qualify the shared training path or
prove full-policy latency gains. Neither new cache was copied into production.

The existing fixed-step Euler loop and precomputed mask remain unchanged.
The existing DiT CUDA Graph helper is now reachable from the CALVIN evaluator.
Its historical local full-policy bundle measured 645.7 -> 183.7 ms with exact
actions; the old grouped-SDPA flag silently fell back in that experiment.
This supports the graph path, not an isolated vision-grouping inference claim
or a CALVIN success-rate improvement. No graph is installed in training.

Detailed immutable reports remain under `/data_16T/lola_util/profile_amlt/`
for corrected ABC07/08/09 and V12/V18/V24, and under
`/data_16T/lola_util/profile_local_20260918/cache_ab_gpu0_01/` and
`/data_16T/lola_util/profile_local_20260918/cache_ab_gpu0_02/` for the local repeats.
The older source hashes describe their original executions, not this integration.

### Training and Profile Launchers

[test_azure_v07c.sh](../../src/lerobot/scripts/test_azure_v07c.sh) is the production
TRAINING launcher despite its name. It now defaults to grouped SDPA, retain 12,
and reduce/allgather buckets 5e8. The same effective vision arguments are passed
to both the training process and resume-search configuration reconstruction.
No optimizer settings, batch size, data, weights or sampling parameters change.
Existing explicitly supplied options still override launcher defaults.

For a baseline, append `--no_vision_batched_sdpa --vision_no_checkpoint_layers 0`.
To disable vision checkpointing entirely, `--no_vision_gradient_checkpointing`
sets retained layers to zero in this launcher, avoiding contradictory settings.
Direct Python trainer and model defaults remain unchanged for compatibility.
Do not pass the experimental C JSON when reproducing B; a custom DeepSpeed
configuration still overrides the generated configuration.

[profile_azure_v07c.sh](../../src/lerobot/scripts/profile_azure_v07c.sh) accepts
`--validated-optimizations` before `--`. This selects grouping, retain 12,
reduce/allgather buckets 5e8 and a default 0.90 budget. Explicit trainer options
after `--` override the preset. Without the flag, historical profile behavior
is unchanged. The preset does not remove a saved custom DeepSpeed JSON.

```bash
bash src/lerobot/scripts/profile_azure_v07c.sh \
  --nnodes 2 --nproc_per_node 8 --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port 9901 \
  --python "$PYTHON_BIN" \
  --training-config "$TRAINING_CONFIG" --output "$NEW_PROFILE_OUTPUT" \
  --validated-optimizations --warmup 40 --steps 200 --trace-steps 0 \
  -- --strategy deepspeed --deepspeed_zero_stage 3 --batch_size 32 \
  --resume "$CHECKPOINT_TAG"
```

Use staged local paths in that example; add the existing localized IO options
for blob storage. The budget is checked at step boundaries and stops the profile
on any-rank excess, not a hard allocator cap. Production training does not gain
an automatic 90% stop from this preset. No new cloud run was launched.

### CALVIN Evaluation Launcher

[eval_lola_v07_summary.sh](../../src/lerobot/scripts/eval_lola_v07_summary.sh)
launches the adjacent CALVIN summary evaluator with grouped SDPA and DiT CUDA
Graph enabled. The evaluator file is OUTSIDE this Git repository at its parent;
deploy the updated evaluator too, or select it using `--eval-script PATH` or
`LOLA_EVAL_SCRIPT`. The launcher defaults `LOLA_LEROBOT_SRC` to this repository's
source directory rather than the evaluator's old machine-specific fallback.
The independent `lola-alpha` copy and legacy validation scripts are unchanged.
Use a CALVIN-capable interpreter with the existing simulator, Hydra and OmegaConf
dependencies plus this LoLA source. The local `lerobot-gcr3` training environment
used for the focused tests is missing Hydra; it is not a verified full CALVIN
runtime. No packages or interpreter settings were changed by this integration.

```bash
bash src/lerobot/scripts/eval_lola_v07_summary.sh \
  --python "$PYTHON_BIN" --nproc_per_node 1 \
  --eval-script "$SUMMARY_EVALUATOR" -- \
  --training_config "$TRAINING_CONFIG" --checkpoint_path "$CHECKPOINT_TAG" \
  --vlm_path "$VLM_PATH" --dataset_root "$DATASET_ROOT" \
  --dataset_dir "$CALVIN_DATASET_DIR" --eval_sequences_path "$EVAL_SEQUENCES" \
  --eval_dir "$NEW_EVAL_DIR"
```

All existing official-input, checkpoint/config, EMA, null-state and protocol
checks remain enabled. Supply the same approved evaluation recipe as before:
the bash does NOT set action execution length, integration steps, thresholds,
seed, sequence count or summary-variant permission. In particular the existing
summary evaluator's integration-step default is 3; the historical timing used
10 steps and must not be claimed for this different setting.

Use `--no_dit_cuda_graph` and/or `--no_vision_batched_sdpa` after `--` to opt out.
Direct evaluator invocation defaults to no graph and checkpoint-config grouping.
The graph wrapper is attached only after weight/EMA loading, device placement
and eval mode. Capture is lazy; first capture is not steady-state latency.
Optimization choices are printed and bound into the evaluation contract, so use
a fresh evaluation directory rather than merging into an older contract.

### Integration Verification

Focused regression covers production bash -> resume arguments -> actual model
configuration, opt-out precedence, profile/localized IO, CALVIN launch arguments
and contract fields, grouped attention outputs/gradients, selective recomputation,
real BF16 DiT graph replay, shape changes and training fallback. Shell syntax
is checked for all three launchers. The CALVIN CLI/contract tests isolate those
functions without loading the simulator. Full CALVIN rollouts and a new full
sixteen-GPU training run were not executed for this integration.

## hpZ A/B Profile, 2026-09-21

This experiment retains global 16-rank ZeRO-3 optimizer/gradient sharding.
The candidate adds node-local secondary parameter partitions with
`zero_hpz_partition_size=8`; it is NOT MiCS or an outer DDP wrapper.
Official DeepSpeed 0.19.7 removed MiCS; no downgrade or quantization is needed
for hpZ. No hpZ throughput or full-model memory result is established yet.

Use `--zero-hpz-partition-size 1` for A and `8` for B, BEFORE `--`.
Omitting this option preserves the original configuration. The override is
applied immediately before DeepSpeed initialization, AFTER custom JSON merging.
It does not modify the source JSON or production training defaults. Explicit
hpZ profiles reject non-BF16/non-ZeRO3, quantization, offload and MiCS settings.
hpZ8 also requires multiple nodes with eight ranks each. Before measurement,
the bench verifies the actual secondary groups and their physical hostnames.
Each rank manifest records the final submitted `deepspeed_config` and the
gathered `hpz_topology`; summarization rejects cross-rank disagreements.

Both nodes should execute this SAME loop in one cluster allocation. Set
`NODE_RANK` to 0/1 and `MASTER_ADDR` to node 0's address. In AMLT templates,
map them from `$$AZUREML_CR_NODE_RANK` and `$$AZ_BATCHAI_JOB_MASTER_NODE_IP`
using the existing template escaping. The paths below match the previous
cluster experiments. Use a NEW output root for every attempt.

```bash
set -euo pipefail
PROFILE_ROOT=/mnt/wangxiaofa/profiles/lola_hpz_ab_01
for HPZ_SIZE in 1 8; do
  bash src/lerobot/scripts/profile_azure_v07c.sh \
    --nnodes 2 --nproc_per_node 8 --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" --master_port "$((9900 + HPZ_SIZE))" \
    --python /home/aiscuser/.conda/envs/lerobot/bin/python \
    --localize_io --storage_account azsussc --storage_container v-wangxiaofa \
    --mount_prefix /mnt/wangxiaofa --local_mirror /scratch/lola_profile_mirror \
    --training-config /mnt/wangxiaofa/checkpoints/lola07/lola-v07-azure-20260829_221224/training_config.json \
    --output "$PROFILE_ROOT/hpz${HPZ_SIZE}" \
    --validated-optimizations --zero-hpz-partition-size "$HPZ_SIZE" \
    --warmup 40 --steps 200 --trace-steps 0 --memory-budget-fraction 0.90 \
    -- --strategy deepspeed --deepspeed_zero_stage 3 --batch_size 32 --seed 0 \
    --ema_decay 0 \
    --dataset_root /mnt/wangxiaofa/robot_dataset/lerobot-format-v30/calvin_task_ABC_D_training_v4 \
    --vlm_path /mnt/wangxiaofa/utils/Cosmos3-Nano \
    --resume /mnt/wangxiaofa/checkpoints/lola07/lola-v07-azure-20260829_221224/step_032929
done
```

Check that the ORIGINAL saved training configuration has `deepspeed_config`
null/absent, global/vision checkpointing enabled and DiT checkpointing disabled.
Do not use the historical C override or a profile's runtime configuration.
Both runs use grouped SDPA, retain 12 vision blocks, 5e8 buckets, live/reuse
2e9, global batch 512, the same seed/checkpoint and zero checkpoint writes.
Each run starts a fresh process and restores the same optimizer state; B does
not continue A's in-memory training. No checkpoint-saving compatibility is
claimed by this read-only profile.

Collect all 16 rank directories for each run, then invoke the existing
`profile_lola_v07.py summarize DIRECTORY` separately for `hpz1` and `hpz8`.
Compare the mean rank-max step-plus-data time, estimated samples/s, maximum
memory-budget fraction and allocator retries over all 200 measured steps.
Keep slow steps; do not relax the 90% budget or change batch size after a
failure. The budget check is at step boundaries, not a hard allocator cap.
A single A-then-B pair is diagnostic, not repeated-job proof. Local tests
check configuration and wiring; two local GPUs cannot measure this 2x8 topology.

## Independent Unfreeze Profile, 2026-09-21

Use `--unfreeze-after N` to diagnose the fresh frozen-to-trainable lifecycle.
It runs the existing production trainer, including actual frozen updates and
the original loss-threshold calculation/collective. Only the deferred trigger
flag is overridden after `training_step`: unfreeze happens exactly after update
N, irrespective of loss. Native threshold messages before N are not an actual
engine rebuild. No loss, gradient, optimizer or model implementation is replaced.

The existing `_unfreeze_vlm_deepspeed` method performs temporary checkpoint
save, cross-node shard exchange, old-engine destruction, optimizer memory
release, GC reconfiguration, engine rebuild, weight reload and EMA rebind.
The recorder releases its old-engine references before this call and attaches
timers to the new engine afterward. An unexpected second replacement fails.
The stable profile mode is unchanged and still rejects an engine replacement.

Requirements: fresh step zero, DeepSpeed ZeRO3, `train_vlm=True`, and a positive
delayed-unfreeze threshold. A saved resume path is rejected, not silently cleared;
pass `--resume ''` explicitly. This diagnostic does not fix auto-resume matching.
Use `--localize_io` for multi-node jobs: temporary shards are scoped to
`NEW_DIR/unfreeze_checkpoint`, and their blob exchange prefix is
`NEW_DIR/unfreeze_exchange`. An inherited production `LOLA_CKPT_BLOB_BASE` is
not used. Normal checkpoint/final saves remain disabled. Temporary leftovers
on a node or after failure stay inside this experiment, not the source run;
allow sufficient local disk and blob space for full model/optimizer shards.

Run the same command on both A100 nodes, with `NODE_RANK=0/1` and a common
`MASTER_ADDR`. This reproduces the real log's switch at update 590 and retains
the original training horizon. Use a fresh output root for every attempt.

```bash
bash src/lerobot/scripts/profile_azure_v07c.sh \
  --nnodes 2 --nproc_per_node 8 --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port 9911 \
  --python /home/aiscuser/.conda/envs/lerobot/bin/python \
  --localize_io --storage_account azsussc --storage_container v-wangxiaofa \
  --mount_prefix /mnt/wangxiaofa --local_mirror /scratch/lola_profile_mirror \
  --training-config /mnt/wangxiaofa/checkpoints/lola07/lola-v07-azure-20260920_224336/training_config.json \
  --output /mnt/wangxiaofa/profiles/lola_unfreeze_01 \
  --unfreeze-after 590 --validated-optimizations --zero-hpz-partition-size 1 \
  --warmup 40 --steps 200 --trace-steps 0 --memory-budget-fraction 0.90 \
  -- --resume '' --strategy deepspeed --deepspeed_zero_stage 3 \
  --train_vlm --vlm_unfreeze_v_loss_threshold 0.6 --batch_size 32 --seed 0 \
  --ema_decay 0 \
  --dataset_root /mnt/wangxiaofa/robot_dataset/lerobot-format-v30/calvin_task_ABC_D_training_v5_1 \
  --vlm_path /mnt/wangxiaofa/utils/Cosmos3-Nano
```

Before launch, confirm the saved config has no custom DeepSpeed override,
global/vision GC on and DiT GC off. No historical checkpoint is loaded: the
VLM loads its pretrained backbone and the rest follows fresh production setup.
This requires 830 updates, not 240: frozen 1-589, transition 590, post-unfreeze
warmup 591-630, measurement 631-830. `--unfreeze-after 40` is a shorter lifecycle
diagnostic, not a reproduction of the 590-update optimizer/allocator history.
To inspect the periodic cleanup at update 1000 as well, use a separately named
run with `--steps 500`; do not silently change an active protocol.

Each rank adds `unfreeze.json` with before/after policy state, effective
DeepSpeed configs, transition duration and completion status. `steps.jsonl`
adds phase labels, attention-call/fallback counts and fallback metadata. State
receipts include actual GC flags, patched attention count, selected backend
and trainable VLM tensor/parameter counts (using `ds_numel`, not empty ZeRO3
parameter views). The hooks add small CPU overhead; they are diagnostic only.

Collect all 16 ranks and summarize normally. The summary validates the single
transition, matching rank receipts and the full phase/step sequence, then
computes throughput only from untraced post-unfreeze measurement steps. It
reports transition duration separately and checks the memory budget over all
phases, including the transition. The 90% check is an estimate at boundaries,
not a hard cap on transient non-PyTorch or initial model-loading allocations.

For a separate CUDA/NCCL trace run, use `--trace-steps 3 --steps 203` with a new
output root; this leaves 200 untraced measurement steps. Tracing begins only
after the post-unfreeze warmup. Do not compare this fresh v5_1 lifecycle to the
historical v4 checkpoint profile as though only the unfreeze flag differed.

Local verification: 26 CPU profile tests passed; a real two-A6000 tiny BF16
Qwen3-VL run completed 2 frozen + 1 warmup + 4 measured updates, including the
production ZeRO3 save/rebuild/reload. All 35 VLM tensors became trainable;
grouped attention had zero fallback, selective GC remained `[true, false]`,
and no model checkpoint files remained in that shared local test output.
This uses local DeepSpeed 0.18.8/PyTorch 2.11.0 and does not validate A100
throughput, cluster DeepSpeed 0.19.7, full-model memory or actual blob exchange.

## Local Results, 2026-09-19

Historical caveat added 2026-09-20: the old attention patch fell back whenever
`output_hidden_states` was forwarded, even when false. The labels below record
requested options, not proof that grouping executed. Keep these historical
measurements, but do not attribute their timing differences to grouped SDPA.
The corrected wrapper-level tests and new ABC protocol are described below.

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
the next timed step. Without the optional memory budget there is no added
per-step collective; the budget adds an all-reduce outside the step timer.
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
image segments into one SDPA call per distinct length per vision layer, retaining
the original image boundaries and token order. Mixed lengths are grouped without
padding, resampling, or cross-image attention. Nonpositive lengths, nonzero
training dropout, other attention backends and unsupported call options fall
back to the original implementation.
The corrected patch allows only the forwarded `output_hidden_states` metadata;
unknown options, including `attention_mask` and `output_attentions`, still go
through the original attention with their arguments unchanged. Do not replace
this with unconditional removal of arbitrary keyword arguments.
It requires the Qwen3-VL vision tower and does not install Flash Attention.
For a distributed A/B, add `--vision-batched-sdpa` before `--` and use a new
output directory, the same checkpoint, seed and data configuration. The production
trainer also accepts `--vision_batched_sdpa` (after `--` when using this bench).

`LoLAV07Config.vision_gradient_checkpointing` defaults to true. The production
argument `--no_vision_gradient_checkpointing` disables recomputation only in the
vision tower, including its individual blocks. Language checkpointing remains
enabled when the existing global setting enables it; DiT behavior is unchanged.
The choice is reapplied when the trainer unfreezes the VLM. Neither optimization
changes parameter names or shapes, and both are off by default for old jobs.

For finer control, `--vision_no_checkpoint_layers N` retains activations only
for the last N vision blocks (default 0). Other vision blocks and the language
model keep checkpointing. Negative counts, counts beyond the visual depth and
combining a positive count with full vision checkpointing disablement are rejected.

### Corrected Grouped-SDPA ABC Protocol, 2026-09-20

This ABC protocol differs from the earlier memory-budget probe below: ALL three
groups retain the last 12 vision blocks. A/B isolate the grouped-SDPA switch;
B/C isolate the two ZeRO retention thresholds. Use the same corrected repository
revision and dependencies for all six nodes, with two nodes/eight GPUs per node
in each separate AMLT job. Do not combine ranks from different groups.

| Group | Grouped SDPA | Last vision blocks retained | Max live / reuse parameters | Output |
| --- | --- | --- | --- | --- |
| A | Off | 12 | 2e9 / 2e9 | `lola_groupfix_A_01` |
| B | On | 12 | 2e9 / 2e9 | `lola_groupfix_B_01` |
| C | On | 12 | 3e9 / 3e9 | `lola_groupfix_C_01` |

Run from the repository root. C uses the checked-in
`deepspeed_lola_zero3_c.json`, which must exist in that same checkout on both
nodes. It provides the WHOLE `zero_optimization` dictionary because the trainer
does a shallow merge. Persistence stays 0 and all three bucket sizes stay 5e8;
only live/reuse increase from 2e9 to 3e9. The localizer does not copy or hash
custom DS JSON. Retain the exact JSON and its SHA256 with the run's configuration
evidence, as well as the repository revision; a manifest path alone is insufficient.

Use the ORIGINAL checkpoint's training configuration shown below, not an old
profile's runtime configuration. Confirm that its `training_args` has
`vision_batched_sdpa` false/absent and `deepspeed_config` null/absent. Omission of
the grouping flag does NOT override an inherited true value. Keep global and
vision checkpointing enabled and DiT checkpointing disabled. The listed paths
match the 04/05/06 source paths; adjust only if the mounted inputs actually move.
Every output must be unused, including the short preflight output.

Before the long jobs, run B's command once with output
`/mnt/wangxiaofa/profiles/lola_groupfix_B_trace_01` and replace its measurement
options with `--warmup 3 --steps 5 --trace-steps 3 --trace-ranks 0,8`.
This is an execution-path check, not a throughput benchmark. With 64 images,
two distinct lengths and 27 visual blocks, the expected VISION SDPA calls per
step are 54 in forward, 30 in backward recomputation and 54 backward attention
operations; over three traced steps this is 252 forward/recompute and 162
backward operations. The old per-image path would be 8064 and 5184, respectively.
Count within the vision ranges or distinguish the vision backend from language
attention; total model SDPA counts also include language/DiT. Inspect actual
shapes/backend and both traced ranks before accepting these expectations.

After that check, use the following long-run commands. Each retains 40 warmup
plus 200 measured steps, without a profiler, and the same 90% per-rank budget.
Do not change the budget or batch size to rescue a failed C run.

**A: grouped off, default ZeRO retention**

```yaml
- >-
  bash src/lerobot/scripts/profile_azure_v07c.sh
  --nnodes 2 --nproc_per_node 8
  --node_rank $$AZUREML_CR_NODE_RANK
  --master_addr $$AZ_BATCHAI_JOB_MASTER_NODE_IP --master_port 9901
  --python /home/aiscuser/.conda/envs/lerobot/bin/python
  --localize_io --storage_account azsussc --storage_container v-wangxiaofa
  --mount_prefix /mnt/wangxiaofa --local_mirror /scratch/lola_profile_mirror
  --training-config /mnt/wangxiaofa/checkpoints/lola07/lola-v07-azure-20260829_221224/training_config.json
  --output /mnt/wangxiaofa/profiles/lola_groupfix_A_01
  --warmup 40 --steps 200 --trace-steps 0 --trace-ranks 0,8
  --memory-budget-fraction 0.90
  --
  --strategy deepspeed --deepspeed_zero_stage 3 --batch_size 32 --seed 0
  --vision_no_checkpoint_layers 12
  --deepspeed_reduce_bucket_size 500000000 --deepspeed_allgather_bucket_size 500000000
  --dataset_root /mnt/wangxiaofa/robot_dataset/lerobot-format-v30/calvin_task_ABC_D_training_v4
  --vlm_path /mnt/wangxiaofa/utils/Cosmos3-Nano
  --resume /mnt/wangxiaofa/checkpoints/lola07/lola-v07-azure-20260829_221224/step_032929
```

**B: grouped on, default ZeRO retention**

```yaml
- >-
  bash src/lerobot/scripts/profile_azure_v07c.sh
  --nnodes 2 --nproc_per_node 8
  --node_rank $$AZUREML_CR_NODE_RANK
  --master_addr $$AZ_BATCHAI_JOB_MASTER_NODE_IP --master_port 9901
  --python /home/aiscuser/.conda/envs/lerobot/bin/python
  --localize_io --storage_account azsussc --storage_container v-wangxiaofa
  --mount_prefix /mnt/wangxiaofa --local_mirror /scratch/lola_profile_mirror
  --training-config /mnt/wangxiaofa/checkpoints/lola07/lola-v07-azure-20260829_221224/training_config.json
  --output /mnt/wangxiaofa/profiles/lola_groupfix_B_01
  --warmup 40 --steps 200 --trace-steps 0 --trace-ranks 0,8
  --memory-budget-fraction 0.90 --vision-batched-sdpa
  --
  --strategy deepspeed --deepspeed_zero_stage 3 --batch_size 32 --seed 0
  --vision_no_checkpoint_layers 12
  --deepspeed_reduce_bucket_size 500000000 --deepspeed_allgather_bucket_size 500000000
  --dataset_root /mnt/wangxiaofa/robot_dataset/lerobot-format-v30/calvin_task_ABC_D_training_v4
  --vlm_path /mnt/wangxiaofa/utils/Cosmos3-Nano
  --resume /mnt/wangxiaofa/checkpoints/lola07/lola-v07-azure-20260829_221224/step_032929
```

**C: grouped on, larger ZeRO retention**

```yaml
- >-
  bash src/lerobot/scripts/profile_azure_v07c.sh
  --nnodes 2 --nproc_per_node 8
  --node_rank $$AZUREML_CR_NODE_RANK
  --master_addr $$AZ_BATCHAI_JOB_MASTER_NODE_IP --master_port 9901
  --python /home/aiscuser/.conda/envs/lerobot/bin/python
  --localize_io --storage_account azsussc --storage_container v-wangxiaofa
  --mount_prefix /mnt/wangxiaofa --local_mirror /scratch/lola_profile_mirror
  --training-config /mnt/wangxiaofa/checkpoints/lola07/lola-v07-azure-20260829_221224/training_config.json
  --output /mnt/wangxiaofa/profiles/lola_groupfix_C_01
  --warmup 40 --steps 200 --trace-steps 0 --trace-ranks 0,8
  --memory-budget-fraction 0.90 --vision-batched-sdpa
  --
  --strategy deepspeed --deepspeed_zero_stage 3 --batch_size 32 --seed 0
  --vision_no_checkpoint_layers 12
  --deepspeed_reduce_bucket_size 500000000 --deepspeed_allgather_bucket_size 500000000
  --deepspeed_config ./deepspeed_lola_zero3_c.json
  --dataset_root /mnt/wangxiaofa/robot_dataset/lerobot-format-v30/calvin_task_ABC_D_training_v4
  --vlm_path /mnt/wangxiaofa/utils/Cosmos3-Nano
  --resume /mnt/wangxiaofa/checkpoints/lola07/lola-v07-azure-20260829_221224/step_032929
```

The doubled dollar syntax is for AMLT substitution only. In a normal shell
replace node rank/master with actual values or single-dollar environment variables.
The grouping flag is before `--`; the underscore-form checkpointing and DS
options are after it. Do not reuse profile04/05/06 output names.

Local fix validation: the new metadata regression failed on the original code
with 8 versus 4 SDPA calls for BOTH true and false. After the fix, all eight
forward tests passed, including actual LoLA `prepare_vlm_inputs` calling a tiny
Cosmos model in hidden-state and hook modes, full output/input/parameter-gradient
comparisons, selective recomputation counts, and unknown-option fallback.
All 16 profiler tests passed, including ABC argument resolution and exact DS
override differences. The real two-GPU BF16 ZeRO-3 vision smoke passed five
updates per rank with four images/two lengths and forwarded metadata: 30 SDPA
calls per rank across forward/recomputation, versus 60 without grouping.
This establishes local execution correctness, not full-checkpoint BF16 gradient
equivalence, A100 speedup or cross-node memory acceptance.

### A100 40 GB Memory Budget

The target is at most 90% device memory on EVERY rank, including startup/warmup,
not just mean allocated memory. Do not disable all VLM checkpointing or change
ZeRO persistence settings merely because the baseline has spare memory.

Use three fresh, otherwise matched profile runs:

| Run | Additional options before `--` | Additional trainer options after `--` |
| --- | --- | --- |
| Baseline | `--memory-budget-fraction 0.90` | None |
| Grouped SDPA | `--memory-budget-fraction 0.90 --vision-batched-sdpa` | None |
| Grouped SDPA, last 12 vision blocks retained | `--memory-budget-fraction 0.90 --vision-batched-sdpa` | `--vision_no_checkpoint_layers 12` |

For a throughput comparison use `--trace-steps 0` on all three. Keep batch size,
seed, input data, checkpoint and environment unchanged. Retain all startup rows
and assess the full latency distribution as well as any explicitly identified
stable tail; the previous baseline's first 10 steps did not remove all long tails.
Use new output names such as `lola_budget90_baseline_01`,
`lola_budget90_grouped_01` and `lola_budget90_vision_retain12_01`.

The optional budget samples device free/total memory at step boundaries and
records PyTorch peak reserved bytes. Its estimate is the maximum of boundary
device usage and peak reserved bytes plus the larger observed non-PyTorch usage.
Each row records the estimate and its limit. An all-reduce makes every rank fail
together if any rank exceeds the limit; journals/failed status are retained and
the localizing launcher uploads the diagnostics. A completed summary includes
`memory_budget.passed` and `max_estimated_peak_fraction`.

This is a conservative diagnostic estimate, NOT a hard allocation cap or a
continuous measurement of device usage. It can overestimate noncoincident peaks;
short-lived non-PyTorch allocations and model setup before the bench can escape
it. An OOM can occur before the next budget check. Use external GPU-memory
telemetry as well for final A100 acceptance. The budget itself adds a collective,
so enable it in both sides of an A/B and use production wall timing before
claiming throughput. Production training does not inherit this bench-only gate.

If the retained-activation run exceeds 90%, keep vision checkpointing enabled
and evaluate grouped SDPA alone; do not silently increase the budget, shrink the
batch, change data, or turn off language checkpointing. A6000 split-model tests
cannot certify the memory usage of a full A100 ZeRO-3 rank.

After A100 validation, the production `test_azure_v07c.sh` launcher accepts
`--vision_batched_sdpa` and optionally `--vision_no_checkpoint_layers 12`.
Append those trainer flags to the existing production command; do not pass the
bench-only `--memory-budget-fraction` to it. Retain external GPU memory telemetry
for the full training job, including checkpoint saves which the bench disables.

The first local B32 full-checkpoint probe (`training_ab_01`, two A6000 GPUs,
VLM on GPU0 and the remaining model on GPU1) rejected fully disabled vision
checkpointing at an estimated 91.0% peak. GPU0 allocated/reserved peaks rose
from 30.462/34.914 GiB to 37.266/42.801 GiB. The run stopped before its planned
alternating timing phase; these probes do not establish a speedup. The failed
status, step journal and original runner are retained. Partial retention uses
a new run identity, without raising the 90% limit or reducing batch size.

The completed `training_ab_02` (2026-09-19) kept the last 12 of 27 vision
blocks' activations. It used the same B32 checkpoint and three real batches,
with fixed seeds, two warmup iterations and six measured iterations per variant
in rotating order. All 33 steps and bound source hashes were verified.

| Variant | Mean forward + backward (s) | Median (s) | Isolated GPU0 allocated / reserved peak (GiB) |
| --- | --- | --- | --- |
| Baseline | 4.474788 | 4.321858 | 30.462 / 34.914 |
| Grouped SDPA | 4.475336 | 4.474509 | 30.462 / 34.914 |
| Grouped SDPA, retain last 12 | 4.422355 | 4.336371 | 33.086 / 38.418 |

The last variant's mean time decreased 1.17%, but its median did not improve
and paired time reductions ranged from -14.33% to +17.48%. Grouping alone
showed no mean gain. This short, noisy local run does NOT establish a stable
training speedup; keep the options experimental until matched A100 runs show
a useful improvement within budget. Actual image lengths were 32 segments of
144 tokens and 32 of 64 tokens.

The maximum budget estimate across all local steps was 81.76%. Alternating
variants share the allocator cache, so use the isolated peaks for memory
comparisons. All six candidate/batch comparisons had identical losses and
identical gradient samples for all 1,479 parameter tensors (up to 1,024 elements
per tensor); full gradients were checked for finiteness, not full equality.
The run included preprocessing, transfer, forward and backward but no ZeRO,
optimizer, clipping or weight updates. It neither certifies A100 memory nor
training quality. The original failed run remains separate.

Local validation also passed five GPU optimization tests, 15 profiler tests,
and both two-rank ZeRO-3 smoke modes (five updates per rank, budget checks,
trace and allocator snapshot, no model checkpoint writes). The real-vision
smoke used BF16 and mixed checkpointed/non-checkpointed visual blocks.

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
real BF16 DiT graph tests, mixed-length SDPA call counts, and actual selective
recomputation/full-gradient checks. Run `tests/test_lola_profile_bench.py` for recorder,
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
and allocator snapshot, checks the memory budget, bounded exit and verifies no model checkpoint
was written. It tests the real training loop and recorder, not full LoLA
checkpoint restore, A100 performance or cross-node networking.
Replace `--distributed-smoke` with `--distributed-vision-smoke` to exercise
an actual small BF16 Qwen3-VL vision tower with four images/two lengths, forwarded
hidden-state metadata, grouped SDPA and one of two vision blocks retaining
activations under ZeRO-3. The smoke asserts exactly six SDPA calls per update
including checkpoint recomputation, so a silent per-image fallback fails.

Use `--distributed-unfreeze-smoke` for seven updates with that same small vision
tower: two frozen updates with an actual production ZeRO3 roundtrip at step 2,
one post-unfreeze warmup and four measured updates (one traced/excluded).
It verifies newly trainable VLM tensors, grouped-attention/GC retention and
absence of retained model checkpoint shards. Use a new output directory.

Numeric comparisons are required on the deployed hardware/software. Local
action agreement is not a robot task-success evaluation. Do not infer an
A100 training speedup from an A6000 split-model forward-only benchmark.