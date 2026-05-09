"""Minimal DeepSpeed ZeRO-2 + OneCycleLR integration test.

Verifies the pattern used in train_lola_azure.py:
  1. get_deepspeed_config() with integer batch sizes (not "auto")
  2. lr_scheduler callable passed to deepspeed.initialize()
  3. model_engine.step() drives both optimizer + scheduler
  4. scheduler.get_last_lr() works for logging
  5. Checkpoint save / load with lr scheduler states

Usage:
    # Single GPU
    python test_deepspeed_minimal.py

    # Multi-GPU (e.g. 4 GPUs)
    torchrun --nproc_per_node=4 test_deepspeed_minimal.py
"""

import os
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR

# ------------------------------------------------------------------
# 1. DeepSpeed config  (same structure as train_lola_azure.py)
# ------------------------------------------------------------------
def get_deepspeed_config(
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    gradient_clip_val: float = 1.0,
    batch_size: int = 4,
    world_size: int = 1,
):
    return {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "contiguous_gradients": True,
            "round_robin_gradients": True,
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": gradient_clip_val,
        "train_batch_size": batch_size * world_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": weight_decay,
            },
        },
    }


# ------------------------------------------------------------------
# 2. Tiny model  (single Linear layer)
# ------------------------------------------------------------------
class TinyModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Linear(dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------
# 3. Synthetic dataset
# ------------------------------------------------------------------
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=256, dim=64):
        self.x = torch.randn(num_samples, dim)
        self.y = (self.x[:, 0] + self.x[:, 1]).unsqueeze(-1)  # simple target

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------
def main():
    import deepspeed

    # --- distributed setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    world_rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)

    # torchrun always sets WORLD_SIZE; we must init process group before deepspeed
    if "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")

    is_main = world_rank == 0

    # --- hyperparams ---
    learning_rate = 1e-3
    batch_size = 8
    total_steps = 40
    warmup_ratio = 0.1
    dim = 64

    # --- dataset / dataloader ---
    dataset = SyntheticDataset(num_samples=256, dim=dim)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        shuffle=False, drop_last=True)

    # --- model ---
    model = TinyModel(dim=dim).cuda()

    # --- deepspeed config ---
    ds_config = get_deepspeed_config(
        learning_rate=learning_rate,
        batch_size=batch_size,
        world_size=world_size,
    )

    # --- lr scheduler callable (key pattern) ---
    # DeepSpeed calls this with the *basic* (unwrapped) optimizer,
    # so OneCycleLR's isinstance(optimizer, Optimizer) check passes.
    def lr_scheduler_callable(optimizer):
        return OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=min(warmup_ratio, 0.1),
            anneal_strategy="cos",
        )

    # --- deepspeed initialize ---
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_config,
        lr_scheduler=lr_scheduler_callable,
        dist_init_required=False,
    )

    if is_main:
        print(f"[Rank {world_rank}] DeepSpeed initialized. Scheduler type: {type(lr_scheduler)}")
        print(f"[Rank {world_rank}] Starting {total_steps} steps of training...")

    # --- training loop ---
    model_engine.train()
    step = 0
    epoch = 0
    prev_lr = None

    while step < total_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_x, batch_y in loader:
            batch_x = batch_x.cuda().to(torch.bfloat16)
            batch_y = batch_y.cuda().to(torch.bfloat16)

            outputs = model_engine(batch_x)
            loss = nn.functional.mse_loss(outputs, batch_y.squeeze(-1))
            model_engine.backward(loss)
            model_engine.step()  # drives optimizer + scheduler internally

            step += 1

            # --- verify scheduler works ---
            lr = lr_scheduler.get_last_lr()[0]
            if is_main and step % 10 == 0:
                print(f"  step {step:3d} | loss={loss.item():.4f} | lr={lr:.6f}")

            # verify lr actually changes during warmup
            if step == 1:
                prev_lr = lr
            elif step == 3 and prev_lr is not None:
                assert lr != prev_lr, f"LR should change but got {lr} == {prev_lr}"

            if step >= total_steps:
                break
        epoch += 1

    # --- checkpoint save ---
    # All ranks must use the same checkpoint directory.
    if is_main:
        ckpt_dir = tempfile.mkdtemp(prefix="ds_test_")
    else:
        ckpt_dir = ""
    if world_size > 1:
        ckpt_dir_list = [ckpt_dir]
        torch.distributed.broadcast_object_list(ckpt_dir_list, src=0)
        ckpt_dir = ckpt_dir_list[0]

    tag = "step_000040"
    client_state = {"step": step, "epoch": epoch}
    model_engine.save_checkpoint(save_dir=ckpt_dir, tag=tag, client_state=client_state)
    if is_main:
        print(f"[Rank {world_rank}] Checkpoint saved to {ckpt_dir}/{tag}")

    # --- checkpoint load ---
    load_path, loaded_state = model_engine.load_checkpoint(
        load_dir=ckpt_dir,
        tag=tag,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    )
    if is_main:
        assert load_path is not None, "Checkpoint load failed!"
        assert loaded_state["step"] == step, f"step mismatch: {loaded_state['step']} != {step}"
        print(f"[Rank {world_rank}] Checkpoint loaded. step={loaded_state['step']}, epoch={loaded_state['epoch']}")

    # --- continue training after load ---
    # Note: scheduler already completed total_steps steps, so we can't call
    # model_engine.step() again with the same OneCycleLR. In production code
    # you'd create a new scheduler or use a larger total_steps. Here we just
    # verify forward still works after resume.
    model_engine.train()
    for batch_x, batch_y in loader:
        batch_x = batch_x.cuda().to(torch.bfloat16)
        batch_y = batch_y.cuda().to(torch.bfloat16)
        outputs = model_engine(batch_x)
        loss = nn.functional.mse_loss(outputs, batch_y.squeeze(-1))
        if is_main:
            print(f"  post-resume forward | loss={loss.item():.4f} (forward-only, no step)")
        break

    if is_main:
        print(f"[Rank {world_rank}] ALL TESTS PASSED!")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
