"""2026-08-12 四项训练管线特性的本地单元测试:

1. LolaImageAugment 样本级参数共享语义 (同 seed → 同参数, 异 seed → 异参数,
   reflection 填充保内容)
2. Trainer EMA 分片方法 (_ema_register / _ema_update / _ema_rebind) 数学正确性
3. LoLAV07Config.observation_delta_indices override (obs_prev_chunk_frame)
4. LoLAV07Policy.inject_prev_chunk_frame chunk 边界缓存行为
5. Trainer 独立 timer checkpoint 调度语义

运行: python tests/test_lola_v07_pipeline_features.py
"""

import sys
import types
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "lerobot" / "scripts"))

import torch

PASS, FAIL = "✅", "❌"
failures = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {PASS if cond else FAIL} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- 1. 图像增强
print("[1] LolaImageAugment 参数共享语义")
from lerobot.datasets.lola_dataset import LolaImageAugment

aug = LolaImageAugment(brightness=0.2, contrast=0.2, saturation=0.2,
                       translate=0.1, scale_min=0.9, scale_max=1.1)

# 同 seed → 两相机 (不同内容) 经历同一组随机参数:
# 用常数图验证 jitter 因子一致: 常数图过 brightness/contrast/saturation 后仍是常数,
# 其值完全由随机因子决定; affine 对常数图 (reflection) 不改变常数值。
img_a = torch.full((3, 32, 32), 0.5)
img_b = torch.full((3, 32, 32), 0.5)
torch.manual_seed(1234)
out_a = aug(img_a)
torch.manual_seed(1234)
out_b = aug(img_b)
check("同 seed 同参数 (两相机输出完全一致)", torch.allclose(out_a, out_b, atol=1e-6))
check("同 seed 输出非平凡 (参数确实随机, 非恒等)", not torch.allclose(out_a, img_a, atol=1e-3))

torch.manual_seed(999)
out_c = aug(img_a)
check("异 seed 异参数 (样本间独立)", not torch.allclose(out_a, out_c, atol=1e-4))

# 4D (T,C,H,W) 输入: 所有帧共享参数 — 常数帧序列输出仍各帧相等
frames = torch.full((2, 3, 32, 32), 0.5)
frames[1] = 0.5  # 同值两帧
torch.manual_seed(77)
out_f = aug(frames)
check("4D 帧间共享参数 (两帧输出一致)", torch.allclose(out_f[0], out_f[1], atol=1e-6))
check("输出值域 clamp 到 [0,1]", float(out_f.min()) >= 0.0 and float(out_f.max()) <= 1.0)

# reflection 填充: 非均匀图平移后边角无黑边 (全图均值 ≈ 原均值, 无 0 值填充带)
tex = torch.rand(3, 64, 64) * 0.8 + 0.1
aug_t = LolaImageAugment(translate=0.1)
torch.manual_seed(5)
out_t = aug_t(tex)
check("reflection 填充无黑边 (min > 0)", float(out_t.min()) > 0.0,
      f"min={float(out_t.min()):.4f}")

# pickle 安全 (DataLoader worker 传递)
import pickle
aug_rt = pickle.loads(pickle.dumps(aug))
check("可 pickle", aug_rt.brightness == aug.brightness and aug_rt.scale_max == aug.scale_max)

# 回归 (2026-08-14 事故): delta_indices 开启时 item 含 "observation.images.{cam}_is_pad"
# 1-D 布尔标记, 增强必须只作用在 camera_keys 上
import types as _types
from lerobot.datasets.lola_dataset import LoLADataset

ds = object.__new__(LoLADataset)  # 绕过重型 __init__
ds._lola_image_transforms = aug
ds.meta = _types.SimpleNamespace(
    camera_keys=["observation.images.top", "observation.images.gripper"])
item = {
    "observation.images.top": torch.full((2, 3, 16, 16), 0.5),      # (T,C,H,W)
    "observation.images.gripper": torch.full((3, 16, 16), 0.5),     # (C,H,W)
    "observation.images.top_is_pad": torch.zeros(2, dtype=torch.bool),      # 1-D 标记
    "observation.images.gripper_is_pad": torch.zeros(2, dtype=torch.bool),  # 1-D 标记
    "observation.state": torch.zeros(7),
}
item_out = ds._apply_image_transforms(item)
check("_is_pad 键原样保留 (未崩未动)",
      item_out["observation.images.top_is_pad"].dtype == torch.bool
      and item_out["observation.images.top_is_pad"].ndim == 1)
check("非相机键未被动过", torch.equal(item_out["observation.state"], torch.zeros(7)))
check("两相机同 seed 同参数 (常数图输出一致)",
      torch.allclose(item_out["observation.images.top"][0],
                     item_out["observation.images.gripper"], atol=1e-6))
check("相机图仍被增强 (非恒等)",
      not torch.allclose(item_out["observation.images.top"],
                         torch.full((2, 3, 16, 16), 0.5), atol=1e-3))
# transforms=None 时直通
ds2 = object.__new__(LoLADataset)
ds2._lola_image_transforms = None
ds2.meta = ds.meta
check("transforms=None 原样直通", ds2._apply_image_transforms(item) is item)


# ---------------------------------------------------------------- 2. EMA 分片
print("[2] Trainer EMA 分片方法 (toy model)")
import torch.nn as nn
from lerobot.scripts.train_lola_v07_azure import LoLAV07Trainer


class _ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(4, 4)
        self.vlm = nn.Linear(4, 4)


def _make_stub(decay: float):
    stub = types.SimpleNamespace()
    stub.policy = _ToyPolicy()
    stub.config = types.SimpleNamespace(ema_decay=decay)
    stub.device = torch.device("cpu")
    stub.world_rank = 0
    stub.is_distributed = False
    stub._ema_state = None
    return stub


stub = _make_stub(decay=0.9)
LoLAV07Trainer._ema_register(stub)
check("register 覆盖全部可训练参数",
      len(stub._ema_state) == sum(1 for _ in stub.policy.named_parameters()))

# 冻结 vlm → register 只收 trainable
stub2 = _make_stub(decay=0.9)
for p in stub2.policy.vlm.parameters():
    p.requires_grad_(False)
LoLAV07Trainer._ema_register(stub2)
n_train = sum(1 for _, p in stub2.policy.named_parameters() if p.requires_grad)
check("register 跳过冻结参数", len(stub2._ema_state) == n_train,
      f"{len(stub2._ema_state)} == {n_train}")

# update 数学: ema = d*ema + (1-d)*p
with torch.no_grad():
    for p in stub.policy.parameters():
        p.add_(1.0)
expected = {k: v * 0.9 + 0.1 * dict(stub.policy.named_parameters())[k].data
            for k, v in stub._ema_state.items()}
LoLAV07Trainer._ema_update(stub)
ok = all(torch.allclose(stub._ema_state[k], expected[k], atol=1e-6) for k in expected)
check("update: ema = d*ema + (1-d)*w", ok)

# rebind: 新增 trainable 参数 (VLM 解冻) → 从当前权重初始化; 已有的保留历史
for p in stub.policy.vlm.parameters():
    p.requires_grad_(False)
LoLAV07Trainer._ema_register(stub)  # 只含 model.*
ema_before = {k: v.clone() for k, v in stub._ema_state.items()}
for p in stub.policy.vlm.parameters():
    p.requires_grad_(True)  # 模拟动态解冻
LoLAV07Trainer._ema_rebind(stub)
vlm_keys = [k for k in stub._ema_state if k.startswith("vlm.")]
check("rebind: 新增 VLM 键", len(vlm_keys) > 0)
check("rebind: VLM 新键以当前权重初始化",
      all(torch.allclose(stub._ema_state[k], dict(stub.policy.named_parameters())[k].data)
          for k in vlm_keys))
check("rebind: 已有键保留历史值",
      all(torch.allclose(stub._ema_state[k], ema_before[k]) for k in ema_before))

# rebind: 冻结后键被移除
for p in stub.policy.vlm.parameters():
    p.requires_grad_(False)
LoLAV07Trainer._ema_rebind(stub)
check("rebind: 重新冻结后 VLM 键被移除",
      all(not k.startswith("vlm.") for k in stub._ema_state))


# ------------------------------------------------------- 3. delta_indices 覆盖
print("[3] observation_delta_indices override")
from lerobot.policies.lola_v07.configuration_lola_v07 import LoLAV07Config

cfg_off = LoLAV07Config(action_chunk_size=10)
cfg_off_n2 = LoLAV07Config(action_chunk_size=10, n_obs_steps=2)
cfg_on = LoLAV07Config(action_chunk_size=10, obs_prev_chunk_frame=True)
check("关闭时与父类一致 ([0])", cfg_off.observation_delta_indices == [0],
      str(cfg_off.observation_delta_indices))
check("关闭时 n_obs_steps=2 → [-1, 0] (父类语义不受影响)",
      cfg_off_n2.observation_delta_indices == [-1, 0])
check("开启时 [-chunk, 0]", cfg_on.observation_delta_indices == [-10, 0],
      str(cfg_on.observation_delta_indices))

# __post_init__ 校验
try:
    LoLAV07Config(ema_decay=1.5)
    check("非法 ema_decay 被拒", False)
except (ValueError, AssertionError):
    check("非法 ema_decay 被拒", True)
try:
    LoLAV07Config(image_aug_scale_min=1.1, image_aug_scale_max=0.9)
    check("scale_min > scale_max 被拒", False)
except (ValueError, AssertionError):
    check("scale_min > scale_max 被拒", True)


# --------------------------------------------- 4. inject_prev_chunk_frame 缓存
print("[4] inject_prev_chunk_frame chunk 边界行为")
from lerobot.policies.lola_v07.modeling_lola_v07 import LoLAV07Policy

policy = object.__new__(LoLAV07Policy)  # 绕过重型 __init__ (VLM 加载)
policy.config = types.SimpleNamespace(obs_prev_chunk_frame=True, action_chunk_size=10)
policy._action_queue = deque(maxlen=50)
policy._prev_chunk_frames = None

cam1 = torch.full((3, 8, 8), 0.1)
cam2 = torch.full((3, 8, 8), 0.2)
obs1 = {"observation.images.top": cam1, "observation.images.wrist": cam2,
        "observation.state": torch.zeros(14)}
out1 = policy.inject_prev_chunk_frame(obs1)
check("首 chunk 自复制 (prev=cur)", torch.allclose(out1["observation.images.top"][0], cam1)
      and torch.allclose(out1["observation.images.top"][1], cam1))
check("非相机键原样透传", torch.equal(out1["observation.state"], obs1["observation.state"]))
check("输出形状 (2,C,H,W)", out1["observation.images.top"].shape == (2, 3, 8, 8))
check("原 obs dict 未被修改", obs1["observation.images.top"].ndim == 3)

# 非边界步 (queue 非空) → 原样返回, 缓存不更新
policy._action_queue.append(torch.zeros(10, 14))
cam3 = torch.full((3, 8, 8), 0.3)
out2 = policy.inject_prev_chunk_frame({"observation.images.top": cam3})
check("非边界步原样返回 (ndim=3)", out2["observation.images.top"].ndim == 3)

# 下一 chunk 边界: prev = 上一 chunk 起始帧 (cam1/cam3 期间的 cam1? 应为 cam1,
# 因为非边界步不更新缓存)
policy._action_queue.clear()
cam4 = torch.full((3, 8, 8), 0.4)
out3 = policy.inject_prev_chunk_frame({"observation.images.top": cam4})
check("次 chunk prev=上一 chunk 起始帧", torch.allclose(out3["observation.images.top"][0], cam1))
check("次 chunk cur=当前帧", torch.allclose(out3["observation.images.top"][1], cam4))

# 开关关闭 → 恒等
policy.config.obs_prev_chunk_frame = False
policy._action_queue.clear()
out4 = policy.inject_prev_chunk_frame({"observation.images.top": cam4})
check("开关关闭时恒等", torch.equal(out4["observation.images.top"], cam4))

# reset 清缓存
policy.config.obs_prev_chunk_frame = True
policy._prev_chunk_frames = {"observation.images.top": cam1}
# reset() 会触碰模块其他状态, 直接模拟其语义行:
policy._prev_chunk_frames = None
out5 = policy.inject_prev_chunk_frame({"observation.images.top": cam4})
check("缓存清空后首 chunk 自复制", torch.allclose(out5["observation.images.top"][0], cam4))


# ------------------------------------------------------- 5. checkpoint timer
print("[5] Trainer checkpoint timer 调度")
timer_trainer = object.__new__(LoLAV07Trainer)
timer_trainer.save_every_n_steps = 10
timer_trainer.save_every_n_epochs = 2
timer_trainer.save_every_n_seconds = 3600.0
timer_trainer.global_step = 20
timer_trainer.is_main_process = True
timer_trainer.is_distributed = False
timer_trainer.device = torch.device("cpu")
timer_trainer._last_checkpoint_time = 100.0
timer_trainer._checkpoint_timer_signal = None

reasons = timer_trainer._checkpoint_reasons(batch_idx=3, epoch=1, now=3699.9)
check("timer 阈值前不触发", reasons == ["step"], str(reasons))
reasons = timer_trainer._checkpoint_reasons(batch_idx=3, epoch=1, now=3700.0)
check("timer 到期与 step 合并为一次保存", reasons == ["step", "timer"], str(reasons))

timer_trainer._mark_checkpoint_saved(now=5000.0)
check("任意成功保存重置 timer", timer_trainer._last_checkpoint_time == 5000.0)
timer_trainer.global_step = 21
reasons = timer_trainer._checkpoint_reasons(batch_idx=1, epoch=1, now=8599.9)
check("重置后重新累计完整 interval", reasons == [], str(reasons))
reasons = timer_trainer._checkpoint_reasons(batch_idx=0, epoch=2, now=8600.0)
check("epoch 固定点与 timer 可同时触发", reasons == ["epoch", "timer"], str(reasons))


print()
if failures:
    print(f"FAILED: {len(failures)} 项 — {failures}")
    sys.exit(1)
print("ALL PASS")
