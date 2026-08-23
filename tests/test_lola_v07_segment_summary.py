"""双 Segment Summary (history_tokenization_mode="segment_summary") 本地单元测试。

覆盖方案 modify3.md 第三部分的可本地验证项:

1. 配置字段与校验 (§6.3): mode↔version 绑定、budget 整除、模式级不变量
2. Dataset 六字段契约 (§22.1): 左 pad / mask 右对齐 / total_length 不截断
3. Field survival (§22.2): hist_ 前缀提取, 必需 vs 可选字段
4. SegmentHistoryPool (§22.3): 单次调用、absent fallback、position 方向、不等 budget
5. Chain-position sampling 与 content dropout (§22.4): 分布、互斥、可复现
6. Train/inference parity (§22.5): 5-token 布局与 mask 逐元素一致
7. Optimizer 覆盖与 resume preflight (§24 / §15.2)

ZeRO-3 save/resume (§25) 需要 world_size>=2, 不在本文件内; 见方案 §25。

运行: python tests/test_lola_v07_segment_summary.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "lerobot" / "scripts"))

import numpy as np
import torch
import torch.nn as nn

PASS, FAIL = "✅", "❌"
failures = []

DATASET_ROOT = "/data_6t_2/lerobot_v30/calvin_task_ABC_D_training_v4/"


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {PASS if cond else FAIL} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def raises(fn, needle: str = "") -> tuple:
    try:
        fn()
        return False, "没有报错"
    except Exception as e:  # noqa: BLE001 — 测试断言需要捕获任意异常类型
        return (needle in str(e) if needle else True), str(e)[:160]


torch.manual_seed(0)

from lerobot.policies.lola_v07.configuration_lola_v07 import LoLAV07Config


def summary_cfg(**kw) -> LoLAV07Config:
    base = dict(
        action_dim=7, state_dim=7, action_chunk_size=8, pred_chunk_size=16,
        dit_hidden_size=512, history_type="state", state_encoder_mode="unified",
        load_full_history=True, use_special_tokens=True, use_previous_task_end=True,
        hist_action_token_drop_rate=0.0, transition_mask_rate=0.0,
        completed_tasks_history_len=4,
        history_tokenization_mode="segment_summary", history_architecture_version=1,
        history_summary_num_heads=8, device="cpu",
    )
    base.update(kw)
    return LoLAV07Config(**base)


def chunks_cfg(**kw) -> LoLAV07Config:
    base = dict(action_dim=7, state_dim=7, action_chunk_size=8, dit_hidden_size=512,
                history_type="state", state_encoder_mode="unified", device="cpu")
    base.update(kw)
    return LoLAV07Config(**base)


# ------------------------------------------------------------- 1. 配置与校验
print("[1] 配置字段与校验 (§6.3)")

check("默认仍是 chunks / version 0",
      chunks_cfg().history_tokenization_mode == "chunks"
      and chunks_cfg().history_architecture_version == 0
      and chunks_cfg().is_segment_summary is False)
check("合法 summary 配置可构造", summary_cfg().is_segment_summary)

ok, err = raises(lambda: summary_cfg(max_task_summary_frames=30), "multiple of action_chunk_size")
check("budget 不整除 chunk_size 被拒 (否则 _pad_and_chunk 会在最新帧右侧补零)", ok, err)
ok, err = raises(lambda: summary_cfg(history_architecture_version=0), "history_architecture_version")
check("mode=summary 但 version=0 被拒", ok, err)
ok, err = raises(lambda: chunks_cfg(history_architecture_version=1), "history_architecture_version")
check("mode=chunks 但 version=1 被拒", ok, err)
ok, err = raises(lambda: summary_cfg(completed_tasks_history_len=5), "completed_tasks_history_len")
check("completed_tasks_history_len=5 被拒 (CALVIN 最多 4 条已完成)", ok, err)
ok, err = raises(lambda: summary_cfg(history_type="action"), "history_type")
check("history_type=action 被拒", ok, err)
ok, err = raises(lambda: summary_cfg(history_summary_num_heads=7), "divisible")
check("num_heads 不整除 dit_hidden_size 被拒", ok, err)
ok, err = raises(lambda: summary_cfg(history_chain_position_mode="foo"), "history_chain_position_mode")
check("非法 chain mode 被拒", ok, err)
ok, err = raises(
    lambda: summary_cfg(history_chain_position_mode="uniform_0_4", history_chain_max_position=3),
    "history_chain_max_position")
check("uniform_0_4 下 max_position 必须为 4", ok, err)

check("drop=1.0 合法 (纯 null-history 对照)",
      summary_cfg(transition_summary_drop_rate=1.0, task_summary_drop_rate=1.0) is not None)
ok, err = raises(lambda: summary_cfg(task_summary_drop_rate=1.1), "[0.0, 1.0]")
check("drop>1.0 被拒", ok, err)

_c = summary_cfg(max_transition_summary_frames=8, max_task_summary_frames=32)
check("不等 budget 的 P = max(T, K)", _c.summary_padded_frames == 32, str(_c.summary_padded_frames))
check("num_chunks = P / chunk_size", _c.summary_num_chunks == 4, str(_c.summary_num_chunks))


# ------------------------------------------------------- 2. Dataset 六字段契约
print("[2] Dataset 六字段契约 (§22.1)")
from lerobot.datasets.lola_dataset import (
    SEGMENT_SUMMARY_OPTIONAL_FIELDS,
    SEGMENT_SUMMARY_REQUIRED_FIELDS,
    LoLADataset,
)

_STATE_MEAN = torch.arange(7, dtype=torch.float32) * 0.1
_STATE_STD = torch.full((7,), 0.5)


def stub_dataset(t_budget=32, k_budget=32, hist_state=None, hist_len=None) -> LoLADataset:
    """绕过重型 __init__ (视频扫描 / parquet), 只装配 segment 分支需要的属性。"""
    ds = object.__new__(LoLADataset)
    ds.state_dim, ds.action_dim = 7, 7
    ds.history_tokenization_mode = "segment_summary"
    ds.max_transition_summary_frames = t_budget
    ds.max_task_summary_frames = k_budget
    ds._hist_state_all = hist_state
    ds._hist_len_all = hist_len
    ds._hist_action_all = None
    ds.norm_action = "zscore"
    ds._state_mean, ds._state_std = _STATE_MEAN, _STATE_STD
    ds._action_mean, ds._action_std = _STATE_MEAN.clone(), _STATE_STD.clone()
    ds.gripper_dim_indices_abs = (6,)
    ds.norm_min, ds.norm_max = -0.65, 0.65
    # ramp: 第 i 行全为 i, 便于断言"最新帧在最右"
    ds._query_hf_dataset = lambda q: {
        "observation.state": torch.tensor(
            [[float(i)] * 7 for i in q["observation.state"]], dtype=torch.float32)
    }
    return ds


EP_START = 1000
ds = stub_dataset()
item = {}
ds._build_segment_history(item, idx=EP_START + 40, ep_idx=0, ep_start=EP_START)
check("恰好产出 6 个字段", set(item) == set(SEGMENT_SUMMARY_REQUIRED_FIELDS), str(sorted(item)))
check("states shape [budget, state_dim]", item["hist_task_states"].shape == (32, 7))
check("frame_mask 为 bool", item["hist_task_frame_mask"].dtype == torch.bool)
check("total_length 为 int64", item["hist_task_total_length"].dtype == torch.int64)
check("不再产出 n_transition_chunks", "n_transition_chunks" not in item)

for task_len in (1, 7, 8, 31, 32, 33, 64):
    it = {}
    idx = EP_START + task_len - 1
    stub_dataset()._build_segment_history(it, idx=idx, ep_idx=0, ep_start=EP_START)
    mask, states = it["hist_task_frame_mask"], it["hist_task_states"]
    keep = min(task_len, 32)
    check(f"task_len={task_len}: 保留 min(len, budget)={keep} 帧", int(mask.sum()) == keep)
    check(f"task_len={task_len}: mask 右对齐 (最新帧在最右)", bool(mask[-1]))
    check(f"task_len={task_len}: total_length 不被 budget 截断",
          int(it["hist_task_total_length"]) == task_len)
    check(f"task_len={task_len}: 最后一行 == 最新帧", abs(float(states[-1, 0]) - idx) < 1e-4)
    if keep < 32:
        check(f"task_len={task_len}: pad 区在 raw space 为 0",
              torch.allclose(states[:32 - keep], torch.zeros(32 - keep, 7)))

_fake_hist = torch.arange(2 * 64 * 7, dtype=torch.float32).reshape(2, 64, 7)
_fake_len = torch.tensor([0, 64], dtype=torch.int64)
ds = stub_dataset(hist_state=_fake_hist, hist_len=_fake_len)
for ep, expect in ((0, 0), (1, 64)):
    it = {}
    ds._build_segment_history(it, idx=EP_START, ep_idx=ep, ep_start=EP_START)
    check(f"transition hist_len={expect}: 保留 {min(expect, 32)} 帧",
          int(it["hist_transition_frame_mask"].sum()) == min(expect, 32))
    check(f"transition hist_len={expect}: total_length={expect}",
          int(it["hist_transition_total_length"]) == expect)
it = {}
ds._build_segment_history(it, idx=EP_START, ep_idx=1, ep_start=EP_START)
check("transition 取 npz 的最后 budget 帧",
      torch.allclose(it["hist_transition_states"], _fake_hist[1][-32:]))

it = {}
stub_dataset(t_budget=8, k_budget=32, hist_state=_fake_hist, hist_len=_fake_len) \
    ._build_segment_history(it, idx=EP_START + 40, ep_idx=1, ep_start=EP_START)
check("不等 budget: 两段各自保持自己的长度",
      it["hist_transition_states"].shape == (8, 7) and it["hist_task_states"].shape == (32, 7))

# raw-zero padding 经 z-score 后必须等于 null_state (与模型 buffer 的唯一一致点)
ds = stub_dataset(hist_state=_fake_hist, hist_len=_fake_len)
it = {}
ds._build_segment_history(it, idx=EP_START + 4, ep_idx=0, ep_start=EP_START)
ds._normalize_item(it)
null = ds.history_null_state
check("history_null_state == (0 - mean) / (std + 1e-8)",
      torch.allclose(null, (0.0 - _STATE_MEAN) / (_STATE_STD + 1e-8), atol=1e-7))
check("自然缺失的 transition 归一化后全为 null",
      torch.allclose(it["hist_transition_states"], null.expand(32, 7), atol=1e-5))
check("task pad 区归一化后 == null", torch.allclose(it["hist_task_states"][:27],
                                                   null.expand(27, 7), atol=1e-5))
check("task 真实区 != null", not torch.allclose(it["hist_task_states"][-1], null, atol=1e-3))

legacy = {"action": torch.randn(4, 7), "hist_states_full": torch.randn(8, 7)}
before = legacy["hist_states_full"].clone()
stub_dataset()._normalize_item(legacy)
check("chunks 模式的 hist_states_full 归一化行为未变",
      torch.allclose(legacy["hist_states_full"], (before - _STATE_MEAN) / (_STATE_STD + 1e-8), atol=1e-6))
check("chunks 模式不产出 summary 字段", "hist_transition_states" not in legacy)


# ------------------------------------------------------------ 3. Field survival
print("[3] Preprocessor field survival (§22.2)")
from train_lola_v07_azure import LoLAV07Trainer

_trainer = object.__new__(LoLAV07Trainer)
batch = {
    "observation.state": torch.zeros(2, 7), "task": ["a", "b"], "index": torch.zeros(2),
    "hist_states_full": torch.zeros(2, 32, 7), "n_transition_chunks": torch.zeros(2),
    "hist_transition_states": torch.zeros(2, 32, 7),
    "hist_task_total_length": torch.zeros(2),
    "hist_transition_content_drop": torch.zeros(2, dtype=torch.bool),
    "action": torch.zeros(2, 16, 7),
}
special = _trainer._extract_special_fields(batch)
check("legacy hist_* 与 n_transition_* 被提取",
      "hist_states_full" in special and "n_transition_chunks" in special)
check("summary 六字段被提取", "hist_transition_states" in special and "hist_task_total_length" in special)
check("瞬态 content_drop 被提取 (带 hist_ 前缀才不会被 batch_to_transition 删掉)",
      "hist_transition_content_drop" in special)
check("action 被提取", "action" in special)
check("非 history 字段留在 batch 里交给 preprocessor",
      set(batch) == {"observation.state", "task", "index"}, str(sorted(batch)))
check("restore 后字段回到 batch",
      "hist_transition_states" in _trainer._restore_special_fields(batch, special))

from lerobot.policies.lola_v07.modeling_lola_v07 import (
    LolaV07StateEncoder,
    LoLAV07Policy,
    LoLAV07Pytorch,
    SegmentHistoryPool,
    assert_param_group_coverage,
    build_lola_v07_param_groups,
)

_policy = object.__new__(LoLAV07Policy)
_policy.config = summary_cfg()
_policy.training = False
full = {k: torch.zeros(2, 4) for k in SEGMENT_SUMMARY_REQUIRED_FIELDS}
full["hist_transition_states"] = torch.zeros(2, 32, 7)
full["hist_task_states"] = torch.zeros(2, 32, 7)
full["hist_transition_total_length"] = torch.zeros(2, dtype=torch.long)
full["hist_task_total_length"] = torch.zeros(2, dtype=torch.long)
full["hist_transition_frame_mask"] = torch.zeros(2, 32, dtype=torch.bool)
full["hist_task_frame_mask"] = torch.zeros(2, 32, dtype=torch.bool)
check("六字段齐全时 prepare_segment_history 正常",
      set(_policy.prepare_segment_history(full)) ==
      {"transition_states", "transition_frame_mask", "transition_total_length",
       "task_states", "task_frame_mask", "task_total_length", "transition_drop", "task_drop"})
for key in SEGMENT_SUMMARY_REQUIRED_FIELDS:
    partial = {k: v for k, v in full.items() if k != key}
    ok, err = raises(lambda p=partial: _policy.prepare_segment_history(p), key)
    check(f"缺 {key} 时 fail-fast", ok, err)
check("评测端缺两个可选 drop 字段不报错",
      all(_policy.prepare_segment_history(full)[k] is None for k in ("transition_drop", "task_drop")))
check("可选字段清单与必需字段不重叠",
      not set(SEGMENT_SUMMARY_OPTIONAL_FIELDS) & set(SEGMENT_SUMMARY_REQUIRED_FIELDS))
ok, err = raises(lambda: _policy.prepare_hist_actions(full), "prepare_segment_history")
check("summary 模式下 prepare_hist_actions 拒绝回退到单帧 state", ok, err)


# ------------------------------------------------------- 4. SegmentHistoryPool
print("[4] SegmentHistoryPool (§22.3)")


def make_encoder(cfg=None):
    cfg = cfg or summary_cfg()
    enc = LolaV07StateEncoder(cfg).float()
    enc.initialize_history_null_state(_STATE_MEAN, _STATE_STD)
    return enc, cfg


B, D = 4, 7


def encode(enc, cfg, t_len=None, k_len=None, t_drop=None, k_drop=None, states=None):
    T, K = cfg.max_transition_summary_frames, cfg.max_task_summary_frames
    ts, ks = (states if states else (torch.randn(B, T, D), torch.randn(B, K, D)))
    tm, km = torch.zeros(B, T, dtype=torch.bool), torch.zeros(B, K, dtype=torch.bool)
    tl, kl = torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)
    for i in range(B):
        n = T if t_len is None else t_len[i]
        m = K if k_len is None else k_len[i]
        if n > 0:
            tm[i, T - n:] = True
        if m > 0:
            km[i, K - m:] = True
        tl[i], kl[i] = n, m
    return enc.encode_segment_summaries(ts, tm, tl, ks, km, kl, t_drop, k_drop)


enc, cfg = make_encoder()
calls = {"enc": 0, "arm": 0, "grip": 0}
_orig_enc, _orig_arm, _orig_grip = (
    enc._encode_chunks, enc.segment_pool.arm_pool.forward, enc.segment_pool.grip_pool.forward)


def _count(key, fn):
    def wrapped(*a, **k):
        calls[key] += 1
        return fn(*a, **k)
    return wrapped


enc._encode_chunks = _count("enc", _orig_enc)
enc.segment_pool.arm_pool.forward = _count("arm", _orig_arm)
enc.segment_pool.grip_pool.forward = _count("grip", _orig_grip)
out = encode(enc, cfg)
check("state encoder 只调用一次 (两段沿 batch 合并)", calls["enc"] == 1, str(calls))
check("arm pool 只调用一次", calls["arm"] == 1, str(calls))
check("grip pool 只调用一次", calls["grip"] == 1, str(calls))
enc._encode_chunks, enc.segment_pool.arm_pool.forward, enc.segment_pool.grip_pool.forward = (
    _orig_enc, _orig_arm, _orig_grip)
for key in ("transition_arm", "transition_grip", "task_arm", "task_grip"):
    check(f"{key} shape [B, hidden]", out[key].shape == (B, 512))

out = encode(enc, cfg, t_len=[0] * B, k_len=[8] * B)
check("全 mask 的 transition: present=False", bool((~out["transition_present"]).all()))
check("全 mask 的 transition: 无 NaN (走 synthetic last null chunk)",
      torch.isfinite(out["transition_arm"]).all() and torch.isfinite(out["transition_grip"]).all())
check("task present=True", bool(out["task_present"].all()))

T = 32
ts = torch.randn(B, T, D)
tm = torch.zeros(B, T, dtype=torch.bool)
tm[:, -8:] = True
tl = torch.full((B,), 8, dtype=torch.long)
ks, km, kl = torch.randn(B, T, D), torch.ones(B, T, dtype=torch.bool), torch.full((B,), 32, dtype=torch.long)
r1 = enc.encode_segment_summaries(ts, tm, tl, ks, km, kl)
ts_dirty = ts.clone()
ts_dirty[:, :24] = 999.0
r2 = enc.encode_segment_summaries(ts_dirty, tm, tl, ks, km, kl)
check("改动被 mask 的 padding 不影响 summary",
      torch.allclose(r1["transition_arm"], r2["transition_arm"], atol=1e-4))

tm_full, tl_full = torch.ones(B, T, dtype=torch.bool), torch.full((B,), 32, dtype=torch.long)
ts_swap = ts.clone()
ts_swap[:, -16:-8], ts_swap[:, -8:] = ts[:, -8:].clone(), ts[:, -16:-8].clone()
check("交换真实 chunk 顺序会改变 summary",
      not torch.allclose(enc.encode_segment_summaries(ts, tm_full, tl_full, ks, km, kl)["transition_arm"],
                         enc.encode_segment_summaries(ts_swap, tm_full, tl_full, ks, km, kl)["transition_arm"],
                         atol=1e-4))

pool = enc.segment_pool
pos4 = pool._chunk_positions(4, torch.device("cpu"))
pos8 = pool._chunk_positions(8, torch.device("cpu"))
check("chunk position 从最新往回数 (最新槽位恒为 0)", pos4.tolist() == [3, 2, 1, 0], str(pos4.tolist()))
check("budget 变化时最新槽位仍为 position 0", int(pos8[-1]) == 0 and int(pos4[-1]) == 0)

enc_u, cfg_u = make_encoder(summary_cfg(max_transition_summary_frames=8, max_task_summary_frames=32))
out_u = encode(enc_u, cfg_u, t_len=[8] * B, k_len=[32] * B)
check("不等 budget: 短段左 pad 到 P 后仍能单次前向",
      torch.isfinite(out_u["transition_arm"]).all() and bool(out_u["transition_present"].all()))

enc_d, cfg_d = make_encoder()
ts_all, ks_all = torch.randn(B, T, D), torch.randn(B, T, D)
r_drop = enc_d.encode_segment_summaries(ts_all, tm_full, tl_full, ks_all, km, kl,
                                        torch.ones(B, dtype=torch.bool), None)
null_states = enc_d.history_null_state.view(1, 1, -1).expand(B, T, D).contiguous()
r_null = enc_d.encode_segment_summaries(null_states, tm_full, tl_full, ks_all, km, kl)
check("content drop=True 等价于输入全 null (torch.where + contiguous)",
      torch.allclose(r_drop["transition_arm"], r_null["transition_arm"], atol=1e-5))
check("transition drop 不影响 task summary",
      torch.allclose(r_drop["task_arm"],
                     enc_d.encode_segment_summaries(ts_all, tm_full, tl_full, ks_all, km, kl)["task_arm"],
                     atol=1e-5))
sel = torch.tensor([True] + [False] * (B - 1))
r_sel = enc_d.encode_segment_summaries(ts_all, tm_full, tl_full, ks_all, km, kl, sel, None)
r_base = enc_d.encode_segment_summaries(ts_all, tm_full, tl_full, ks_all, km, kl)
check("逐样本隔离: 只 drop 样本 0 时其余样本不变",
      torch.allclose(r_sel["transition_arm"][1:], r_base["transition_arm"][1:], atol=1e-5))
check("被 drop 的样本 0 确实变了",
      not torch.allclose(r_sel["transition_arm"][0], r_base["transition_arm"][0], atol=1e-4))

norms = out["transition_arm"].norm(dim=-1)
check("output norm 后 summary 尺度稳定", bool((norms > 5).all() and (norms < 60).all()), str(norms.tolist()))

enc_c = LolaV07StateEncoder(chunks_cfg())
check("chunks 模式不构造 segment_pool", enc_c.segment_pool is None)
check("chunks 模式不新增 history_null_state state-dict key",
      "history_null_state" not in enc_c.state_dict())
check("summary 模式有 history_null_state key",
      "history_null_state" in LolaV07StateEncoder(summary_cfg()).state_dict())
check("chunks 模式 forward 输出形状不变",
      enc_c(torch.randn(2, 16, 7)).shape == (2, 4, 512))
ok, err = raises(lambda: LolaV07StateEncoder(summary_cfg()).initialize_history_null_state(
    torch.zeros(5), torch.ones(5)), "dim mismatch")
check("null buffer stats 维度不匹配时报错", ok, err)


# --------------------------------------------- 5. chain-position 与 content drop
print("[5] Chain-position sampling 与 content dropout (§22.4)")
from train_lola_v07_azure import stateless_uniform

idx = np.arange(2000)
u1 = stateless_uniform(7, 0, idx, 1)
check("stateless RNG 完全确定", np.array_equal(u1, stateless_uniform(7, 0, idx, 1)))
check("值域 [0, 1)", u1.min() >= 0 and u1.max() < 1)
check("不同 stream / epoch / seed 互不相同",
      not np.array_equal(u1, stateless_uniform(7, 0, idx, 2))
      and not np.array_equal(u1, stateless_uniform(7, 1, idx, 1))
      and not np.array_equal(u1, stateless_uniform(8, 0, idx, 1)))
check("与 batch 顺序无关 (按 sample_index 索引)",
      np.allclose(stateless_uniform(7, 0, idx[::-1], 1), u1[::-1]))

NB = 4000


def aug_trainer(cfg, seed=7, step=0):
    t = object.__new__(LoLAV07Trainer)
    t.config, t.seed, t.global_step = cfg, seed, step
    t._batches_per_epoch, t._last_history_aug_stats = 100, {}
    return t


def aug_batch(n_completed=4):
    return {
        "index": torch.arange(NB),
        "completed_tasks": [[f"t{j}" for j in range(n_completed)] for _ in range(NB)],
        "completed_tasks_ann": [[f"a{j}" for j in range(n_completed)] for _ in range(NB)],
        "hist_transition_states": torch.zeros(NB, 32, 7),
        "hist_transition_frame_mask": torch.ones(NB, 32, dtype=torch.bool),
        "hist_transition_total_length": torch.full((NB,), 64, dtype=torch.long),
        "hist_task_states": torch.zeros(NB, 32, 7),
        "hist_task_frame_mask": torch.ones(NB, 32, dtype=torch.bool),
        "hist_task_total_length": torch.full((NB,), 32, dtype=torch.long),
    }


cfg_a = summary_cfg(history_chain_position_mode="uniform_0_4",
                    transition_summary_drop_rate=0.7, task_summary_drop_rate=0.7)
tr, batch = aug_trainer(cfg_a), aug_batch()
tr._apply_history_augmentation(batch)
stats = tr._last_history_aug_stats
reset = batch["hist_transition_total_length"] == 0

check("产出两个 hist_ 前缀的 drop 决策字段",
      all(k in batch for k in SEGMENT_SUMMARY_OPTIONAL_FIELDS))
check("trainer 未改写 hist_*_states (null 填充由模型负责)",
      bool((batch["hist_transition_states"] == 0).all()))
check("context reset rate ≈ 0.2 (对齐 CALVIN 首子任务占比)",
      abs(stats["context_reset_rate"] - 0.2) < 0.02, str(stats["context_reset_rate"]))
_reset_idx = torch.nonzero(reset).flatten().tolist()[:50]
check("reset 样本的 completed_tasks / _ann 一起清空",
      all(len(batch["completed_tasks"][i]) == 0 and len(batch["completed_tasks_ann"][i]) == 0
          for i in _reset_idx))
check("reset 样本 transition frame_mask 全 False",
      bool((~batch["hist_transition_frame_mask"][reset]).all()))
check("非 reset 样本 frame_mask 未被改动", bool(batch["hist_transition_frame_mask"][~reset].all()))
check("context reset 与 transition content drop 互斥",
      int((batch["hist_transition_content_drop"] & reset).sum()) == 0)
_lens = np.array([len(x) for x in batch["completed_tasks"]])
check("completed_tasks 长度分布近似均匀 0..4",
      all(abs(float((_lens == p).mean()) - 0.2) < 0.02 for p in range(5)),
      str([round(float((_lens == p).mean()), 3) for p in range(5)]))
check("completed_tasks 与 _ann 长度始终一致",
      all(len(batch["completed_tasks"][i]) == len(batch["completed_tasks_ann"][i])
          for i in range(NB)))

# 方案 §11.5 的预期内容分布
t_real = (~reset) & (~batch["hist_transition_content_drop"])
k_real = ~batch["hist_task_content_drop"]
check("transition 真实内容比例 ≈ 0.24 (0.8 × 0.3)",
      abs(stats["transition_real_rate"] - 0.24) < 0.025, str(stats["transition_real_rate"]))
check("task 真实内容比例 ≈ 0.30", abs(stats["task_real_rate"] - 0.30) < 0.025,
      str(stats["task_real_rate"]))
check("两段都无真实内容 ≈ 0.532",
      abs(float(((~t_real) & (~k_real)).float().mean()) - 0.532) < 0.025)
check("日志按 reset / non-reset 分层 (含 TT/TN/NT/NN)",
      all(f"nonreset_{k}" in stats for k in ("TT", "TN", "NT", "NN"))
      and all(f"chain_pos_{p}" in stats for p in range(5)))

tr2, b2 = aug_trainer(cfg_a), aug_batch()
tr2._apply_history_augmentation(b2)
check("同 (seed, epoch, sample_index) → 决策完全一致",
      torch.equal(batch["hist_task_content_drop"], b2["hist_task_content_drop"])
      and torch.equal(batch["hist_transition_total_length"], b2["hist_transition_total_length"]))
perm = torch.randperm(NB)
tr3, b3 = aug_trainer(cfg_a), aug_batch()
b3["index"] = torch.arange(NB)[perm]
tr3._apply_history_augmentation(b3)
check("打乱 batch 顺序后逐样本决策不变",
      torch.equal(b3["hist_task_content_drop"], batch["hist_task_content_drop"][perm]))
tr4, b4 = aug_trainer(cfg_a, step=50), aug_batch()
tr4._apply_history_augmentation(b4)
check("同 epoch 内跨 step 决策不变 (resume 可复现)",
      torch.equal(b4["hist_task_content_drop"], batch["hist_task_content_drop"]))
tr5, b5 = aug_trainer(cfg_a, step=100), aug_batch()
tr5._apply_history_augmentation(b5)
check("跨 epoch 决策变化", not torch.equal(b5["hist_task_content_drop"], batch["hist_task_content_drop"]))

tr6, b6 = aug_trainer(summary_cfg(history_chain_position_mode="none")), aug_batch()
tr6._apply_history_augmentation(b6)
check("chain=none 时不截断 completed_tasks 且无 reset",
      all(len(x) == 4 for x in b6["completed_tasks"][:100])
      and tr6._last_history_aug_stats["context_reset_rate"] < 1e-9)

tr7, b7 = aug_trainer(summary_cfg(transition_summary_drop_rate=0.0, task_summary_drop_rate=0.0)), aug_batch()
tr7._apply_history_augmentation(b7)
check("drop=0 时无任何 content drop",
      not b7["hist_transition_content_drop"].any() and not b7["hist_task_content_drop"].any())
tr8, b8 = aug_trainer(summary_cfg(transition_summary_drop_rate=1.0, task_summary_drop_rate=1.0)), aug_batch()
tr8._apply_history_augmentation(b8)
check("drop=1.0 时全部 drop (纯 null-history 对照可跑)",
      bool(b8["hist_transition_content_drop"].all()) and bool(b8["hist_task_content_drop"].all()))

tr9, b9 = aug_trainer(chunks_cfg()), aug_batch()
tr9._apply_history_augmentation(b9)
check("chunks 模式不注入 drop 字段、不改 completed_tasks",
      "hist_transition_content_drop" not in b9 and all(len(x) == 4 for x in b9["completed_tasks"][:50]))


# --------------------------------------------------- 6. 5-token 布局与 parity
print("[6] 5-token 布局与 train/inference parity (§22.5 / §12)")

VH, LAYERS, VLEN = 128, (0, 1, 2), 12
e2e_cfg = summary_cfg(dit_num_heads=8, dit_double_layers=1, dit_single_layers=1,
                      action_bottleneck_dim=256, grip_bottleneck_dim=128,
                      state_bottleneck_dim=256, state_grip_bottleneck_dim=128,
                      vlm_hidden_size=VH, vlm_extract_layers=LAYERS, vlm_bridge_mode="legacy")
model = LoLAV07Pytorch(e2e_cfg)
model.state_encoder.initialize_history_null_state(_STATE_MEAN, _STATE_STD)
model.eval()

EB = 3
hs = {i: torch.randn(EB, VLEN, VH) for i in LAYERS}
ids = torch.zeros(EB, VLEN, dtype=torch.long)
tgt = torch.randn(EB, 16, 7)
cap = {}
_orig_dit = model.dit.forward


def _spy(*a, **k):
    cap["hist"] = k.get("hist_actions", a[1] if len(a) > 1 else None)
    cap["mask"] = (k.get("joint_attention_kwargs") or {}).get("attention_mask")
    cap["vlm"] = k.get("vlm_features", a[2] if len(a) > 2 else None)
    return _orig_dit(*a, **k)


model.dit.forward = _spy


def e2e_history(t_lens=(32, 0, 16)):
    ts, ks = torch.randn(EB, 32, 7), torch.randn(EB, 32, 7)
    tm, km = torch.zeros(EB, 32, dtype=torch.bool), torch.ones(EB, 32, dtype=torch.bool)
    tl, kl = torch.zeros(EB, dtype=torch.long), torch.full((EB,), 32, dtype=torch.long)
    for i, n in enumerate(t_lens):
        if n > 0:
            tm[i, 32 - n:] = True
        tl[i] = n
    return dict(transition_states=ts, transition_frame_mask=tm, transition_total_length=tl,
                task_states=ks, task_frame_mask=km, task_total_length=kl,
                transition_drop=None, task_drop=None)


sh = e2e_history()
out = model(hidden_states_all_layers=hs, input_ids=ids, hist_actions=None,
            target_actions=tgt, segment_history=sh)
check("summary forward 产生有限 loss", bool(torch.isfinite(out["total_loss"])))
check("返回 summary_diagnostics 供日志使用", "summary_diagnostics" in out)
check("history stream 恒为 10 token (arm 5 + grip 5)", cap["hist"].shape[1] == 10,
      str(cap["hist"].shape))

vl = cap["vlm"].shape[1]
grip_seg, arm_seg = cap["mask"][:, vl:vl + 5], cap["mask"][:, vl + 5:vl + 10]
present = torch.tensor([n > 0 for n in (32, 0, 16)])
check("arm 流 mask 为 [True, True, PTE, True, True]",
      bool(arm_seg[:, 0].all() and arm_seg[:, 1].all() and arm_seg[:, 3].all() and arm_seg[:, 4].all()))
check("PTE 可见性 == transition_present", torch.equal(arm_seg[:, 2], present),
      f"{arm_seg[:, 2].tolist()} vs {present.tolist()}")
check("grip 流 PTE 与 arm 一致", torch.equal(grip_seg[:, 2], present))

model.train()
model(hidden_states_all_layers=hs, input_ids=ids, hist_actions=None,
      target_actions=tgt, segment_history=e2e_history())["total_loss"].backward()
_bad = [n for n, p in model.state_encoder.segment_pool.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
check("pool 全部参数拿到非零有限梯度", not _bad, str(_bad))
model.zero_grad(set_to_none=True)
model.eval()

sh_parity = e2e_history()
cap.clear()
with torch.no_grad():
    model(hidden_states_all_layers=hs, input_ids=ids, hist_actions=None,
          target_actions=tgt, segment_history={**sh_parity})
f_hist, f_mask, f_vl = cap["hist"].clone(), cap["mask"].clone(), cap["vlm"].shape[1]
cap.clear()
with torch.no_grad():
    model.sample_actions(hidden_states_all_layers=hs, hist_actions=None,
                         segment_history={**sh_parity})
s_hist, s_mask, s_vl = cap["hist"].clone(), cap["mask"].clone(), cap["vlm"].shape[1]
check("parity: 训练与推理的 history stream 逐元素一致",
      torch.allclose(f_hist, s_hist, atol=1e-5), str((f_hist - s_hist).abs().max().item()))
check("parity: 训练与推理的 history mask 一致",
      torch.equal(f_mask[:, f_vl:f_vl + 10], s_mask[:, s_vl:s_vl + 10]))

sh_base = e2e_history()
cap.clear()
with torch.no_grad():
    model(hidden_states_all_layers=hs, input_ids=ids, hist_actions=None, target_actions=tgt,
          segment_history={**sh_base, "transition_drop": torch.ones(EB, dtype=torch.bool)})
h_drop, m_drop = cap["hist"].clone(), cap["mask"].clone()
cap.clear()
with torch.no_grad():
    model(hidden_states_all_layers=hs, input_ids=ids, hist_actions=None, target_actions=tgt,
          segment_history={**sh_base})
h_base = cap["hist"].clone()
check("content drop 改变 transition summary token (index 1)",
      not torch.allclose(h_drop[:, 1], h_base[:, 1], atol=1e-4))
check("content drop 不影响 task summary token (index 3)",
      torch.allclose(h_drop[:, 3], h_base[:, 3], atol=1e-5))
check("content drop 不改变 attention mask (几何与长度信息保留)",
      torch.equal(m_drop[:, vl:vl + 10], cap["mask"][:, vl:vl + 10]))

model_c = LoLAV07Pytorch(chunks_cfg(
    pred_chunk_size=16, dit_num_heads=8, dit_double_layers=1, dit_single_layers=1,
    action_bottleneck_dim=256, grip_bottleneck_dim=128,
    state_bottleneck_dim=256, state_grip_bottleneck_dim=128,
    vlm_hidden_size=VH, vlm_extract_layers=LAYERS, vlm_bridge_mode="legacy",
    use_special_tokens=True, use_previous_task_end=True))
model_c.eval()
hist, hmask = torch.randn(EB, 32, 7), torch.ones(EB, 32)
oc = model_c(hidden_states_all_layers=hs, input_ids=ids, hist_actions=hist, target_actions=tgt,
             hist_actions_mask=hmask, n_transition_chunks=1,
             n_transition_chunks_batch=torch.tensor([1, 0, 1]))
check("chunks 模式 forward 未回归", bool(torch.isfinite(oc["total_loss"])))
check("chunks 模式不返回 summary_diagnostics", "summary_diagnostics" not in oc)
with torch.no_grad():
    ac = model_c.sample_actions(hidden_states_all_layers=hs, hist_actions=hist,
                                hist_actions_mask=hmask, n_transition_chunks=1)
check("chunks 模式 sample_actions 未回归", ac.shape[0] == EB and bool(torch.isfinite(ac).all()))


# -------------------------------------------- 7. Optimizer 覆盖与 resume 校验
print("[7] Optimizer 覆盖与 resume preflight (§24 / §15.2)")


class _FakePolicy(nn.Module):
    """只提供 build_lola_v07_param_groups 需要的模块结构。"""

    def __init__(self, cfg, with_vlm=True):
        super().__init__()
        self.model = nn.Module()
        for name in ("dit", "vlm_bridge", "action_encoder", "arm_dit_to_latent", "grip_dit_to_latent"):
            setattr(self.model, name, nn.Linear(4, 4))
        self.model.state_encoder = LolaV07StateEncoder(cfg).float()
        if with_vlm:
            self.vlm = nn.Linear(4, 4)


cfg_o = summary_cfg(encoder_lr_mult=1.0)
pol = _FakePolicy(cfg_o)
for p in pol.vlm.parameters():
    p.requires_grad = False
groups = build_lola_v07_param_groups(pol, cfg_o, base_lr=1e-4, include_vlm=False)
in_opt = {id(x) for g in groups for x in g["params"]}
pool_params = list(pol.model.state_encoder.segment_pool.parameters())
check("pool 参数全部进入 optimizer (state_encoder 子模块, 6 处枚举零修改)",
      len(pool_params) > 0 and all(id(x) in in_opt for x in pool_params))
_names = {n for n, _ in pol.model.state_encoder.segment_pool.named_parameters()}
check("query / type / gate / 投影 / length MLP / norm 均在组内",
      {"summary_queries", "segment_type_embeddings", "last_chunk_gates",
       "arm_pool.k_proj.weight", "grip_pool.out_proj.weight",
       "length_mlp.0.weight", "output_norm.weight"} <= _names)
check("覆盖集合 == trainable 集合",
      in_opt == {id(x) for x in pol.parameters() if x.requires_grad})

_missing = [g for g in groups
            if not any(id(x) in {id(y) for y in pool_params} for x in g["params"])]
ok, err = raises(lambda: assert_param_group_coverage(pol, _missing), "NOT in optimizer")
check("漏登记 state_encoder 组被断言抓住 (否则静默冻结)", ok, err)
ok, err = raises(lambda: assert_param_group_coverage(
    pol, groups + [{"params": list(pol.model.dit.parameters()), "lr": 1.0}]), "multiple groups")
check("参数重复进组被断言抓住", ok, err)
ok, err = raises(lambda: assert_param_group_coverage(
    pol, groups + [{"params": list(pol.vlm.parameters()), "lr": 1.0}]))
check("冻结参数混入被断言抓住", ok, err)

pol2 = _FakePolicy(cfg_o)
g2 = build_lola_v07_param_groups(pol2, cfg_o, base_lr=1e-4, vlm_lr=1e-6, include_vlm=True)
check("include_vlm 增加一组且覆盖完整",
      len(g2) == len(groups) + 1
      and {id(x) for g in g2 for x in g["params"]} ==
      {id(x) for x in pol2.parameters() if x.requires_grad})
ok, _ = raises(lambda: build_lola_v07_param_groups(pol2, cfg_o, 1e-4, include_vlm=True))
check("include_vlm 缺 vlm_lr 报错", ok)

cfg_m = summary_cfg(encoder_lr_mult=1.5)
pol3 = _FakePolicy(cfg_m)
for p in pol3.vlm.parameters():
    p.requires_grad = False
g3 = build_lola_v07_param_groups(pol3, cfg_m, base_lr=1e-4, include_vlm=False)
_se_ids = {id(x) for x in pol3.model.state_encoder.parameters()}
_se_group = next(g for g in g3 if any(id(x) in _se_ids for x in g["params"]))
check("state_encoder 组 (含 pool) 跟随 encoder_lr_mult",
      abs(_se_group["lr"] - 1.5e-4) < 1e-12, str(_se_group["lr"]))

from train_lola_v07_azure import ARCH_FINGERPRINT_KEYS


def preflight_trainer(cfg):
    t = object.__new__(LoLAV07Trainer)
    t.config, t._ema_state, t.history_state_stats = cfg, None, None
    return t


_root = tempfile.mkdtemp(prefix="lola_ckpt_test_")


def write_ckpt_cfg(name, cfg_dict):
    d = os.path.join(_root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "training_config.json"), "w") as f:
        json.dump({"lola_config": cfg_dict}, f)
    return d


cur = summary_cfg()
same = {k: getattr(cur, k) for k in ARCH_FINGERPRINT_KEYS}
d_same = write_ckpt_cfg("same", same)
ok, err = raises(lambda: preflight_trainer(cur)._assert_checkpoint_arch_compatible(d_same))
check("相同 fingerprint 通过", not ok, err)
_tag = os.path.join(d_same, "step_000100")
os.makedirs(_tag, exist_ok=True)
ok, err = raises(lambda: preflight_trainer(cur)._assert_checkpoint_arch_compatible(_tag))
check("tag 子目录能回溯父目录的 training_config.json", not ok, err)

_chunks_json = dict(same, history_tokenization_mode="chunks", history_architecture_version=0)
ok, err = raises(lambda: preflight_trainer(cur)._assert_checkpoint_arch_compatible(
    write_ckpt_cfg("chunks", _chunks_json)), "history_tokenization_mode")
check("chunks checkpoint 被拒绝加载到 summary 模式", ok, err)
ok, err = raises(lambda: preflight_trainer(cur)._assert_checkpoint_arch_compatible(
    write_ckpt_cfg("budget", dict(same, max_task_summary_frames=64))), "max_task_summary_frames")
check("budget 不同被拒绝", ok, err)

_legacy = {k: v for k, v in same.items() if not k.startswith(("history_", "max_"))}
ok, err = raises(lambda: preflight_trainer(chunks_cfg())._assert_checkpoint_arch_compatible(
    write_ckpt_cfg("legacy", _legacy)))
check("旧 chunks json (缺全部新字段) 仍能加载到 chunks (legacy defaults 生效)", not ok, err)

_nocfg = os.path.join(_root, "nocfg")
os.makedirs(_nocfg, exist_ok=True)
ok, err = raises(lambda: preflight_trainer(cur)._assert_checkpoint_arch_compatible(_nocfg),
                 "training_config.json")
check("summary 拒绝加载无 training_config.json 的 checkpoint", ok, err)
ok, err = raises(lambda: preflight_trainer(chunks_cfg())._assert_checkpoint_arch_compatible(_nocfg))
check("chunks 模式无 config 时只警告", not ok, err)


class _EmaPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(2, 2), nn.Linear(2, 2)


t_ema = preflight_trainer(cur)
t_ema.policy = _EmaPolicy()
t_ema._ema_state = {n: p.detach().clone() for n, p in t_ema.policy.named_parameters()}
ok, err = raises(t_ema._assert_ema_state_complete)
check("EMA state 完整时通过", not ok, err)
t_ema._ema_state.pop("b.weight")
ok, err = raises(t_ema._assert_ema_state_complete, "b.weight")
check("EMA 缺 key 时报错 (否则 _ema_update 会静默跳过)", ok, err)


class _BufPolicy(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.model = nn.Module()
        self.model.state_encoder = encoder


_enc_b, _ = make_encoder()
t_buf = preflight_trainer(cur)
t_buf.policy = _BufPolicy(_enc_b)
t_buf.history_state_stats = (_STATE_MEAN, _STATE_STD)
ok, err = raises(t_buf._assert_null_buffer_consistent)
check("null buffer 与 dataset stats 一致时通过", not ok, err)
t_buf.history_state_stats = (_STATE_MEAN + 1.0, _STATE_STD)
ok, err = raises(t_buf._assert_null_buffer_consistent, "history_null_state")
check("null buffer 与 stats 不一致时报错 (禁止静默覆盖)", ok, err)


# ------------------------------------------------- 8. 真实数据集一致性 (可选)
print("[8] 真实数据集一致性 (缺数据集时跳过)")
_stats_path = Path(DATASET_ROOT) / "meta" / "stats.json"
if not _stats_path.is_file():
    print(f"  ⏭️  跳过: 未找到 {_stats_path}")
else:
    _stats = json.loads(_stats_path.read_text())
    for _key, _label in (("observation.state", "original"),
                         ("observation.state_incremental", "incremental")):
        if _key not in _stats:
            continue
        _m = torch.tensor(_stats[_key]["mean"], dtype=torch.float32)
        _s = torch.tensor(_stats[_key]["std"], dtype=torch.float32)
        _ds = stub_dataset()
        _ds._state_mean, _ds._state_std = _m, _s
        _enc = LolaV07StateEncoder(summary_cfg()).float()
        _enc.initialize_history_null_state(_m, _s)
        check(f"[{_label}] dataset padding 值 == 模型 null buffer",
              torch.allclose(_ds.history_null_state, _enc.history_null_state, atol=1e-7))
        _it = {"hist_task_states": torch.zeros(3, 7), "hist_transition_states": torch.zeros(3, 7)}
        _ds._normalize_item(_it)
        check(f"[{_label}] raw-zero padding 归一化后 == buffer",
              torch.allclose(_it["hist_task_states"][0], _enc.history_null_state, atol=1e-6))

    # state 第 7 维必须是连续 gripper width, 不是 action 侧的 ±1 command (方案 §3)
    _sv, _av = _stats["observation.state"], _stats["action"]
    check("state dim6 是连续 gripper width (|值| < 0.2)",
          abs(_sv["min"][6]) < 0.2 and abs(_sv["max"][6]) < 0.2,
          f'min={_sv["min"][6]} max={_sv["max"][6]}')
    check("action dim6 是二值 gripper command (±1)",
          abs(abs(_av["min"][6]) - 1.0) < 1e-6 and abs(abs(_av["max"][6]) - 1.0) < 1e-6)
    _sigma = (1.0 - _sv["mean"][6]) / _sv["std"][6]
    check("把 ±1 当作 state dim6 会造成 >20σ 的 OOD (evaluator 必须用 robot_obs[:7])",
          _sigma > 20, f"{_sigma:.2f}σ")


import shutil

shutil.rmtree(_root, ignore_errors=True)

print()
if failures:
    print(f"FAILED: {len(failures)} 项 — {failures}")
    sys.exit(1)
print("ALL PASS")
