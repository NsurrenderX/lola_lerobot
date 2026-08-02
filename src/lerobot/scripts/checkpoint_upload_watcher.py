"""Checkpoint 异步上传 watchdog (每节点一个进程)。

背景: 训练只把 checkpoint 写到节点本地 NVMe (blobfuse 挂载点在网络波动下大文件
顺序写会 IO 错误, 曾导致 ZeRO-3 per-rank 分片保存崩溃)。本进程常驻后台:

  1. 扫描 local_root 下带 .upload_ready 标记的 tag 目录
     (训练进程 local_rank==0 在 DeepSpeed save_checkpoint 返回后落标记;
      DS save_checkpoint 末尾有 dist.barrier(), 返回即保证本节点分片已完整)
  2. azcopy 上传到 blob 对应 run/tag 目录 (每节点只上传本节点 ranks 写的分片,
     多节点写同一 blob 目录天然并集无冲突), 成功后落 .uploaded 标记
  3. 上传 run 级小文件 (latest / training_config.json / zero_to_fp32.py, 强覆盖)
  4. 本地磁盘管理: 每个 run 只保留最近 keep_last 个已上传 tag, 超龄的已上传 tag
     本地删除; 未上传成功的一律不删 (宁可报警不丢 checkpoint)
  5. drain 语义: 训练结束后 launcher 创建 drain 标记文件, 本进程排空队列后退出,
     保证 final checkpoint 落 blob 后 AMLT job 才结束

跳过 _unfreeze_tmp_* (动态解冻引擎重建的临时 roundtrip, 不需持久化)。

用法 (由 test_azure_v07c.sh 拉起, 一般不需手动执行):
  python checkpoint_upload_watcher.py \
      --local_root /scratch/lola_mirror/checkpoints/lola-v07c \
      --blob_base https://X.blob.core.windows.net/Y/checkpoints/lola-v07c
"""

import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_azure_azcopy import install_azcopy, run_azcopy_transfer

READY_MARKER = ".upload_ready"
DONE_MARKER = ".uploaded"
# 上传 tag 目录时排除标记文件 (避免把本地状态泄漏到 blob)
EXCLUDE_MARKERS = f"--exclude-pattern={READY_MARKER};{DONE_MARKER}"
# run 级需要同步的小文件 (latest 指针随保存变化, 必须强覆盖)
RUN_LEVEL_FILES = ("latest", "training_config.json", "zero_to_fp32.py")


def log(msg):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [watcher] {msg}", flush=True)


def tag_sort_key(tag_name):
    """step_000003 < step_000100 < final < 其他"""
    m = re.match(r"step_(\d+)$", tag_name)
    if m:
        return (0, int(m.group(1)))
    if tag_name == "final":
        return (1, 0)
    return (2, 0)


def iter_run_dirs(local_root):
    if not os.path.isdir(local_root):
        return
    for run in sorted(os.listdir(local_root)):
        # 跳过隐藏目录与 _unfreeze_tmp_*/_xxx 临时目录
        if run.startswith((".", "_")):
            continue
        run_dir = os.path.join(local_root, run)
        if os.path.isdir(run_dir):
            yield run, run_dir


def iter_tag_dirs(run_dir, require_ready=False, require_uploaded=False):
    for tag in sorted(os.listdir(run_dir), key=tag_sort_key):
        if tag.startswith((".", "_")):
            continue
        tag_dir = os.path.join(run_dir, tag)
        if not os.path.isdir(tag_dir):
            continue
        has_ready = os.path.exists(os.path.join(tag_dir, READY_MARKER))
        has_done = os.path.exists(os.path.join(tag_dir, DONE_MARKER))
        if require_ready and not has_ready:
            continue
        if require_uploaded and not has_done:
            continue
        if require_ready and has_done:
            continue  # 已上传的不算 pending
        yield tag, tag_dir


def scan_pending(local_root):
    """待上传: 有 .upload_ready 且无 .uploaded 的 tag 目录, 按 run+step 排序"""
    pending = []
    for run, run_dir in iter_run_dirs(local_root):
        for tag, tag_dir in iter_tag_dirs(run_dir, require_ready=True):
            pending.append((run, run_dir, tag, tag_dir))
    return pending


def upload_run_level_files(azcopy_bin, run_dir, blob_run_url):
    """同步 run 级小文件 (latest 指针等), 强覆盖; 失败只警告不阻塞 tag 标记"""
    for fname in RUN_LEVEL_FILES:
        fpath = os.path.join(run_dir, fname)
        if os.path.isfile(fpath):
            ok = run_azcopy_transfer(azcopy_bin, fpath, f"{blob_run_url}/{fname}",
                                     max_retries=3, overwrite="true")
            if not ok:
                log(f"⚠️ run 级文件 {fname} 上传失败, 下轮重试")


def cleanup_uploaded(local_root, keep_last):
    """每个 run 只保留最近 keep_last 个已上传 tag 的本地副本"""
    for run, run_dir in iter_run_dirs(local_root):
        uploaded = [td for _, td in iter_tag_dirs(run_dir, require_uploaded=True)]
        excess = len(uploaded) - keep_last
        for tag_dir in uploaded[:max(0, excess)]:
            log(f"🧹 本地清理已上传 checkpoint: {tag_dir}")
            shutil.rmtree(tag_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Checkpoint 异步上传 watchdog")
    parser.add_argument("--local_root", type=str, required=True, help="本地 ckpt 根目录 (对应 blob_base)")
    parser.add_argument("--blob_base", type=str, required=True, help="blob 目标根 URL (含 container 前缀路径)")
    parser.add_argument("--keep_last", type=int, default=2, help="每个 run 本地保留的已上传 tag 数 (默认 2)")
    parser.add_argument("--poll_interval", type=int, default=30, help="扫描间隔秒数 (默认 30)")
    parser.add_argument("--max_retries", type=int, default=5, help="单个 tag 每轮上传重试次数 (默认 5)")
    parser.add_argument("--drain_flag", type=str, default=None,
                        help="drain 标记文件路径 (默认 <local_root>/_upload_drain); 存在且队列排空后退出")
    parser.add_argument("--drain_timeout", type=int, default=7200, help="drain 最长等待秒数 (默认 7200)")
    parser.add_argument("--azcopy-path", type=str, default="./azcopy", help="azcopy 二进制存放路径")
    args = parser.parse_args()

    drain_flag = args.drain_flag or os.path.join(args.local_root, "_upload_drain")
    blob_base = args.blob_base.rstrip("/")

    azcopy_bin = install_azcopy(args.azcopy_path)
    subprocess.run([azcopy_bin, "login", "--identity"], check=True)
    log(f"watchdog 启动: {args.local_root} -> {blob_base} "
        f"(keep_last={args.keep_last}, poll={args.poll_interval}s, drain_flag={drain_flag})")

    drain_deadline = None
    while True:
        try:
            pending = scan_pending(args.local_root)
            for run, run_dir, tag, tag_dir in pending:
                blob_tag_url = f"{blob_base}/{run}/{tag}"
                log(f"📤 上传 {run}/{tag} -> {blob_tag_url}")
                ok = run_azcopy_transfer(azcopy_bin, tag_dir, blob_tag_url,
                                         max_retries=args.max_retries,
                                         overwrite="ifSourceNewer",
                                         extra_copy_args=[EXCLUDE_MARKERS])
                if ok:
                    open(os.path.join(tag_dir, DONE_MARKER), "a").close()
                    log(f"✅ {run}/{tag} 上传完成")
                    upload_run_level_files(azcopy_bin, run_dir, f"{blob_base}/{run}")
                else:
                    log(f"⚠️ {run}/{tag} 本轮上传失败, 下轮继续重试 (不阻塞训练)")
            cleanup_uploaded(args.local_root, args.keep_last)
        except Exception as e:
            # watchdog 自身任何异常都不应致命: 记录后下轮继续
            log(f"⚠️ watchdog 循环异常 (下轮继续): {type(e).__name__}: {e}")

        if os.path.exists(drain_flag):
            if not scan_pending(args.local_root):
                log("✅ drain 完成: 所有 checkpoint 已上传, watchdog 退出")
                return 0
            if drain_deadline is None:
                drain_deadline = time.time() + args.drain_timeout
                log(f"drain 标记已出现, 等待队列排空 (超时 {args.drain_timeout}s)...")
            elif time.time() > drain_deadline:
                remaining = [f"{r}/{t}" for r, _, t, _ in scan_pending(args.local_root)]
                log(f"🚨 drain 超时! 未上传: {remaining}")
                return 2

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
