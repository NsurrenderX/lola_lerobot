"""Isolated, bounded BERT workload for the first resumed data batch."""

import argparse
import ctypes
import json
import math
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def _arm_parent_death(expected_parent: int):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Resume GPU keepalive requires Linux parent-death protection")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != expected_parent:
        raise RuntimeError("Resume parent exited before keepalive initialization")


class ResumeGPUKeepalive:
    """Own one child, including signal cleanup and failure notification."""

    def __init__(
        self, local_rank: int, global_rank: int, batch_size: int = 8,
        max_seconds: float = 3600, startup_timeout: float = 120,
        shutdown_timeout: float = 10, log=print,
    ):
        if batch_size <= 0 or local_rank < 0:
            raise ValueError("Keepalive batch size must be positive and local rank nonnegative")
        if any(not math.isfinite(value) or value <= 0
               for value in (max_seconds, startup_timeout, shutdown_timeout)):
            raise ValueError("Keepalive time limits must be finite and positive")
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.batch_size = batch_size
        self.max_seconds = max_seconds
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.log = log
        self._process = None
        self._control = None
        self._monitor = None
        self._stopping = threading.Event()
        self._failure = None
        self._handlers = {}
        self._started = False
        self._ready = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stop()
        except Exception as error:
            if exc_type is None:
                raise
            self.log(f"Resume keepalive cleanup error: {error}")

    def _command(self, control_fd: int):
        return [
            sys.executable, str(Path(__file__).resolve()),
            "--local-rank", str(self.local_rank), "--global-rank", str(self.global_rank),
            "--batch-size", str(self.batch_size), "--max-seconds", str(self.max_seconds),
            "--parent-pid", str(os.getpid()), "--control-fd", str(control_fd),
        ]

    def _on_signal(self, signum, frame):
        if self._failure is not None:
            raise RuntimeError(self._failure)
        previous = self._handlers.get(signum)
        if callable(previous):
            previous(signum, frame)
        raise SystemExit(128 + signum)

    def _watch(self, deadline):
        try:
            exit_code = self._process.wait(timeout=max(0.001, deadline - time.monotonic()))
            failure = f"Resume BERT exited unexpectedly (code={exit_code})"
        except subprocess.TimeoutExpired:
            failure = f"Resume BERT exceeded its {self.max_seconds}s runtime limit"
        if not self._stopping.is_set():
            self._failure = failure
            os.kill(os.getpid(), signal.SIGTERM)

    def start(self):
        if self._started:
            raise RuntimeError("Resume keepalive can only start once")
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Resume keepalive must be managed from the training main thread")
        self._started = True
        started = time.monotonic()
        self._handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
        for signum in self._handlers:
            signal.signal(signum, self._on_signal)
        self._control, child_control = socket.socketpair()
        environment = os.environ.copy()
        for key in tuple(environment):
            if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                       "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"} or key.startswith("TORCHELASTIC_"):
                environment.pop(key)
        environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1",
                           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        try:
            self._process = subprocess.Popen(
                self._command(child_control.fileno()), env=environment,
                pass_fds=(child_control.fileno(),), stdin=subprocess.DEVNULL,
            )
            child_control.close()
            self.log(f"Resume BERT starting: pid={self._process.pid} cuda:{self.local_rank}")
            deadline = started + min(self.startup_timeout, self.max_seconds)
            response = bytearray()
            while b"\n" not in response:
                self._control.settimeout(max(0.001, deadline - time.monotonic()))
                chunk = self._control.recv(4096)
                if not chunk or len(response) + len(chunk) > 4096:
                    raise RuntimeError("Resume BERT failed before its first completed step")
                response.extend(chunk)
            ready = json.loads(response)
            if ready.get("event") != "ready":
                raise RuntimeError(f"Invalid resume BERT startup message: {ready}")
            self._ready = True
            self.log(f"Resume BERT ready: pid={self._process.pid} "
                     f"peak_tensor_bytes={ready.get('peak_tensor_bytes')} "
                     f"startup_s={time.monotonic() - started:.2f}")
            self._monitor = threading.Thread(
                target=self._watch, args=(started + self.max_seconds,), daemon=True,
                name="resume-bert-monitor",
            )
            self._monitor.start()
        except BaseException:
            self.stop()
            raise
        finally:
            child_control.close()

    def stop(self):
        already_stopping = self._stopping.is_set()
        self._stopping.set()
        started = time.monotonic()
        try:
            if not already_stopping and self._ready and self._process is not None:
                exit_code = self._process.poll()
                if exit_code is not None:
                    self._failure = self._failure or f"Resume BERT exited unexpectedly (code={exit_code})"
            if self._control is not None:
                try:
                    self._control.settimeout(0.1)
                    self._control.sendall(b"STOP\n")
                except OSError:
                    pass
                self._control.close()
                self._control = None
            if self._process is not None:
                process = self._process
                try:
                    process.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    self.log(f"Resume BERT stop timeout: terminating pid={process.pid}")
                    process.terminate()
                    try:
                        process.wait(timeout=self.shutdown_timeout)
                    except subprocess.TimeoutExpired:
                        self.log(f"Resume BERT termination timeout: killing pid={process.pid}")
                        process.kill()
                        process.wait(timeout=self.shutdown_timeout)
                if self._monitor is not None:
                    self._monitor.join(timeout=self.shutdown_timeout)
                self.log(f"Resume BERT reaped: pid={process.pid} code={process.returncode} "
                         f"shutdown_s={time.monotonic() - started:.2f}")
                self._process = None
        finally:
            for signum, handler in self._handlers.items():
                signal.signal(signum, handler)
            self._handlers.clear()
        if self._failure is not None:
            failure, self._failure = self._failure, None
            raise RuntimeError(failure)


def _build_bert_workload(device, batch_size):
    import torch
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        vocab_size=4096, hidden_size=256, num_hidden_layers=4,
        num_attention_heads=4, intermediate_size=1024,
        max_position_embeddings=128, num_labels=2,
    )
    model = BertForSequenceClassification(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    token_ids = torch.randint(config.vocab_size, (batch_size, 128), device=device)
    labels = torch.randint(2, (batch_size,), device=device)
    return model, optimizer, token_ids, labels


def _run_worker(args):
    _arm_parent_death(args.parent_pid)
    started = time.monotonic()
    control = socket.socket(fileno=args.control_fd)
    stopping = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopping.set())

    import torch

    torch.set_num_threads(1)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.manual_seed(1701 + args.global_rank)
    model, optimizer, token_ids, labels = _build_bert_workload(device, args.batch_size)
    steps = 0
    while not stopping.is_set():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError("Resume BERT runtime limit reached")
        if select.select([control], [], [], 0)[0]:
            control.recv(4096)
            break
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=token_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        steps += 1
        if steps == 1:
            control.sendall((json.dumps({
                "event": "ready", "peak_tensor_bytes": torch.cuda.max_memory_allocated(device),
            }) + "\n").encode())
    control.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-rank", type=int, required=True)
    parser.add_argument("--global-rank", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--control-fd", type=int, required=True)
    _run_worker(parser.parse_args())


if __name__ == "__main__":
    main()