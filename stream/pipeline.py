"""Streaming INT8 -> FP16 routing pipeline.

    capture -> [q_cap] -> preprocess -> [q_pre] -> INT8 (ORT CPU) -> [q_route] -> router
            -> output | [q_esc] -> FP16 escalation worker (PyTorch MPS) -> output -> [q_out] -> sink

One thread per stage, connected by bounded queues. ORT and PyTorch release the GIL inside
inference, so INT8 and FP16 run concurrently and an escalation never blocks the INT8 path.

Library code: ONNX Runtime / PyTorch inference calls, `queue`, `threading`.
Mine (this file): stage wiring and backpressure/drop policy, routing rule, keep-warm scheduler,
micro-batching collector, deadline-aware admission (service-time EWMA), per-frame timestamps.

Tail-latency fixes (each independently switchable via `Config`):
  keep_warm     - when the GPU worker has been idle for `warm_idle_ms`, run a full batch-1 dummy
                  inference. (Measured: tiny GPU ops do NOT prevent the ~23 ms post-idle slowdown;
                  only full-model work does.)
  microbatch    - after the first escalation arrives, wait up to `mb_wait_ms` or `mb_max` frames
                  and run them as one GPU batch.
  deadline      - at routing time, estimate when an escalation would finish (queue depth x EWMA
                  of GPU batch service time); if that misses the frame's deadline, keep INT8.
"""
import math
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ptq.common import logits_from_embeddings, text_embeddings

STOP = object()


@dataclass
class Config:
    mode: str = "route"            # "int8" | "fp16" | "route"
    threshold: float = 0.16        # escalate if INT8 top1-top2 softmax margin < threshold
    keep_warm: bool = False
    warm_idle_ms: float = 15.0
    microbatch: bool = False
    mb_wait_ms: float = 4.0
    mb_max: int = 4
    deadline_aware: bool = False
    deadline_ms: float = 33.3
    q_cap: int = 8                 # capture queue: full -> frame dropped at capture
    q_esc: int = 16                # escalation queue: full -> keep INT8 answer
    ort_threads: int = -1          # -1 = performance-core count (see perf_cores); 0 = ORT default


@dataclass
class Frame:
    fid: int
    t_cap: float                   # capture (scheduled arrival) time, perf_counter seconds
    raw: object
    label: int = -1
    x: np.ndarray = None
    int8_pred: int = -1
    margin: float = float("nan")
    escalated: bool = False
    deadline_kept: bool = False
    pred: int = -1
    path: str = ""
    t_out: float = 0.0
    extra: dict = field(default_factory=dict)


def perf_cores() -> int:
    """Apple Silicon P-core count. ORT's default (all cores) puts intra-op work on efficiency cores,
    which straggle: measured on M1 Pro (6P+2E) at 30 fps, default 8 threads -> INT8 p50 13.2 ms with
    stalls up to 58 ms; 6 threads -> p50 10.4 ms, max 19 ms."""
    import subprocess
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True,
                                  text=True, check=True).stdout)
    except Exception:
        return 0


def _softmax_margin(logits):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    top2 = np.sort(p, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


class Pipeline:
    def __init__(self, cfg: Config, preprocess, on_output=None):
        import onnxruntime as ort
        import torch
        from ptq.build import int8_path
        from ptq.common import ImageEncoder, load_clip

        self.cfg, self.preprocess_fn, self.on_output = cfg, preprocess, on_output
        self.torch = torch
        self.te = text_embeddings()
        so = ort.SessionOptions()
        threads = perf_cores() if cfg.ort_threads == -1 else cfg.ort_threads
        if threads:
            so.intra_op_num_threads = threads
        self.ort_threads = threads
        self.int8 = ort.InferenceSession(str(int8_path("kcenter", 512)), so, providers=["CPUExecutionProvider"])
        self.fp16 = ImageEncoder(load_clip()).to("mps", torch.float16).eval()
        self._dummy = torch.zeros(1, 3, 224, 224, device="mps", dtype=torch.float16)

        self.q_cap = queue.Queue(cfg.q_cap)
        self.q_pre = queue.Queue(8)
        self.q_route = queue.Queue(8)
        self.q_esc = queue.Queue(cfg.q_esc)
        self.q_out = queue.Queue()
        self.records, self.dropped = [], []
        self.gpu_inflight = 0
        self.svc_ewma_ms = 10.0       # GPU batch service time estimate (updated online)
        self.n_warm_runs = 0
        self.threads = []

    # ---------------- model calls ----------------
    def _run_int8(self, x):
        return logits_from_embeddings(self.int8.run(None, {"pixel_values": x})[0], self.te)

    def _run_fp16(self, x):
        with self.torch.no_grad():
            e = self.fp16(self.torch.from_numpy(x).to("mps", self.torch.float16)).float().cpu().numpy()
        return logits_from_embeddings(e, self.te)

    def warmup(self, x, n=30):
        for _ in range(n):
            self._run_int8(x)
            self._run_fp16(x)
        # MPS compiles kernels per input shape; without this, the first batch of each new size
        # mid-stream costs ~100+ ms.
        if self.cfg.microbatch:
            for b in range(2, self.cfg.mb_max + 1):
                for _ in range(3):
                    self._run_fp16(np.repeat(x, b, axis=0))

    # ---------------- stages ----------------
    def submit(self, f: Frame) -> bool:
        """Capture stage entry. Non-blocking: a full capture queue drops the frame."""
        try:
            self.q_cap.put_nowait(f)
            return True
        except queue.Full:
            self.dropped.append(f)
            return False

    def _preprocess(self):
        while (f := self.q_cap.get()) is not STOP:
            f.extra["t_pre0"] = time.perf_counter()
            f.x = self.preprocess_fn(f.raw)
            f.extra["t_pre1"] = time.perf_counter()
            f.raw = None
            self.q_pre.put(f)
        self.q_pre.put(STOP)

    def _int8_stage(self):
        while (f := self.q_pre.get()) is not STOP:
            if self.cfg.mode == "fp16":
                f.escalated = True
                self.q_esc.put(f)          # FP16-only baseline: blocking put = backpressure
                continue
            f.extra["t_i0"] = time.perf_counter()
            lg = self._run_int8(f.x)
            f.extra["t_i1"] = time.perf_counter()
            f.int8_pred, f.margin = int(lg.argmax()), float(_softmax_margin(lg)[0])
            self.q_route.put(f)
        self.q_route.put(STOP)

    def _router(self):
        cfg = self.cfg
        while (f := self.q_route.get()) is not STOP:
            f.extra["t_r"] = time.perf_counter()
            if cfg.mode == "int8" or f.margin >= cfg.threshold:
                self._emit(f, f.int8_pred, "int8")
                continue
            if cfg.deadline_aware:
                pending = self.q_esc.qsize() + self.gpu_inflight
                batch = cfg.mb_max if cfg.microbatch else 1
                wait = cfg.mb_wait_ms if cfg.microbatch else 0.0
                est_done = time.perf_counter() + (math.ceil((pending + 1) / batch) * self.svc_ewma_ms + wait) / 1e3
                if (est_done - f.t_cap) * 1e3 > cfg.deadline_ms:
                    f.deadline_kept = True
                    self._emit(f, f.int8_pred, "int8_deadline")
                    continue
            try:
                f.escalated = True
                self.q_esc.put_nowait(f)
            except queue.Full:
                f.escalated, f.deadline_kept = False, True
                self._emit(f, f.int8_pred, "int8_qfull")
        self.q_esc.put(STOP)

    def _gpu_worker(self):
        cfg = self.cfg
        timeout = cfg.warm_idle_ms / 1e3 if cfg.keep_warm else None
        done = False
        while not done:
            try:
                first = self.q_esc.get(timeout=timeout)
            except queue.Empty:            # idle for warm_idle_ms -> keep the GPU busy
                with self.torch.no_grad():
                    self.fp16(self._dummy).float().cpu()
                self.n_warm_runs += 1
                continue
            if first is STOP:
                break
            batch = [first]
            if cfg.microbatch:
                t_end = time.perf_counter() + cfg.mb_wait_ms / 1e3
                while len(batch) < cfg.mb_max:
                    rem = t_end - time.perf_counter()
                    if rem <= 0:
                        break
                    try:
                        nxt = self.q_esc.get(timeout=rem)
                    except queue.Empty:
                        break
                    if nxt is STOP:
                        done = True
                        break
                    batch.append(nxt)
            self.gpu_inflight = len(batch)
            t = time.perf_counter()
            for f in batch:
                f.extra["t_g0"] = t
            lg = self._run_fp16(np.concatenate([f.x for f in batch]))
            self.svc_ewma_ms = 0.8 * self.svc_ewma_ms + 0.2 * (time.perf_counter() - t) * 1e3
            self.gpu_inflight = 0
            t1 = time.perf_counter()
            for f, l in zip(batch, lg):
                f.extra["gpu_batch"] = len(batch)
                f.extra["t_g1"] = t1
                self._emit(f, int(l.argmax()), "fp16")
        self.q_out.put(STOP)

    def _emit(self, f, pred, path):
        f.pred, f.path = pred, path
        f.t_out = time.perf_counter()
        f.x = None
        self.q_out.put(f)

    def _sink(self):
        while (f := self.q_out.get()) is not STOP:
            self.records.append(f)
            if self.on_output:
                self.on_output(f)

    # ---------------- lifecycle ----------------
    def start(self):
        for fn in (self._preprocess, self._int8_stage, self._router, self._gpu_worker, self._sink):
            th = threading.Thread(target=fn, daemon=True, name=fn.__name__)
            th.start()
            self.threads.append(th)

    def close(self):
        """Signal end of stream and wait for every in-flight frame to reach the sink."""
        self.q_cap.put(STOP)
        for th in self.threads:
            th.join()
