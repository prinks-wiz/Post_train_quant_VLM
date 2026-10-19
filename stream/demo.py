"""Live streaming demo (visual only - no accuracy is computed for webcam frames).

    python -m stream.demo --source webcam                       # all fixes on, temporal reuse on
    python -m stream.demo --source webcam --no-keep-warm --no-reuse
    python -m stream.demo --source replay --fps 30              # CIFAR-10 replay in a window
    python -m stream.demo --source webcam --headless --max-frames 300   # no window; print stats

The window shows each frame's prediction, margin (top1-top2 softmax prob), whether it escalated to
FP16, and a rolling p50/p99 capture->output latency and output fps. Press q to quit.

Mine: capture loop, temporal reuse (skip inference when the frame barely changed), overlay/stats.
Library: OpenCV capture and drawing; the pipeline itself is stream/pipeline.py.
"""
import argparse
import collections
import threading
import time

import numpy as np

from ptq.common import CIFAR10_CLASSES

from .pipeline import Config, Frame, Pipeline
from .sources import replay_preprocess, replay_subset, webcam_preprocess
from .thresholds import threshold_for


class Stats:
    def __init__(self, window=120):
        self.lat = collections.deque(maxlen=window)
        self.t_out = collections.deque(maxlen=window)
        self.last = None
        self.lock = threading.Lock()
        self.n_esc = self.n_reuse = self.n = 0

    def add(self, f: Frame):
        with self.lock:
            self.lat.append((f.t_out - f.t_cap) * 1e3)
            self.t_out.append(f.t_out)
            self.last = f
            self.n += 1
            self.n_esc += f.path == "fp16"
            self.n_reuse += f.path == "reuse"

    def snapshot(self):
        with self.lock:
            lat = np.array(self.lat) if self.lat else np.array([np.nan])
            fps = (len(self.t_out) - 1) / (self.t_out[-1] - self.t_out[0]) if len(self.t_out) > 1 else 0.0
            return self.last, np.percentile(lat, 50), np.percentile(lat, 99), fps


def tiny_gray(bgr_or_chw):
    """32x32 grayscale thumbnail for the temporal-reuse difference test."""
    import cv2
    img = bgr_or_chw.transpose(1, 2, 0) if bgr_or_chw.shape[0] == 3 else bgr_or_chw
    g = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def draw(img, last, p50, p99, fps, stats):
    import cv2
    h = img.shape[0]
    scale = max(h / 480, 0.6)
    lines = [f"p50 {p50:5.1f} ms   p99 {p99:5.1f} ms   {fps:4.1f} fps",
             f"escalated {stats.n_esc}/{stats.n}   reused {stats.n_reuse}"]
    if last is not None:
        tag = {"fp16": "ESCALATED -> FP16 (GPU)", "int8": "INT8 (CPU)", "reuse": "REUSED previous",
               "int8_deadline": "INT8 (escalation skipped: deadline)", "int8_qfull": "INT8 (esc. queue full)"}
        lines = [f"{CIFAR10_CLASSES[last.pred]}   margin {last.margin:.2f}",
                 tag.get(last.path, last.path)] + lines
    cv2.rectangle(img, (0, 0), (img.shape[1], int((len(lines) * 28 + 12) * scale)), (20, 20, 20), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (10, int((28 + 28 * i) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale,
                    (240, 240, 240), max(1, int(2 * scale)), cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcam", "replay"], default="webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0, help="replay rate")
    ap.add_argument("--esc-frac", type=float, default=0.10)
    ap.add_argument("--no-keep-warm", action="store_true")
    ap.add_argument("--no-microbatch", action="store_true")
    ap.add_argument("--no-deadline", action="store_true")
    ap.add_argument("--no-reuse", action="store_true")
    ap.add_argument("--reuse-thresh", type=float, default=4.0, help="mean abs gray diff (0-255) to reuse")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    import cv2

    cfg = Config(mode="route", threshold=threshold_for(a.esc_frac), keep_warm=not a.no_keep_warm,
                 microbatch=not a.no_microbatch, deadline_aware=not a.no_deadline, deadline_ms=1000 / 30)
    stats = Stats()
    prep = webcam_preprocess if a.source == "webcam" else replay_preprocess
    pipe = Pipeline(cfg, prep, on_output=stats.add)
    pipe.warmup(replay_preprocess(np.zeros((3, 224, 224), np.uint8)))
    pipe.start()

    if a.source == "webcam":
        cap = cv2.VideoCapture(a.camera)
        if not cap.isOpened():
            raise SystemExit("Could not open webcam (on macOS, grant Camera access to your terminal/IDE "
                             "in System Settings > Privacy & Security > Camera).")
    else:
        from ptq.data_cache import load
        x, _ = load("test")
        idx = replay_subset(10000)

    prev_thumb, fid, t_next = None, 0, time.perf_counter()
    try:
        while not a.max_frames or fid < a.max_frames:
            if a.source == "webcam":
                ok, raw = cap.read()
                if not ok:
                    break
                shown = raw
            else:
                t_next += 1 / a.fps
                while (d := t_next - time.perf_counter()) > 0:
                    time.sleep(d - 0.008 if d > 0.010 else 0)
                raw = np.asarray(x[idx[fid % len(idx)]])
                shown = cv2.resize(raw.transpose(1, 2, 0)[:, :, ::-1], (448, 448), interpolation=cv2.INTER_NEAREST)
            t_cap = time.perf_counter()
            thumb = tiny_gray(raw)
            last = stats.last
            if (not a.no_reuse and prev_thumb is not None and last is not None
                    and float(np.abs(thumb - prev_thumb).mean()) < a.reuse_thresh):
                f = Frame(fid=fid, t_cap=t_cap, raw=None, pred=last.pred, margin=last.margin, path="reuse")
                f.t_out = time.perf_counter()
                stats.add(f)
            else:
                prev_thumb = thumb
                pipe.submit(Frame(fid=fid, t_cap=t_cap, raw=raw))
            fid += 1
            last, p50, p99, fps = stats.snapshot()
            if a.headless:
                if fid % 30 == 0:
                    print(f"frame {fid}: p50 {p50:.1f} ms p99 {p99:.1f} ms {fps:.1f} fps "
                          f"esc {stats.n_esc} reuse {stats.n_reuse} dropped {len(pipe.dropped)}", flush=True)
                continue
            cv2.imshow("CLIP INT8 -> FP16 streaming router", draw(shown.copy(), last, p50, p99, fps, stats))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipe.close()
        if a.source == "webcam":
            cap.release()
        cv2.destroyAllWindows()
    _, p50, p99, fps = stats.snapshot()
    print(f"final: {stats.n} outputs, p50 {p50:.1f} ms, p99 {p99:.1f} ms, {fps:.1f} fps, "
          f"escalated {stats.n_esc}, reused {stats.n_reuse}, dropped {len(pipe.dropped)}")


if __name__ == "__main__":
    main()
