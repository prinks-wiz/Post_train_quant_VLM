"""Inference worker process: `python -m demo.worker <spec>`.

Imports only numpy, onnxruntime and lean.runtime (never torch), so the RSS the UI shows for this
process is what the lean deployment pays. Talks JSON lines: commands on stdin, events on stdout.

Mine: replay scheduler, queue/drop policy, deadline veto wiring, per-stage tracing.
Library: onnxruntime (via lean.runtime.Model).

Commands: {"cmd": "set", "source": "replay"|"external"|"idle", "fps": 30, "deadline": bool}
          {"cmd": "frame", "rgb": <base64 of 32x32x3 uint8>}
          {"cmd": "quit"}
Events:   {"type": "status", ...}  {"type": "frame", ...}
"""
import base64
import json
import queue
import resource
import sys
import threading
import time

import numpy as np

from lean.runtime import LEAN_ART, Model, load_spec, margin

N_REPLAY = 600          # same seeded subset size as lean/stream_sim.py and stream/


def emit(obj, _lock=threading.Lock()):
    with _lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20   # bytes on macOS


def softmax(lg):
    z = np.exp(lg - lg.max())
    return z / z.sum()


def main():
    spec_name = sys.argv[1]
    t0 = time.perf_counter()
    emit({"type": "status", "state": "loading", "msg": f"loading {spec_name}"})
    spec = load_spec(spec_name)
    model = Model(spec)

    # Record each stage's logits and timing without changing Model's cascade logic.
    trace = []
    run_stage = model.run_stage

    def traced(raw, res):
        t = time.perf_counter()
        lg = run_stage(raw, res)
        trace.append((res, lg, (time.perf_counter() - t) * 1e3))
        return lg
    model.run_stage = traced

    test = np.load(LEAN_ART / "test_raw.npy", mmap_mode="r")
    labels_all = np.load(LEAN_ART / "test_labels.npy", mmap_mode="r")
    idx = np.random.default_rng(0).permutation(10000)[:N_REPLAY]
    frames = np.ascontiguousarray(test[idx])
    labels = np.asarray(labels_all[idx])
    del test, labels_all
    load_ms = (time.perf_counter() - t0) * 1e3

    emit({"type": "status", "state": "warming", "msg": "warming up every stage"})
    for i in range(20):
        for res in model.stages:
            model.run_stage(frames[i], res)
    trace.clear()
    emit({"type": "status", "state": "ready", "msg": "ready", "load_ms": load_ms,
          "warm_ms": (time.perf_counter() - t0) * 1e3 - load_ms, "stages": model.stages,
          "threshold": spec.get("threshold"), "head_kind": spec.get("head_kind"),
          "peak_rss_mb": peak_rss_mb()})

    ctl = {"source": "idle", "fps": 30.0, "deadline": False, "gen": 0}
    q = queue.Queue(2)       # small bounded queue: a full queue drops the frame
    est2 = [0.010]           # EWMA of the 224 px pass duration (s), learned online
    drops = [0]

    def infer():
        while (item := q.get()) is not None:
            kind, i, raw, label, t_cap = item
            trace.clear()
            vetoed = [False]
            t_mid = [0.0]
            deadline_s = 1.0 / max(ctl["fps"], 1.0)

            def allow_second():
                t_mid[0] = time.perf_counter()
                ok = q.empty() and (t_mid[0] - t_cap + est2[0]) <= deadline_s
                vetoed[0] = not ok
                return ok
            lg, esc = model(raw, allow_second if ctl["deadline"] else None)
            t_out = time.perf_counter()
            if esc and t_mid[0]:
                est2[0] = 0.8 * est2[0] + 0.2 * (t_out - t_mid[0])
            p = softmax(lg[0])
            top = np.argsort(p)[::-1][:3]
            emit({"type": "frame", "kind": kind, "i": int(i), "label": label,
                  "pred": int(top[0]), "top3": [[int(c), float(p[c])] for c in top],
                  "margin1": float(margin(trace[0][1])[0]), "res1": trace[0][0],
                  "escalated": bool(esc), "vetoed": vetoed[0],
                  "stage_ms": [round(t[2], 2) for t in trace],
                  "lat_ms": (t_out - t_cap) * 1e3, "drops": drops[0],
                  "deadline_ms": deadline_s * 1e3, "deadline_on": ctl["deadline"],
                  "img": base64.b64encode(np.ascontiguousarray(raw).tobytes()).decode(),
                  "peak_rss_mb": peak_rss_mb()})

    def replay():
        """Frames at scheduled capture times; latency is measured from the scheduled time."""
        k, gen, t_next = 0, -1, 0.0
        while True:
            if ctl["source"] != "replay":
                time.sleep(0.02)
                continue
            if gen != ctl["gen"]:
                gen, t_next = ctl["gen"], time.perf_counter() + 0.02
            while (d := t_next - time.perf_counter()) > 0:      # coarse sleep, then spin
                time.sleep(d - 0.008 if d > 0.010 else 0)
            i = k % N_REPLAY
            try:
                q.put_nowait(("replay", i, frames[i], int(labels[i]), t_next))
            except queue.Full:
                drops[0] += 1
            k += 1
            t_next += 1.0 / max(ctl["fps"], 1.0)
            if time.perf_counter() - t_next > 0.5:       # fell far behind (e.g. laptop stall): resync
                t_next = time.perf_counter()

    threading.Thread(target=infer, daemon=True).start()
    threading.Thread(target=replay, daemon=True).start()

    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        cmd = msg.get("cmd")
        if cmd == "set":
            for key in ("source", "fps", "deadline"):
                if key in msg:
                    ctl[key] = msg[key]
            ctl["gen"] += 1
        elif cmd == "frame":
            raw = np.frombuffer(base64.b64decode(msg["rgb"]), np.uint8).reshape(32, 32, 3)
            try:
                q.put_nowait(("external", -1, raw, None, time.perf_counter()))
            except queue.Full:
                drops[0] += 1
        elif cmd == "quit":
            break
    sys.exit(0)


if __name__ == "__main__":
    main()
