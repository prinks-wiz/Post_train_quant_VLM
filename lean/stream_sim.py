"""Online replay for the lean CPU-only path (single process, no torch, no GPU).

    python -m lean.stream_sim --spec NAME --source fixed30 --out JSON

capture thread (scheduled arrivals of raw 32x32 CIFAR-10 test frames, same seeded subset/schedules as
stream/) -> bounded queue (full = drop) -> one inference worker (preprocess + trunk [+ cascade stage B]
+ aligned head) -> records. Latency = output time - scheduled capture time. Peak RSS of the whole
process is reported alongside a pre/post machine-health canary.

Mine: all of it except ORT/OpenCV calls.
"""
import argparse
import json
import queue
import threading
import time

import numpy as np

SOURCES = {"fixed30": ("fixed", 30.0), "fixed60": ("fixed", 60.0), "bursty": ("bursty", 30.0)}


def schedule(n, mode, fps, seed=0):
    # Same definitions as stream/sources.py (duplicated so this process stays torch-free).
    if mode == "fixed":
        return np.arange(n) / fps
    rng = np.random.default_rng(seed)
    t, out = 0.0, []
    while len(out) < n:
        for _ in range(rng.geometric(1 / 8)):
            out.append(t)
            t += 1 / 60
        t += rng.uniform(0.10, 0.40)
    return np.array(out[:n])


def replay_subset(n, seed=0):
    return np.random.default_rng(seed).permutation(10000)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--source", default="fixed30", choices=list(SOURCES))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--deadline-cascade", action="store_true",
                    help="skip the 2nd (224px) pass if another frame is queued or it would miss the deadline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from .bench import peak_mb, rss_now_mb
    from .runtime import LEAN_ART, Model, load_spec
    raw = np.load(LEAN_ART / "test_raw.npy", mmap_mode="r")
    labels = np.load(LEAN_ART / "test_labels.npy")
    idx = replay_subset(a.n)
    frames = np.ascontiguousarray(raw[np.sort(idx)])[np.argsort(np.argsort(idx))]
    y = labels[idx]
    rss_base = rss_now_mb()
    model = Model(load_spec(a.spec), threads=a.threads)
    for i in range(30):
        model(frames[i])

    def canary():
        ts = []
        for i in range(40):
            t = time.perf_counter()
            model(frames[i])
            ts.append((time.perf_counter() - t) * 1e3)
        return float(np.median(ts))

    pre = canary()
    mode, fps = SOURCES[a.source]
    offs = schedule(a.n, mode, fps)
    q = queue.Queue(a.q)
    t_cap = np.zeros(a.n)
    t_out = np.full(a.n, np.nan)
    pred = np.full(a.n, -1)
    stage_b = np.zeros(a.n, bool)
    dropped = []

    deadline_s = 1.0 / fps
    est2 = [0.010]          # EWMA of the second-pass duration (s), learned online
    vetoed = [0]

    def worker():
        while (i := q.get()) is not None:
            t_mid = [0.0]

            def allow_second():
                t_mid[0] = time.perf_counter()
                ok = q.empty() and (t_mid[0] - t_cap[i] + est2[0]) <= deadline_s
                vetoed[0] += not ok
                return ok
            lg, esc = model(frames[i], allow_second if a.deadline_cascade else None)
            t_out[i] = time.perf_counter()
            if esc and t_mid[0]:
                est2[0] = 0.8 * est2[0] + 0.2 * (t_out[i] - t_mid[0])
            pred[i], stage_b[i] = int(lg.argmax()), esc

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    t0 = time.perf_counter() + 0.05
    for i, off in enumerate(offs):
        t_cap[i] = t0 + off
        while (d := t_cap[i] - time.perf_counter()) > 0:
            time.sleep(d - 0.008 if d > 0.010 else 0)
        try:
            q.put_nowait(i)
        except queue.Full:
            dropped.append(i)
    q.put(None)
    th.join()
    post = canary()

    done = ~np.isnan(t_out)
    lat = (t_out[done] - t_cap[done]) * 1e3
    deadline = 1000.0 / fps
    res = {"spec": a.spec + ("+deadline" if a.deadline_cascade else ""), "source": a.source, "n": a.n,
           "second_pass_vetoed": vetoed[0], "dropped": len(dropped),
           "late": int((lat > deadline).sum()), "miss_rate": float(((lat > deadline).sum() + len(dropped)) / a.n),
           "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
           "p99_ms": float(np.percentile(lat, 99)), "max_ms": float(lat.max()),
           "acc_all": float((pred == y).sum() / a.n), "frac_stage_b": float(stage_b[done].mean()),
           "rss_base_mb": rss_base, "peak_rss_mb": peak_mb(), "peak_delta_mb": peak_mb() - rss_base,
           "canary_ms": max(pre, post)}
    with open(a.out, "w") as fh:
        json.dump(res, fh)


if __name__ == "__main__":
    main()
