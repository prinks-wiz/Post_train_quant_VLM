"""Benchmark the lean model zoo (RAM x latency x accuracy) and run the online replay for finalists.

    python -m lean.run_bench              # bench grid + ablations + stream finalists + report
    python -m lean.run_bench --report     # rebuild tables/plots from saved runs

Every measurement is its own process (torch-free for lean specs), 3 interleaved repeats, median.
The INT8->FP16 router from stream/ is measured for peak RSS the same way (whole-process max RSS via
`/usr/bin/time -l`) so RAM numbers are comparable. Holds `caffeinate -dims` while running.
Mine: all orchestration, aggregation and plots.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from .runtime import LEAN_ART, ROOT

OUT = ROOT / "results" / "lean"
RUNS = OUT / "runs"
REPS = 3
# Chosen from the accuracy table (results/lean/accuracy.csv) before any latency was measured.
FINALISTS = ["int8_dyn_d12_aligned_r224", "int8_dyn_d12_aligned_r128-224_f20", "int8_dyn_d12_aligned_r160-224_f10",
             "int8_static_d12_aligned_r128-224_f20", "int4_d12_aligned_r128-224_f20", "int8_dyn_d12_probe_r128-224_f10"]
ABLATE = ("int8_dyn_d12_aligned_r224", "int8_dyn_d12_aligned_r128-224_f20")
# Round 2 (RAM): INT4 with INT8 compute (accuracy_level=4), OpenCV-free preprocessing (__np), and ORT weight
# prepacking disabled (__np_nopack); plus a deadline-aware cascade in the online replay.
ROUND2_BASES = ("int8_dyn_d12_aligned_r224", "int8_dyn_d12_aligned_r128-224_f20", "int8_dyn_d12_aligned_r160-224_f10",
                "int4_a4_d12_aligned_r224", "int4_a4_d12_aligned_r128-224_f20", "int8_dyn_d12_probe_r128-224_f10")
FINALISTS2 = [("int4_a4_d12_aligned_r224__np", False), ("int4_a4_d12_aligned_r128-224_f20__np", False),
              ("int4_a4_d12_aligned_r128-224_f20__np", True), ("int8_dyn_d12_aligned_r224__np_nopack", False),
              ("int8_dyn_d12_aligned_r128-224_f20__np", False), ("int8_dyn_d12_aligned_r128-224_f20__np", True),
              ("int8_dyn_d12_probe_r128-224_f10__np", True)]


def specs_for_bench():
    names = sorted(p.stem for p in (LEAN_ART / "specs").glob("*.json"))
    zs = [n for n in names if "_aligned_" in n]
    probe = [n for n in names if "_probe_" in n and n.startswith("int8_dyn")]
    return zs + probe


def run_json(cmd, path):
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(2):  # one retry: a run once died silently mid-grid (exit code logged)
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0 and path.exists():
            return json.loads(path.read_text())
        print(f"run failed (exit {r.returncode}, attempt {attempt}): {' '.join(cmd[2:5])}\n{r.stderr[-500:]}",
              flush=True)
    raise RuntimeError(f"giving up: {cmd}")


def bench(spec, rep, threads=-1, lean_mem=True):
    tag = f"{spec}__t{threads}__{'lean' if lean_mem else 'arena'}__rep{rep}"
    path = RUNS / "bench" / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "lean.bench", "--spec", spec, "--out", str(path), "--threads", str(threads)]
    if not lean_mem:
        cmd.append("--no-lean-mem")
    r = run_json(cmd, path)
    return {**r, "rep": rep}


def stream(spec, source, rep, deadline=False):
    path = RUNS / "stream" / f"{spec}{'+deadline' if deadline else ''}__{source}__rep{rep}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "lean.stream_sim", "--spec", spec, "--source", source, "--out", str(path)]
    r = run_json(cmd + (["--deadline-cascade"] if deadline else []), path)
    return {**r, "rep": rep}


def router_rss(variant, source, rep):
    """Peak RSS of the stream/ INT8->FP16 (torch + MPS) process for the same replay."""
    path = RUNS / "router" / f"{variant}__{source}__rep{rep}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text())
    spec = json.dumps({"source": source, "variant": variant, "esc_frac": 0.1, "n": 600, "rep": rep})
    r = subprocess.run(["/usr/bin/time", "-l", sys.executable, "-m", "stream.run_bench", "--one", spec],
                       cwd=ROOT, capture_output=True, text=True)
    res = json.loads(r.stdout.strip().splitlines()[-1])
    res["peak_rss_mb"] = int(re.search(r"(\d+)\s+maximum resident set size", r.stderr).group(1)) / 2**20
    res["rep"] = rep
    path.write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--round2", action="store_true")
    a = ap.parse_args()
    if a.round2:
        awake = subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
        try:
            names = [p.stem for p in (LEAN_ART / "specs").glob("*.json")]
            r2 = sorted(n for n in names if n.startswith("int4_a4") and "_aligned_" in n)
            r2 += [f"{b}__{suf}" for b in ROUND2_BASES for suf in ("np", "np_nopack")]
            for rep in range(REPS):
                for s in r2:
                    bench(s, rep)
                print(f"round2 bench rep {rep} done", flush=True)
            for rep in range(REPS):
                for src in ("fixed30", "fixed60", "bursty"):
                    for s, dl in FINALISTS2:
                        stream(s, src, rep, deadline=dl)
                    stream("int8_dyn_d12_aligned_r128-224_f20", src, rep, deadline=True)
                print(f"round2 stream rep {rep} done", flush=True)
        finally:
            awake.terminate()
    elif not a.report:
        awake = subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
        try:
            specs = specs_for_bench()
            for rep in range(REPS):
                for s in specs:
                    bench(s, rep)
                for s in ABLATE:
                    for t in (1, 2, 4):
                        bench(s, rep, threads=t)
                    bench(s, rep, lean_mem=False)
                print(f"bench rep {rep} done", flush=True)
            for rep in range(REPS):
                for src in ("fixed30", "fixed60", "bursty"):
                    for s in FINALISTS:
                        stream(s, src, rep)
                    for v in ("route_naive", "fp16_only", "int8_only"):
                        router_rss(v, src, rep)
                print(f"stream rep {rep} done", flush=True)
        finally:
            awake.terminate()
    from .report import write_report
    write_report()


if __name__ == "__main__":
    main()
