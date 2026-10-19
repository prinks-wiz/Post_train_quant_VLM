"""Pre-flight check: `python -m demo.check [--probe-webcam]`.

Verifies models, specs, results files, a real worker launch (load + warm-up time, RSS, no torch) and
webcam presence, then prints what will and won't work in the demo.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time

from demo import offline
from demo.configs import CONFIGS

ROOT = offline.ROOT
LEAN = ROOT / "artifacts" / "lean"
OK, WARN, BAD = "\033[32m✓\033[0m", "\033[33m!\033[0m", "\033[31m✗\033[0m"


def line(mark, msg):
    print(f"  {mark} {msg}")


def check_configs():
    print("Live configs (artifacts/lean, gitignored; rebuild with `python -m lean.build`)")
    shared = [LEAN / "test_raw.npy", LEAN / "test_labels.npy", ROOT / "artifacts" / "text_emb.npy"]
    ok_shared = all(p.exists() for p in shared)
    for p in shared:
        if not p.exists():
            line(BAD, f"missing {p.relative_to(ROOT)}")
    usable = []
    for key, c in CONFIGS.items():
        sp = LEAN / "specs" / f"{c['spec']}.json"
        if not sp.exists():
            line(BAD, f"{c['label']}: spec missing ({sp.relative_to(ROOT)})")
            continue
        spec = json.loads(sp.read_text())
        need = [LEAN / spec["onnx"]] + [LEAN / h for h in spec["heads"].values()]
        need += [LEAN / "pos" / f"pos_r{r}.npy" for r in spec["stages"]]
        need += [LEAN / "resize" / f"bicubic_32to{r}.npy" for r in spec["stages"]]
        miss = [p for p in need if not p.exists()]
        if miss or not ok_shared:
            line(BAD, f"{c['label']}: missing {', '.join(str(p.relative_to(ROOT)) for p in miss) or 'shared data'}")
        else:
            line(OK, f"{c['label']}  ({(LEAN / spec['onnx']).stat().st_size / 2**20:.0f} MB model)")
            usable.append(key)
    return usable


def check_worker(key):
    print(f"Worker launch ({CONFIGS[key]['label']})")
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-m", "demo.worker", CONFIGS[key]["spec"]], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ready = None
    for raw in p.stdout:
        ev = json.loads(raw)
        if ev.get("state") == "ready":
            ready = ev
            break
    if ready is None:
        line(BAD, "worker did not become ready: " + p.stderr.read()[-400:])
        return False
    p.stdin.write('{"cmd": "set", "source": "replay", "fps": 30}\n')
    p.stdin.flush()
    lats = []
    for raw in p.stdout:
        ev = json.loads(raw)
        if ev.get("type") == "frame":
            lats.append(ev["lat_ms"])
            if len(lats) >= 60:
                break
    p.stdin.write('{"cmd": "quit"}\n')
    p.stdin.flush()
    p.wait(3)
    lats.sort()
    line(OK, f"ready in {time.time() - t0:.1f} s (load {ready['load_ms']:.0f} ms + warm-up {ready['warm_ms']:.0f} ms)")
    line(OK, f"peak RSS {ready['peak_rss_mb']:.0f} MB · 60 replay frames, p50 {lats[len(lats) // 2]:.1f} ms")
    # The worker must never import torch (it would add hundreds of MB to the RSS readout).
    probe = subprocess.run([sys.executable, "-c", "import sys, demo.worker; print('torch' in sys.modules, 'cv2' in sys.modules)"],
                           cwd=ROOT, capture_output=True, text=True)
    torch_in, cv2_in = probe.stdout.split()
    line(OK if torch_in == "False" else BAD, f"torch imported by worker: {torch_in}")
    line(OK if cv2_in == "False" else WARN, f"OpenCV imported by worker: {cv2_in}")
    return True


def check_results():
    print("Offline results (tabs 2-4, committed in results/)")
    d = offline.load()
    for k, v in offline.PLOTS.items():
        line(OK if k in d["plots"] else WARN, v)
    line(OK if d["phase1"]["auroc"] else WARN, "phase 1 routing AUROC + sweep")
    line(OK if any(r.get("fixed30") for r in d["phase2"]["table"]) else WARN, "phase 2 stream p99 table")
    line(OK if d["phase2"]["warm_idle"] else WARN, "phase 2 FP16 warm/idle numbers (README)")
    line(OK if d["phase3"]["key"] else WARN, f"phase 3 key table ({len(d['phase3']['key'])} rows)")
    line(OK if d["phase3"]["online"] else WARN, f"phase 3 scatter ({len(d['phase3']['online'])} points)")
    for m in d["missing"]:
        line(WARN, f"missing: {m} (tab shows a notice instead)")
    return not d["missing"]


def check_webcam(probe):
    print("Webcam")
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=10).stdout
        names = [l.strip().rstrip(":") for l in out.splitlines() if l.startswith("    ") and l.strip().endswith(":")]
        line(OK if names else WARN, f"cameras found: {', '.join(names) or 'none'}")
    except Exception as e:
        line(WARN, f"could not query cameras ({e})")
        names = []
    line(OK, "the browser asks for camera permission on first use (localhost counts as a secure context)")
    if probe:
        if importlib.util.find_spec("cv2") is None:
            line(WARN, "--probe-webcam needs OpenCV")
        else:
            import cv2
            cap = cv2.VideoCapture(0)
            ok, frame = cap.read()
            cap.release()
            line(OK if ok else BAD, f"grabbed a frame: {frame.shape if ok else 'failed (permission?)'}")
    return bool(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-webcam", action="store_true", help="grab one frame with OpenCV (may trigger a macOS prompt)")
    a = ap.parse_args()
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    for mod in ("numpy", "onnxruntime", "psutil"):
        line(OK if importlib.util.find_spec(mod) else (WARN if mod == "psutil" else BAD), f"import {mod}")
    usable = check_configs()
    worker_ok = check_worker("cascade" if "cascade" in usable else usable[0]) if usable else False
    results_ok = check_results()
    cam = check_webcam(a.probe_webcam)

    print("\nSummary")
    line(OK if worker_ok else BAD, f"Tab 1 Live: {'works' if worker_ok else 'will NOT run inference'}"
         f" ({len(usable)}/{len(CONFIGS)} configs available)")
    line(OK if results_ok else WARN, "Tabs 2-4: " + ("all artifacts present" if results_ok else "some panels will show a missing-artifact notice"))
    line(OK if cam else WARN, "Webcam: " + ("camera present" if cam else "use CIFAR replay, or launch with --no-webcam"))
    print("\nLaunch: python -m demo.app   (or: python run_demo.py)")
    sys.exit(0 if worker_ok else 1)


if __name__ == "__main__":
    main()
