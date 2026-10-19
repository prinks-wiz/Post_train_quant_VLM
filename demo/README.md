# Interview demo (`demo/`)

A local web app that runs the lean CPU pipeline live and walks through the three phases.

```bash
.venv/bin/python -m demo.check          # pre-flight: models, specs, results, worker launch, webcam
.venv/bin/python -m demo.app            # opens http://127.0.0.1:7860/  (same as: python run_demo.py)
.venv/bin/python -m demo.app --no-webcam --port 8000 --no-browser
.venv/bin/python -m demo.build_static   # static GitHub Pages build in demo/_site (stop demo.app first)
```

Keys `1`–`4` switch tabs; `#p1`, `#p2` and `#p3` deep-link to a tab.

## How it works

```
browser (one HTML page, polls /api/state at ~8 Hz)
   │  webcam/upload: center-crop + downsample to 32×32 RGB in the browser
   ▼
demo.app   stdlib http.server · rolling p50/p99/fps/escalation · reads worker RSS via psutil
   │  JSON lines over stdin/stdout
   ▼
demo.worker   numpy + onnxruntime + lean.runtime only (torch and OpenCV never imported)
              replay scheduler (coarse sleep + spin) → bounded queue (full = drop) → lean.runtime.Model
              cascade, with the deadline veto from lean/stream_sim.py
```

- **Honest RSS:** the readout is the *worker's* RSS, measured from outside the process. Switching
  config restarts the worker, so the number always covers exactly one config.
- **Same pipeline as the benchmarks:** `Model.__call__` from `lean/runtime.py` runs unchanged.
  The worker only wraps `run_stage` to record each stage's logits and timing.
- **Webcam/upload** frames are downsampled to 32×32, so they go through the same 32→res resize
  matrices as CIFAR. That's a domain shift, so the demo makes no accuracy claim for them.
- **Live vs offline:** tab 1 shows live, noisy laptop numbers next to the committed offline
  numbers for the same config. Tabs 2–4 read only `results/` (plus the FP16 warm/idle probe numbers,
  which exist only in the README's root-cause table). A missing file shows a notice instead of
  crashing.
- **Static build:** GitHub Pages can't run inference, so `build_static` records 450 real frames from the
  worker and the page replays them, labelled "recorded, not live".

| Mine | Library |
|---|---|
| worker isolation and supervision, replay scheduler, drop policy, deadline-veto wiring, rolling metrics, results parsing, charts, recorded playback | onnxruntime, `http.server`, psutil, browser `getUserMedia` |
