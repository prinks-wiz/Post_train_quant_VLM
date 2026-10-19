# Project prompt: CLIP post-training quantization, online routing, and lean CPU inference

You are working in `Post_train_quant_VLM`, a self-contained research sandbox. It studies how to
run a vision-language model (OpenAI CLIP ViT-B/32, zero-shot image classification) fast and with
little memory on an edge-class laptop. The work has three phases, each in its own package. Each
builds on the previous one and must not modify it. Read this whole prompt before changing anything;
most of the non-obvious lessons are in "Pitfalls" below.

## Setting

| | |
|---|---|
| Model | `openai/clip-vit-base-patch32` (HF Transformers). Only the **image encoder** is optimized; text embeddings are computed once in FP32 and frozen |
| Task | CIFAR-10 zero-shot, frozen prompt `"a photo of a {}."`, logit scale 100, cosine similarity |
| Data | HF parquet mirror `uoft-cs/cifar10` (same images, order and labels as the original). The calibration/fitting pool is 5,000 fixed images from the **train** split (seed 0); the test split is evaluation only |
| Hardware | Apple M1 Pro (**6 performance + 2 efficiency cores**), 16 GB unified memory, macOS. GPU = PyTorch MPS. Shared, interactive laptop (expect background noise) |
| Env | `.venv` (Python 3.13): torch 2.14.1, transformers 5.20.1, onnxruntime 1.31.0, onnx 1.23.2; pins in `requirements.txt`. The python.org Python lacks SSL certs, so download with curl or `huggingface_hub` |
| Baseline accuracy | FP32 = 88.30% (original PIL preprocessing), 88.41% (OpenCV-bicubic preprocessing used in `lean/`) |

`artifacts/` (models, caches, features) and `data/` are gitignored and rebuildable; `results/` is
committed.

## Phase 1: `ptq/` (static INT8 PTQ + offline confidence routing)

**Hypothesis tested:** choosing calibration images by embedding-space coverage (greedy k-center on
FP32 CLIP embeddings) beats random selection for static INT8 PTQ.

- **Method:** ONNX export of the image encoder, then ONNX Runtime static QDQ quantization (per-channel
  INT8 weights, UINT8 activations, MinMax calibration; MatMul/Gemm/Conv only, LayerNorm/Softmax/GELU
  kept FP32). Calibration sets N = 32/128/512, k-center vs random (3 seeds), nested prefixes.
- **Result: negative.** K-center was never better and was worse at N=512 (84.26% vs 85.68% random
  mean). Likely cause: farthest-point sampling picks outliers, and MinMax ranges then stretch to
  cover them. N barely matters for random selection.
- **INT8 on ORT CPU:** 2.4× faster than FP32 (13.0 vs 30.9 ms batch-1), 4× smaller file, −2.5 pt
  accuracy.
- **Offline routing:** if INT8's top1−top2 softmax margin is low, re-run on FP16 (MPS). Margin
  detects INT8≠FP32 disagreements with AUROC 0.879. Escalating 20% recovers 88.0%. But an
  end-to-end run measured 33.9 ms p99 vs 22.7 ms estimated from per-model timings, and that gap
  motivated Phase 2.
- **Entry points:** `python -m ptq.run_all` (about 35 min) and `python -m ptq.demo`.

## Phase 2: `stream/` (online INT8 → FP16 routing pipeline)

- **Design:** one thread per stage with bounded queues: capture → preprocess → INT8 (ORT CPU) →
  router → FP16 worker (MPS) → output. Every fix is a flag in `stream.pipeline.Config`:
  - keep-warm (dummy GPU inference when idle)
  - micro-batching of escalations
  - deadline-aware routing (keep INT8 if the escalation would miss the frame deadline)
- **Sources:** seeded CIFAR test replay at 30 fps, 60 fps and bursty (bursts at 60 fps, gaps of
  100–400 ms); webcam demo with temporal reuse. Escalation threshold = 10% quantile of INT8 margins
  on the train pool.
- **Root cause found:** warm FP16 batch-1 on MPS takes 8.8 ms, but after ≥50 ms idle it takes
  about 23 ms. Only full-model forward passes keep it warm; tiny ops, layer chunks, unrelated
  matmuls and CPU spinning do not.
- **Result:** FP16 on the GPU for every frame beat every routing variant on p99 and accuracy (30 fps:
  19.1 ms p99, 87.5%), because continuous traffic keeps the GPU warm. Keep-warm and micro-batching
  did not lower p99. Deadline-aware routing was the only fix that trimmed the tail, and it costs
  accuracy when deadlines are tight (at 60 fps it disables escalation entirely).
- **Entry points:** `python -m stream.run_bench` (about 1 h; 5 interleaved repeats + health-check
  filter) and `python -m stream.demo --source webcam`.

## Phase 3: `lean/` (lowest RAM × lowest latency, CPU-only, no torch at runtime)

**Goal:** online inference at minimum RAM and latency, using embedding-space reorganization rather
than retraining.

**Why CPU-only:** on unified memory the torch + MPS router process peaks at about 1,010 MB.

**Embedding-space techniques (all label-free unless noted):**
1. **Aligned heads.** A ridge-regression map (768→512, λ chosen on a holdout) sends a cheaper
   encoder's post-LN CLS feature (lower resolution via interpolated position embeddings, or fewer
   layers) to the full FP32 CLIP image embedding. The frozen text embeddings then classify as
   before. Without it, early-exit features give 10% (chance) up to layer 10.
2. **Quantization-aware alignment.** Fit the map on the *quantized* trunk's own features. At 224 px
   this recovers INT8-static from 86.35% to 88.04% and INT4 from 82.60% to 86.41%; INT8-dynamic
   reaches 89.08% (above FP32).
3. **Shared-weight resolution cascade.** The ONNX trunk takes the position-embedding table as an
   input, so one session (one weight copy) serves any resolution. Run at 128 px; if the margin is
   below a train-pool threshold, re-run at 224 px. It beat depth cascades offline.
4. **Linear probe (uses 5,000 train labels; not zero-shot, reported separately):** 93.2–93.7%.

**Deployment levers:**
- INT8 dynamic is the best zero-shot precision.
- INT4 `MatMulNBits` needs `accuracy_level=4` (INT8 compute), or it is as slow as FP32.
- OpenCV-free preprocessing: exact OpenCV bicubic resize matrices applied with numpy, which cuts
  base RSS from 74 to 50 MB.
- Rejected: ORT arena off (only about −10 MB, kept as default), prepacking off (hurts INT4 badly),
  fewer threads (no RAM saving).
- Deadline-aware cascade: veto the 224 px pass if a frame is queued or it would miss the deadline.

**Key results** (batch-1, preprocessing included, full-test accuracy, whole-process peak RSS):

| Config | Top-1 | p50 / p99 (ms) | Peak RSS |
|---|---|---|---|
| torch + MPS router (reference) | 86.7% | 12.6 / 34.9 (30 fps replay) | ~1,010 MB |
| INT8-dyn 224, numpy prep | 88.8% | 10.4 / 11.2 | 188 MB |
| INT8-dyn cascade 128→224, numpy prep | 88.4% | 5.2 / 16.0 | 182 MB |
| INT4(a4) 224, numpy prep | 86.7% | 11.7 / 12.9 | 171 MB |
| INT8-dyn probe cascade (labels) | 93.2% | 5.1 / 16.0 | 184 MB |

**Entry points:**
- `python -m lean.build`: models, heads, accuracy table
- `python -m lean.run_bench`, then `--round2`: about 2.5 h
- `python -m lean.accuracy_variants`
- `python -m lean.run_bench --report`

**Layout:** deploy code is in `lean/runtime.py` (numpy, onnxruntime and optional OpenCV only).
Offline torch code is in `lean/offline.py` and `lean/build.py`. Specs are JSON files in
`artifacts/lean/specs/` with these fields: `onnx`, `stages` (resolutions), `heads` per resolution,
`threshold`, `prep`, `ort_config`.

## Pitfalls already discovered (don't rediscover them)

- **ORT 1.31 `CalibMaxIntermediateOutputs`** silently drops calibration batches (it clears without
  merging). Never set it. It's also unnecessary: calibration only stores per-tensor min/max.
- **ORT default threads = all 8 cores** includes efficiency cores, which causes stragglers (up to
  58 ms stalls). Use `intra_op_num_threads = hw.perflevel0.physicalcpu` (6).
- **macOS `time.sleep` oversleeps 4–5 ms.** Schedule with a coarse sleep, then a yield-spin.
- **Display sleep / App Nap throttles timers.** Hold `caffeinate -dims` during benchmarks.
- **Background load on the shared laptop** produces p99 outliers. Use isolated processes,
  interleaved repeats, medians, and a variant-independent health-check canary. Treat differences
  under about 1 ms or 10 MB as noise.
- **MPS compiles kernels per input shape.** Warm up every batch size you will use.
- **`quant_pre_process` with symbolic shape inference** fails on the dynamic-resolution graph (use
  `skip_symbolic_shape=True`), and it can leave a large `*.data` temp file in the CWD (now
  gitignored).
- **Never read latency from a process that imported torch** when measuring the lean path's RAM:
  the import alone costs hundreds of MB.
- **Probe heads are overconfident on their own training pool,** so train-pool margin thresholds
  under-predict test escalation rates.

## Working rules for this repo

- **Don't modify earlier phases.** `ptq/` code and results are frozen; `stream/` likewise except
  for bug fixes. Add new work as a new package and append a README section.
- **No test-set tuning.** Fixed seeds; calibration, alignment and thresholds use the train pool
  only.
- **Benchmark hygiene:** one model or variant per fresh process, under `caffeinate`, on
  performance-core threads, with interleaved repeats. Report medians plus ranges, and say which
  backend and device each number came from.
- **Report honestly:** negative results, noise, and mixed-device caveats go in the README. Never
  simulate a result you could not run.
- **Label authorship:** mark which logic is ours (scheduling, routing, alignment, cascades,
  metrics) and which is library code.
- **Commit and push only when asked.**

## Open directions

- **Refit heads on numpy-preprocessed features** to recover the 0.2–0.3 pt lost when OpenCV was
  removed.
- **Lower RAM floors:** ORT minimal build or C API runtime (drops the ~50 MB Python base), sharing
  prepacked weights, external-data mmap.
- **Distillation:** a distilled small encoder (e.g. 4–6 layers at 128 px) trained to the aligned
  target, as the next step beyond linear heads.
- **Better calibration:** percentile/entropy calibration for static INT8, and class-balanced or
  density-weighted calibration selection (Phase 1 follow-ups).
- **Non-CIFAR inputs:** validate on native-resolution camera frames, where low-resolution gains
  will be smaller than on 32×32 CIFAR.
