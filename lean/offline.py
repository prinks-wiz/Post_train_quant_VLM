"""Offline (torch) side of the lean path: raw data, truncated/low-res CLIP trunks, all-layer features,
label-free embedding-space alignment, ONNX export and quantization.

Mine: LeanTrunk (fixed interpolated position embeddings, depth truncation, cascade split), the
alignment fit (ridge map from layer-k / low-res features to the full FP32 CLIP image embedding),
quantization-aware refit of that map, variant orchestration.
Library: HF CLIP weights/layers, torch.onnx export, ORT quantization tools.
"""
import numpy as np
import torch
from torch import nn

from .runtime import LEAN_ART, TEXT_EMB, LOGIT_SCALE, preprocess

FEAT_DIR = LEAN_ART / "feats"
RES_GRID = (224, 160, 128, 96, 64)      # 7x7, 5x5, 4x4, 3x3, 2x2 patches (+CLS)
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


# ---------------------------------------------------------------- data
def raw_data():
    """CIFAR-10 at native 32x32 RGB uint8: pool = ptq's fixed 5,000 train images; test = full test set."""
    LEAN_ART.mkdir(parents=True, exist_ok=True)
    p_pool, p_test = LEAN_ART / "pool_raw.npy", LEAN_ART / "test_raw.npy"
    if not p_test.exists():
        from ptq.common import CIFAR10Parquet
        from ptq.data_cache import pool_indices
        tr, te = CIFAR10Parquet(True), CIFAR10Parquet(False)
        pool = pool_indices()
        np.save(p_pool, np.stack([np.asarray(tr.pil(int(i))) for i in pool]))
        np.save(LEAN_ART / "pool_labels.npy", np.array([tr.targets[i] for i in pool]))
        np.save(p_test, np.stack([np.asarray(te.pil(i)) for i in range(len(te))]))
        np.save(LEAN_ART / "test_labels.npy", np.array(te.targets))
    return (np.load(p_pool), np.load(LEAN_ART / "pool_labels.npy"),
            np.load(p_test), np.load(LEAN_ART / "test_labels.npy"))


# ---------------------------------------------------------------- model
def clip_vision():
    from ptq.common import load_clip
    clip = load_clip()
    return clip.vision_model.eval(), clip.visual_projection.eval()


class LeanTrunk(nn.Module):
    """CLIP ViT-B/32 vision trunk at resolution `res`, layers [start, stop).

    start == 0: input pixel_values; patch embedding + CLS + position embeddings interpolated to the
    res/32 grid (precomputed constant, so the exported graph has no resize op) + pre-LN.
    start > 0: input `hidden` (B, T, 768) from a previous stage (cascade stage B).
    Outputs post-LN CLS features (B, 768); with `emit_hidden`, also the hidden states.
    """

    def __init__(self, vm, res, start=0, stop=12, emit_hidden=False):
        super().__init__()
        self.start, self.emit_hidden = start, emit_hidden
        emb = vm.embeddings
        self.patch = emb.patch_embedding
        self.cls = emb.class_embedding
        with torch.no_grad():
            dummy = torch.zeros(1, 1 + (res // 32) ** 2, 768)
            pos = emb.interpolate_pos_encoding(dummy, res, res)
        self.register_buffer("pos", pos.detach().clone())
        self.pre_ln = vm.pre_layrnorm
        self.layers = nn.ModuleList(vm.encoder.layers[start:stop])
        self.post_ln = vm.post_layernorm

    def embed(self, x):
        p = self.patch(x).flatten(2).transpose(1, 2)
        h = torch.cat([self.cls.expand(x.shape[0], 1, -1), p], dim=1) + self.pos
        return self.pre_ln(h)

    def forward(self, inp):
        h = self.embed(inp) if self.start == 0 else inp
        for layer in self.layers:
            h = layer(h, None)
        feat = self.post_ln(h[:, 0])
        return (feat, h) if self.emit_hidden else feat

    @torch.no_grad()
    def all_layers(self, x):
        """Post-LN CLS after every layer: (B, n_layers, 768)."""
        h = self.embed(x)
        outs = []
        for layer in self.layers:
            h = layer(h, None)
            outs.append(self.post_ln(h[:, 0]))
        return torch.stack(outs, 1)


# ---------------------------------------------------------------- features
@torch.no_grad()
def extract_features(res_grid=RES_GRID, batch=250):
    """Per resolution: post-LN CLS after each of the 12 layers for pool and test (float16 on disk).
    Also the FP32 full-model (224, 12 layers) projected embedding for the pool = alignment target."""
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    pool, _, test, _ = raw_data()
    vm, proj = clip_vision()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    vm, proj = vm.to(dev), proj.to(dev)
    np.savez(LEAN_ART / "proj.npz", W=proj.weight.detach().cpu().numpy().T)
    for res in res_grid:
        trunk = LeanTrunk(vm, res).to(dev).eval()
        for split, raw in (("pool", pool), ("test", test)):
            path = FEAT_DIR / f"r{res}_{split}.npy"
            if path.exists():
                continue
            out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(raw), 12, 768))
            for i in range(0, len(raw), batch):
                x = torch.from_numpy(preprocess(raw[i:i + batch], res)).to(dev)
                out[i:i + batch] = trunk.all_layers(x).float().cpu().numpy().astype(np.float16)
            out.flush()
            print(f"features r{res} {split}", flush=True)
    tgt = LEAN_ART / "target_pool.npy"
    if not tgt.exists():
        f = np.load(FEAT_DIR / "r224_pool.npy", mmap_mode="r")[:, 11].astype(np.float32)
        z = f @ np.load(LEAN_ART / "proj.npz")["W"]
        np.save(tgt, z / np.linalg.norm(z, axis=1, keepdims=True))


def feats(res, split, depth):
    return np.load(FEAT_DIR / f"r{res}_{split}.npy", mmap_mode="r")[:, depth - 1].astype(np.float32)


# ---------------------------------------------------------------- alignment (embedding-space map)
def fit_ridge(F, T, lambdas=RIDGE_LAMBDAS, holdout=1000, seed=0):
    """Label-free linear map F (n,768) -> T (n,512, unit-norm FP32 CLIP embeddings). lambda chosen by
    holdout MSE (relative to mean feature energy); then refit on all rows. Returns W, b, lambda."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(F))
    va, tr = perm[:holdout], perm[holdout:]

    def solve(idx, lam):
        mf, mt = F[idx].mean(0), T[idx].mean(0)
        Fc, Tc = F[idx] - mf, T[idx] - mt
        G = Fc.T @ Fc
        G += lam * np.trace(G) / G.shape[0] * np.eye(G.shape[0], dtype=G.dtype)
        W = np.linalg.solve(G, Fc.T @ Tc)
        return W, mt - mf @ W

    best = min(lambdas, key=lambda lam: (lambda W, b: ((F[va] @ W + b - T[va]) ** 2).sum(1).mean())(*solve(tr, lam)))
    W, b = solve(np.arange(len(F)), best)
    return W.astype(np.float32), b.astype(np.float32), best


def zeroshot_acc(Z, y):
    T = np.load(TEXT_EMB)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return float(((LOGIT_SCALE * Z @ T.T).argmax(1) == y).mean())


def alignment_grid():
    """FP32 zero-shot accuracy for every (res, depth): naive head (post-LN -> CLIP projection) vs
    aligned head (ridge map fit on unlabeled train pool)."""
    _, _, _, y = raw_data()
    T = np.load(LEAN_ART / "target_pool.npy")
    P = np.load(LEAN_ART / "proj.npz")["W"]
    rows = []
    for res in RES_GRID:
        for depth in range(1, 13):
            Ftr, Fte = feats(res, "pool", depth), feats(res, "test", depth)
            W, b, lam = fit_ridge(Ftr, T)
            rows.append({"res": res, "depth": depth, "tokens": 1 + (res // 32) ** 2,
                         "acc_naive": zeroshot_acc(Fte @ P, y), "acc_aligned": zeroshot_acc(Fte @ W + b, y),
                         "ridge_lambda": lam})
            print(rows[-1], flush=True)
    return rows


# ---------------------------------------------------------------- export & quantize
def export_trunk(path, res, start=0, stop=12, emit_hidden=False):
    if path.exists():
        return path
    vm, _ = clip_vision()
    m = LeanTrunk(vm, res, start, stop, emit_hidden).eval()
    if start == 0:
        dummy, in_name = torch.randn(1, 3, res, res), "pixel_values"
    else:
        dummy, in_name = torch.randn(1, 1 + (res // 32) ** 2, 768), "hidden"
    out_names = ["feat", "hidden"] if emit_hidden else ["feat"]
    axes = {in_name: {0: "batch"}, **{o: {0: "batch"} for o in out_names}}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m, (dummy,), str(path), input_names=[in_name], output_names=out_names,
                      dynamic_axes=axes, opset_version=17, dynamo=False)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    quant_pre_process(str(path), str(path), skip_symbolic_shape=False)
    return path


def quantize_variant(fp32_path, out_path, precision, calib_inputs=None):
    """precision: int8_dyn (weights INT8 per-channel, activations quantized on the fly - no calibration),
    int8_static (QDQ, MinMax-calibrated like ptq/), int4 (weight-only MatMulNBits, block 32, FP32 compute),
    int4_a4 (same weights, accuracy_level=4: INT8 compute)."""
    if out_path.exists():
        return out_path
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic,
                                          quantize_static)
    if precision == "int8_dyn":
        quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=["MatMul", "Gemm"])
    elif precision == "int8_static":
        name, arr = calib_inputs

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter([arr[i:i + 16] for i in range(0, len(arr), 16)])

            def get_next(self):
                b = next(self.it, None)
                return None if b is None else {name: b}

        quantize_static(str(fp32_path), str(out_path), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
                        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    elif precision in ("int4", "int4_a4"):
        # int4_a4: accuracy_level=4 lets ORT compute with INT8 activations instead of dequantizing to FP32.
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
        q = MatMulNBitsQuantizer(onnx.load(str(fp32_path)), block_size=32, is_symmetric=True,
                                 accuracy_level=4 if precision == "int4_a4" else None)
        q.process()
        q.model.save_model_to_file(str(out_path), use_external_data_format=False)
    else:
        raise ValueError(precision)
    return out_path
