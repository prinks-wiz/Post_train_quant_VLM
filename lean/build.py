"""Build the lean model zoo and its accuracy table.

    python -m lean.build

1. One dynamic-resolution trunk (12 layers) exported to ONNX with inputs (pixel_values, pos);
   position tables for each resolution saved as .npy (interpolated once, offline).
2. Precisions: fp32, int8_dyn, int8_static (calibrated on mixed resolutions), int4 (weight-only).
3. For each precision, the quantized trunk's OWN features (pool + test, each resolution) -> heads:
   aligned (label-free ridge to FP32 full CLIP embeddings) and probe (logistic regression, uses labels).
   Fitting on the quantized features = quantization-aware embedding alignment.
4. Accuracy on the full 10k test set for single-resolution models and resolution cascades (thresholds
   set on train-pool margins for target second-pass fractions).
5. Spec JSONs consumed by lean.bench / lean.stream_sim.
Mine: everything here; library: torch.onnx, ORT quantization, scikit-learn LogisticRegression.
"""
import json

import numpy as np
import torch
from torch import nn

from .offline import LeanTrunk, clip_vision, fit_ridge, quantize_variant, raw_data
from .runtime import LEAN_ART, LOGIT_SCALE, TEXT_EMB, margin, preprocess, session

RESOLUTIONS = (128, 160, 224)
PRECISIONS = ("fp32", "int8_dyn", "int8_static", "int4", "int4_a4")
CASCADES = ((128, 224), (160, 224))
CASCADE_FRACS = (0.10, 0.20, 0.30)
MODELS, POS, HEADS, QFEATS, SPECS = (LEAN_ART / d for d in ("models", "pos", "heads", "qfeats", "specs"))


class DynTrunk(nn.Module):
    """LeanTrunk with the position table as an input -> one graph for every resolution."""

    def __init__(self, vm, depth=12):
        super().__init__()
        t = LeanTrunk(vm, 224, 0, depth)
        self.patch, self.cls, self.pre_ln, self.layers, self.post_ln = t.patch, t.cls, t.pre_ln, t.layers, t.post_ln

    def forward(self, pixel_values, pos):
        p = self.patch(pixel_values).flatten(2).transpose(1, 2)
        h = self.pre_ln(torch.cat([self.cls.expand(pixel_values.shape[0], 1, -1), p], dim=1) + pos)
        for layer in self.layers:
            h = layer(h, None)
        return self.post_ln(h[:, 0])


def export(depth=12):
    path = MODELS / f"trunk_d{depth}_fp32.onnx"
    POS.mkdir(parents=True, exist_ok=True)
    vm, _ = clip_vision()
    for r in (64, 96, 128, 160, 224):
        p = POS / f"pos_r{r}.npy"
        if not p.exists():
            with torch.no_grad():
                pos = vm.embeddings.interpolate_pos_encoding(torch.zeros(1, 1 + (r // 32) ** 2, 768), r, r)
            np.save(p, pos.numpy().astype(np.float32))
    if path.exists():
        return path
    MODELS.mkdir(parents=True, exist_ok=True)
    m = DynTrunk(vm, depth).eval()
    torch.onnx.export(m, (torch.randn(1, 3, 224, 224), torch.from_numpy(np.load(POS / "pos_r224.npy"))),
                      str(path), input_names=["pixel_values", "pos"], output_names=["feat"],
                      dynamic_axes={"pixel_values": {0: "batch", 2: "h", 3: "w"}, "pos": {1: "tokens"},
                                    "feat": {0: "batch"}}, opset_version=17, dynamo=False)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    # Symbolic shape inference can't resolve the dynamic token axis; ONNX shape inference suffices.
    quant_pre_process(str(path), str(path), skip_symbolic_shape=True)
    return path


from onnxruntime.quantization import CalibrationDataReader  # noqa: E402


class MixedResReader(CalibrationDataReader):
    """Static-INT8 calibration on 48 random pool images at each deployed resolution."""

    def __init__(self, pool):
        idx = np.random.default_rng(0).permutation(len(pool))[:48]
        self.batches = [{"pixel_values": preprocess(pool[idx[i:i + 16]], r), "pos": np.load(POS / f"pos_r{r}.npy")}
                        for r in RESOLUTIONS for i in range(0, 48, 16)]
        self.it = iter(self.batches)

    def get_next(self):
        return next(self.it, None)


def build_models(depth=12):
    fp32 = export(depth)
    pool, *_ = raw_data()
    out = {"fp32": fp32}
    for prec in PRECISIONS[1:]:
        dst = MODELS / f"trunk_d{depth}_{prec}.onnx"
        if prec == "int8_static" and not dst.exists():
            from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
            quantize_static(str(fp32), str(dst), MixedResReader(pool), quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                            op_types_to_quantize=["MatMul", "Gemm", "Conv"],
                            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
        elif prec != "int8_static":
            quantize_variant(fp32, dst, prec)
        out[prec] = dst
    return out


def model_feats(prec, path, res, split, raw, batch=100):
    p = QFEATS / f"{path.stem}_r{res}_{split}.npy"
    if p.exists():
        return np.load(p)
    QFEATS.mkdir(parents=True, exist_ok=True)
    s = session(path, lean_mem=False)
    pos = np.load(POS / f"pos_r{res}.npy")
    f = np.concatenate([s.run(None, {"pixel_values": preprocess(raw[i:i + batch], res), "pos": pos})[0]
                        for i in range(0, len(raw), batch)])
    np.save(p, f)
    return f


def fit_heads(F_pool, y_pool, tag):
    HEADS.mkdir(parents=True, exist_ok=True)
    W, b, lam = fit_ridge(F_pool, np.load(LEAN_ART / "target_pool.npy"))
    np.savez(HEADS / f"{tag}_aligned.npz", W=W, b=b, kind="aligned", ridge_lambda=lam)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(F_pool, y_pool)
    np.savez(HEADS / f"{tag}_probe.npz", W=clf.coef_.T.astype(np.float32), b=clf.intercept_.astype(np.float32),
             kind="probe")
    return {"aligned": f"heads/{tag}_aligned.npz", "probe": f"heads/{tag}_probe.npz"}


def head_logits(path, F):
    d = np.load(LEAN_ART / path)
    z = F @ d["W"] + d["b"]
    if str(d["kind"]) == "probe":
        return z
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return LOGIT_SCALE * z @ np.load(TEXT_EMB).T


def main(depth=12):
    pool, py, test, ty = raw_data()
    models = build_models(depth)
    SPECS.mkdir(parents=True, exist_ok=True)
    rows = []
    for prec, path in models.items():
        heads, lg_test, lg_pool = {}, {}, {}
        for r in RESOLUTIONS:
            Fp, Ft = model_feats(prec, path, r, "pool", pool), model_feats(prec, path, r, "test", test)
            heads[r] = fit_heads(Fp, py, f"{path.stem}_r{r}")
            for kind in ("aligned", "probe"):
                lg_test[r, kind], lg_pool[r, kind] = head_logits(heads[r][kind], Ft), head_logits(heads[r][kind], Fp)
        size_mb = path.stat().st_size / 2**20
        for kind in ("aligned", "probe"):
            for r in RESOLUTIONS:
                name = f"{prec}_d{depth}_{kind}_r{r}"
                acc = float((lg_test[r, kind].argmax(1) == ty).mean())
                spec = {"name": name, "onnx": f"models/{path.name}", "stages": [r],
                        "heads": {str(r): heads[r][kind]}, "precision": prec, "head_kind": kind}
                (SPECS / f"{name}.json").write_text(json.dumps(spec))
                rows.append({**spec, "acc": acc, "frac_second": 0.0, "model_file_mb": size_mb})
            for r1, r2 in CASCADES:
                m_pool = margin(lg_pool[r1, kind])
                m_test = margin(lg_test[r1, kind])
                for f in CASCADE_FRACS:
                    thr = float(np.quantile(m_pool, f))
                    esc = m_test < thr
                    pred = np.where(esc, lg_test[r2, kind].argmax(1), lg_test[r1, kind].argmax(1))
                    name = f"{prec}_d{depth}_{kind}_r{r1}-{r2}_f{int(f * 100)}"
                    spec = {"name": name, "onnx": f"models/{path.name}", "stages": [r1, r2], "threshold": thr,
                            "heads": {str(r1): heads[r1][kind], str(r2): heads[r2][kind]},
                            "precision": prec, "head_kind": kind}
                    (SPECS / f"{name}.json").write_text(json.dumps(spec))
                    rows.append({**spec, "acc": float((pred == ty).mean()), "frac_second": float(esc.mean()),
                                 "model_file_mb": size_mb})
        print(f"{prec}: " + ", ".join(f"{r['name'].split('_d12_')[1]}={r['acc']:.4f}" for r in rows
                                       if r["precision"] == prec and "-" not in r["name"]), flush=True)
    import pandas as pd
    out = pd.DataFrame(rows).drop(columns=["heads"])
    (LEAN_ART.parent.parent / "results" / "lean").mkdir(parents=True, exist_ok=True)
    out.to_csv(LEAN_ART.parent.parent / "results" / "lean" / "accuracy.csv", index=False)
    return out


if __name__ == "__main__":
    main()
