"""ONNX export of the CLIP image encoder and ONNX Runtime static INT8 PTQ."""
import numpy as np
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from .common import MODEL_DIR, ImageEncoder, load_clip

FP32_ONNX = MODEL_DIR / "image_encoder_fp32.onnx"
PREP_ONNX = MODEL_DIR / "image_encoder_fp32_prep.onnx"


def export_fp32(force: bool = False):
    if FP32_ONNX.exists() and not force:
        return FP32_ONNX
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    enc = ImageEncoder(load_clip()).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(enc, (dummy,), str(FP32_ONNX), input_names=["pixel_values"],
                      output_names=["image_embeds"], dynamic_axes={"pixel_values": {0: "batch"},
                                                                   "image_embeds": {0: "batch"}},
                      opset_version=17, dynamo=False)
    # Shape inference + graph optimization recommended by ORT before quantization.
    quant_pre_process(str(FP32_ONNX), str(PREP_ONNX), skip_symbolic_shape=False)
    return FP32_ONNX


class ArrayReader(CalibrationDataReader):
    def __init__(self, images: np.ndarray, batch_size: int = 16):
        self.batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
        self.it = iter(self.batches)

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"pixel_values": b}

    def rewind(self):
        self.it = iter(self.batches)


def quantize(calib_images: np.ndarray, out_path, method=CalibrationMethod.MinMax):
    """Static INT8 QDQ quantization: per-channel symmetric INT8 weights, UINT8 activations.

    Only MatMul/Gemm/Conv are quantized; LayerNorm/Softmax/GELU stay FP32 (standard practice for
    ViTs, where those ops are the most quantization-sensitive). Activation ranges for the quantized
    ops' inputs come from the calibration images.
    """
    export_fp32()
    quantize_static(
        str(PREP_ONNX), str(out_path), ArrayReader(calib_images),
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        per_channel=True, calibrate_method=method, op_types_to_quantize=["MatMul", "Gemm", "Conv"],
        # Do NOT set CalibMaxIntermediateOutputs: in ORT 1.31 MinMaxCalibrator.collect_data clears
        # buffered batches at the limit without merging them, silently dropping calibration data.
        # Not needed anyway: the augmented model only emits per-tensor ReduceMin/ReduceMax scalars.
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    return out_path
