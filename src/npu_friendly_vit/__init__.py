"""RMSNorm + QK-Norm teacher -> fixed PWL calibrated NPU-oriented ViT."""
from .model import (
    StandardViT, NPUViT, TokenRMSNorm, MatMulRMSNorm,
    HeadRMSNorm, MatMulHeadRMSNorm, MatMulLayerNorm,
    StaticChannelNorm, CalibratedReLU, FixedPWLSoftmax, LearnedMLPSoftmax, fit_mlp_softmax,
    patchify, patchify_np, fit_static_norms, calibrate_npu, convert_to_npu,
    measure_saturation, fold_affine_norms, export_onnx, audit_onnx,
)

__all__ = [
    "StandardViT", "NPUViT", "TokenRMSNorm", "MatMulRMSNorm", "HeadRMSNorm", "MatMulHeadRMSNorm",
    "MatMulLayerNorm", "StaticChannelNorm", "CalibratedReLU", "FixedPWLSoftmax", "LearnedMLPSoftmax", "fit_mlp_softmax",
    "patchify", "patchify_np", "fit_static_norms", "calibrate_npu", "convert_to_npu",
    "measure_saturation", "fold_affine_norms", "export_onnx", "audit_onnx",
]
