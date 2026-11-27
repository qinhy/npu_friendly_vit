"""Vision Transformer training and calibrated NPU export."""
from .model import (
    StandardViT, NPUViT, TokenRMSNorm, MatMulRMSNorm,
    HeadRMSNorm, MatMulHeadRMSNorm, MatMulLayerNorm,
    DynamicTanh, DynamicHardTanh,
    StaticChannelNorm, CalibratedReLU, FixedPWLSoftmax, LearnedMLPSoftmax, fit_mlp_softmax,
    patchify, patchify_np, fit_static_norms, calibrate_npu, convert_to_npu,
    measure_saturation, fold_affine_norms, export_onnx, audit_onnx,
)

__all__ = [
    "StandardViT", "NPUViT", "TokenRMSNorm", "MatMulRMSNorm", "HeadRMSNorm", "MatMulHeadRMSNorm",
    "DynamicTanh", "DynamicHardTanh",
    "MatMulLayerNorm", "StaticChannelNorm", "CalibratedReLU", "FixedPWLSoftmax", "LearnedMLPSoftmax", "fit_mlp_softmax",
    "patchify", "patchify_np", "fit_static_norms", "calibrate_npu", "convert_to_npu",
    "measure_saturation", "fold_affine_norms", "export_onnx", "audit_onnx",
]
