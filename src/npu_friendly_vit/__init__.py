"""Conv-free, NPU-friendly Vision Transformer for AMD Ryzen AI (Hawk Point / XDNA1)."""

from npu_friendly_vit.model import (
    COEFS,
    KNOTS,
    SHIFT,
    XINT8_LISTED,
    Z_MIN,
    Attention,
    Block,
    BNLast,
    NPUViT,
    PWLSoftmax,
    ReLUAttnNorm,
    audit_onnx,
    export_onnx,
    fold_batchnorms,
    make_attn_norm,
    patchify,
    patchify_np,
)

__version__ = "0.1.0"

__all__ = [
    "Attention",
    "BNLast",
    "Block",
    "COEFS",
    "KNOTS",
    "NPUViT",
    "PWLSoftmax",
    "ReLUAttnNorm",
    "SHIFT",
    "XINT8_LISTED",
    "Z_MIN",
    "__version__",
    "audit_onnx",
    "export_onnx",
    "fold_batchnorms",
    "make_attn_norm",
    "patchify",
    "patchify_np",
]
