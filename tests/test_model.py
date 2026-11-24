"""RMSNorm/QK-Norm conversion, calibration, folding and ONNX tests."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from npu_friendly_vit import (
    TokenRMSNorm, MatMulRMSNorm, HeadRMSNorm, MatMulHeadRMSNorm,
    MatMulLayerNorm, StandardViT, NPUViT, CalibratedReLU, FixedPWLSoftmax,
    fit_static_norms, convert_to_npu, calibrate_npu, fold_affine_norms,
    export_onnx, audit_onnx, measure_saturation, patchify, patchify_np,
)
from npu_friendly_vit.train import save_student, load_student, load_teacher


def tiny_teacher(norm_type="rms", qk_norm=True):
    torch.manual_seed(13)
    return StandardViT(dim=32, depth=2, heads=4, patch=8,
                       norm_type=norm_type, qk_norm=qk_norm).eval()


def calibration():
    torch.manual_seed(7)
    return [torch.randn(4, 3, 32, 32), torch.randn(3, 3, 32, 32)]


def test_patchify():
    x = torch.randn(4, 3, 32, 32)
    assert patchify(x, 4).shape == (4, 64, 48)
    np.testing.assert_allclose(patchify(x).numpy(), patchify_np(x.numpy()))
    with pytest.raises(ValueError):
        patchify(x, 5)


def test_rmsnorm_math_matches_matmul():
    torch.manual_seed(17)
    a, b = TokenRMSNorm(32), MatMulRMSNorm(32)
    with torch.no_grad():
        a.weight.uniform_(0.5, 1.7)
        b.load_state_dict(a.state_dict())
    x = torch.randn(3, 12, 32)
    torch.testing.assert_close(a(x), b(x), atol=1e-6, rtol=1e-5)
    assert b.avg.shape == (32, 1)


def test_qk_norm_math_matches_matmul_and_stable():
    torch.manual_seed(11)
    a, b = HeadRMSNorm(4, 8), MatMulHeadRMSNorm(4, 8)
    with torch.no_grad():
        a.weight.uniform_(0.5, 1.3)
        b.load_state_dict(a.state_dict())
    x = torch.randn(2, 4, 16, 8)
    torch.testing.assert_close(a(x), b(x), atol=1e-6, rtol=1e-5)
    assert a(torch.zeros_like(x)).abs().max() == 0
    assert torch.isfinite(b(torch.zeros_like(x))).all()


def test_teacher_architecture_and_qk_gradient():
    model = tiny_teacher()
    block = model.blocks[0]
    assert isinstance(block.norm1, TokenRMSNorm)
    assert isinstance(block.attn.q_norm, HeadRMSNorm)
    assert isinstance(block.attn.k_norm, HeadRMSNorm)
    assert isinstance(block.act, nn.GELU)
    assert isinstance(block.attn.norm, nn.Softmax)
    x = patchify(torch.randn(2, 3, 32, 32), 8)
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(block.attn.q_norm.weight.grad).all()
    assert torch.isfinite(block.attn.k_norm.weight.grad).all()


def test_norm_and_qk_conversion_no_calibration_error():
    teacher = tiny_teacher()
    student = fit_static_norms(teacher, calibration())
    assert isinstance(student, NPUViT)
    assert isinstance(student.norm, MatMulRMSNorm)
    assert isinstance(student.blocks[0].attn.q_norm, MatMulHeadRMSNorm)
    assert isinstance(student.blocks[0].attn.k_norm, MatMulHeadRMSNorm)
    x = patchify(torch.randn(2, 3, 32, 32), 8)
    with torch.no_grad():
        torch.testing.assert_close(student(x), teacher(x), atol=2e-6, rtol=1e-5)
    # Initial teacher->student RMSNorm/QK-Norm conversion must not require fitting.
    assert student.norm_mode == "rms_matmul"


def test_pwl_no_row_max_mean_center_and_geometric_knots():
    low = torch.full((4,), -9.0)
    high = torch.full((4,), 9.0)
    module = FixedPWLSoftmax(16, low, high, 12,
                             score_center="mean", knot_spacing="geometric")
    scores = torch.randn(3, 4, 16, 16) * 2
    y = module(scores)
    assert torch.isfinite(y).all()
    assert (y >= 0).all()
    torch.testing.assert_close(y.sum(-1), torch.ones(3, 4, 16), atol=2e-5, rtol=0)
    assert all(not p.requires_grad for p in module.parameters())
    # Mean centering preserves an additive common shift unless clipped/numerical noise.
    torch.testing.assert_close(y, module(scores + 3.0), atol=2e-5, rtol=1e-4)


def test_full_conversion_fold_and_checkpoint(tmp_path):
    teacher = tiny_teacher()
    calib = calibration()
    student = convert_to_npu(teacher, calib, softmax_segments=12,
                             score_center="mean", knot_spacing="geometric", verbose=False)
    x = patchify(calib[0], 8)
    assert isinstance(student.blocks[0].act, CalibratedReLU)
    assert isinstance(student.blocks[0].attn.norm, FixedPWLSoftmax)
    assert isinstance(student.blocks[0].norm1, MatMulRMSNorm)
    assert isinstance(student.blocks[0].attn.q_norm, MatMulHeadRMSNorm)
    assert not any(isinstance(m, (nn.LayerNorm, nn.GELU)) for m in student.modules())
    with torch.no_grad():
        y = student(x)
        folded = fold_affine_norms(student)
        assert isinstance(folded.blocks[0].act, nn.ReLU)
        assert isinstance(folded.blocks[0].norm1, MatMulRMSNorm)
        torch.testing.assert_close(folded(x), y, atol=2e-6, rtol=1e-5)
    p = tmp_path / "vit_npu.pt"
    save_student(student, p)
    loaded = load_student(p, "cpu")
    assert loaded.norm_type == "rms"
    assert loaded.qk_norm is True
    assert loaded.norm_mode == "rms_matmul"
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), y, atol=2e-6, rtol=1e-5)
    assert len(measure_saturation(loaded, calib)) == 2
    calibrate_npu(loaded, calib, softmax_segments=12, verbose=False)


def test_legacy_ln_branch_and_checkpoint(tmp_path):
    teacher = tiny_teacher(norm_type="ln", qk_norm=False)
    student = fit_static_norms(teacher, calibration(), norm_mode="matmul")
    assert isinstance(student.blocks[0].norm1, MatMulLayerNorm)
    x = patchify(torch.randn(3, 3, 32, 32), 8)
    with torch.no_grad():
        torch.testing.assert_close(teacher(x), student(x), atol=2e-6, rtol=1e-5)
    # v0.2 checkpoint without norm metadata is treated as legacy LN, never RMS.
    ckpt = tmp_path / "legacy_teacher.pt"
    old_cfg = dict(teacher.config)
    old_cfg.pop("norm_type")
    old_cfg.pop("qk_norm")
    torch.save({"config": old_cfg, "model": teacher.state_dict()}, ckpt)
    legacy = load_teacher(ckpt, "cpu")
    assert legacy.norm_type == "ln"
    assert legacy.qk_norm is False


def test_no_high_level_norm_or_softmax_in_trace():
    teacher = tiny_teacher()
    student = convert_to_npu(teacher, calibration(), verbose=False)
    folded = fold_affine_norms(student).eval()
    example = torch.randn(1, 16, 192)
    with torch.no_grad():
        traced = torch.jit.trace(folded, example)
    ops = set(torch.jit.export_opnames(traced))
    assert not any("softmax" in op or "gelu" in op or "layer_norm" in op
                   or "reduce_max" in op for op in ops)


def test_export_onnx_and_ort_parity(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    teacher = tiny_teacher()
    student = convert_to_npu(teacher, calibration(), verbose=False)
    path = tmp_path / "vit.onnx"
    export_onnx(student, path)
    ops = audit_onnx(path)
    for op in ("Softmax", "LayerNormalization", "BatchNormalization",
               "Exp", "Gelu", "Erf", "ReduceMax"):
        assert op not in ops
    assert "MatMul" in ops
    x = patchify(calibration()[0][:1], 8)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = student(x).numpy()
    np.testing.assert_allclose(sess.run(None, {"input": x.numpy()})[0],
                               expected, atol=4e-4, rtol=4e-4)
