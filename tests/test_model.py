"""RMSNorm/QK-Norm conversion, calibration, folding and ONNX tests."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from npu_friendly_vit import (
    TokenRMSNorm, MatMulRMSNorm, HeadRMSNorm, MatMulHeadRMSNorm,
    DynamicTanh, DynamicHardTanh,
    MatMulLayerNorm, StandardViT, NPUViT, CalibratedReLU, FixedPWLSoftmax, LearnedMLPSoftmax, fit_mlp_softmax,
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


def test_harddyt_training_blend():
    layer = DynamicHardTanh(3, hard_fraction=0.5)
    x = torch.tensor([-4.0, 1.0, 2.0], requires_grad=True)
    z = layer.alpha.detach() * x.detach()
    expected = 0.5 * torch.tanh(z) + 0.5 * z.clamp(-1, 1)
    torch.testing.assert_close(layer(x), expected)
    layer(x).sum().backward()
    assert x.grad[0] > 0  # Tanh still carries a gradient outside the clip range.
    layer.set_hard_fraction(0)
    torch.testing.assert_close(layer(x), torch.tanh(z))
    layer.set_hard_fraction(1)
    torch.testing.assert_close(layer(x), z.clamp(-1, 1))


@pytest.mark.parametrize("kind, norm_class, activation", [
    ("dyt", DynamicTanh, torch.tanh),
    ("harddyt", DynamicHardTanh, lambda x: x.clamp(-1, 1)),
])
def test_pointwise_norm_transfer_and_folding(kind, norm_class, activation):
    torch.manual_seed(32)
    teacher = StandardViT(dim=16, depth=1, heads=2, patch=8,
                          norm_type=kind, qk_norm_type=kind).eval()
    norms = [norm for norm in teacher.modules() if type(norm) is norm_class]
    assert len(norms) == 5  # two block sites, Q, K, and final norm
    with torch.no_grad():
        for norm in norms:
            norm.weight.uniform_(-1, 2)
            norm.bias.uniform_(-0.5, 0.5)
            if norm.heads is None:
                norm.alpha.fill_(0.9)
            else:
                norm.alpha.copy_(torch.tensor([0.75, -1.25]))
    q_norm = teacher.blocks[0].attn.q_norm
    x = torch.randn(3, 2, 5, 8) * 2
    y = q_norm(x)
    expected = activation(x * q_norm.alpha[:, None, None])
    expected = expected * q_norm.weight[:, None, :] + q_norm.bias[:, None, :]
    torch.testing.assert_close(y, expected)
    y.square().mean().backward()
    assert (q_norm.alpha.grad.abs() > 0).all()

    calib = calibration()
    patches = patchify(calib[0], 8)
    student = fit_static_norms(teacher, calib)
    assert student.norm_mode == "pointwise"
    with torch.no_grad():
        torch.testing.assert_close(student(patches), teacher(patches))
        calibrate_npu(student, calib, softmax_kind="native", verbose=False)
        folded = fold_affine_norms(student)
        torch.testing.assert_close(folded(patches), student(patches), atol=2e-6, rtol=1e-5)
    assert folded.blocks[0].attn.q_norm.alpha is None
    assert folded.blocks[0].attn.k_norm.alpha is None
    assert folded.blocks[0].norm1.weight is None
    assert folded.norm.weight is None
    assert student.blocks[0].attn.q_norm.alpha is not None


def test_harddyt_export_operators_and_ort_parity(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    teacher = StandardViT(dim=16, depth=1, heads=2, patch=8,
                          norm_type="harddyt", qk_norm_type="harddyt").eval()
    calib = calibration()
    student = convert_to_npu(teacher, calib, softmax_kind="mlp", mlp_fit_steps=0, verbose=False)
    path = tmp_path / "harddyt.onnx"
    export_onnx(student, path)
    ops = audit_onnx(path)
    assert ops["Clip"] == 5
    assert not ({"Tanh", "Sqrt", "Rsqrt", "ReduceMean", "ReduceMax"} & ops.keys())
    patches = patchify(calib[0][:1], 8)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = student(patches).numpy()
    np.testing.assert_allclose(sess.run(None, {"input": patches.numpy()})[0],
                               expected, atol=4e-4, rtol=4e-4)


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
    student = convert_to_npu(teacher, calib, softmax_kind="pwl", softmax_segments=12,
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
    calibrate_npu(loaded, calib, softmax_kind="pwl", softmax_segments=12, verbose=False)


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
    old_cfg.pop("qk_norm_type")
    torch.save({"config": old_cfg, "model": teacher.state_dict()}, ckpt)
    legacy = load_teacher(ckpt, "cpu")
    assert legacy.norm_type == "ln"
    assert legacy.qk_norm is False


def test_no_high_level_norm_or_softmax_in_trace():
    teacher = tiny_teacher()
    student = convert_to_npu(teacher, calibration(), softmax_kind="pwl", verbose=False)
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
    student = convert_to_npu(teacher, calibration(), softmax_kind="pwl", verbose=False)
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


def test_learned_mlp6_valid_and_fitting_improves_kl():
    from npu_friendly_vit.model import _row_kl
    torch.manual_seed(0)
    module = LearnedMLPSoftmax(
        32, torch.full((4,), -6.0), torch.full((4,), 6.0), hidden=6,
        score_center="mean", knot_spacing="geometric",
    )
    assert sum(p.numel() for p in module.parameters()) == 4 * (2 * 6 + 1)
    scores = torch.randn(512, 4, 32) * 1.8
    stats = fit_mlp_softmax(module, scores, steps=100, lr=0.0003)
    assert stats["best_kl"] <= stats["initial_kl"] + 1e-7
    # Parameters should respond to an optimization objective, not just be a
    # relabeled fixed PWL function.
    module.train()
    logits = torch.randn(2, 4, 8, 32)
    probs = module(logits)
    assert torch.isfinite(probs).all()
    assert (probs > 0).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, 4, 8), atol=1e-5, rtol=0)
    torch.testing.assert_close(probs, module(logits + 2), atol=1e-5, rtol=1e-4)
    _row_kl(module, scores[:16]).backward()
    assert any(p.grad is not None for p in module.parameters())
    module.freeze()
    assert not any(p.requires_grad for p in module.parameters())


def test_learned_mlp_conversion_reload_fold_trace(tmp_path):
    teacher = tiny_teacher()
    calib = calibration()
    student = convert_to_npu(teacher, calib, softmax_kind="mlp", mlp_hidden=6,
                             mlp_fit_steps=10, mlp_rows=64, verbose=False)
    assert isinstance(student.blocks[0].attn.norm, LearnedMLPSoftmax)
    x = patchify(calib[0], 8)
    with torch.no_grad():
        result = student(x)
        folded = fold_affine_norms(student)
        torch.testing.assert_close(folded(x), result, atol=3e-6, rtol=2e-5)
        traced = torch.jit.trace(folded, x[:1])
    ops = set(torch.jit.export_opnames(traced))
    assert not any("softmax" in op or "gelu" in op or "exp" in op or "layer_norm" in op
                   or "reduce_max" in op for op in ops)
    path = tmp_path / "mlp.pt"
    save_student(student, path)
    reloaded = load_student(path, "cpu")
    assert reloaded.blocks[0].attn.norm.hidden == 6
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), result, atol=2e-6, rtol=1e-5)
    assert len(measure_saturation(reloaded, calib)) == 2


def test_native_softmax_baseline_save_reload(tmp_path):
    teacher = tiny_teacher()
    student = convert_to_npu(teacher, calibration(), softmax_kind="native", verbose=False)
    assert isinstance(student.blocks[0].attn.norm, nn.Softmax)
    path = tmp_path / "native.pt"
    save_student(student, path)
    reloaded = load_student(path, "cpu")
    assert isinstance(reloaded.blocks[0].attn.norm, nn.Softmax)
    x = patchify(calibration()[0], 8)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), student(x), atol=2e-6, rtol=1e-5)
    assert all(v["attention_per_head"] is None for v in measure_saturation(reloaded, calibration()).values())


def test_learned_mlp_onnx_no_exp_and_ort_parity(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    teacher = tiny_teacher()
    student = convert_to_npu(teacher, calibration(), softmax_kind="mlp",
                             mlp_fit_steps=8, mlp_rows=64, verbose=False)
    path = tmp_path / "mlp6.onnx"
    export_onnx(student, path)
    ops = audit_onnx(path)
    assert not ({"Softmax", "Exp", "Gelu", "Erf", "LayerNormalization", "ReduceMax"} & set(ops))
    assert "Relu" in ops
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = patchify(calibration()[0][:1], 8)
    with torch.no_grad():
        torch_out = student(x).numpy()
    np.testing.assert_allclose(sess.run(None, {"input": x.numpy()})[0],
                               torch_out, atol=5e-4, rtol=5e-4)


def _legacy_hinge_reference(module, scores):
    """Reference for the previous sequential implementation, used only in tests."""
    z = module.center_scores(scores)
    z = torch.minimum(torch.maximum(z, module.low), module.high)
    y = module.base
    for i in range(module.hidden if isinstance(module, LearnedMLPSoftmax) else module.segments):
        bias = module.biases[..., i:i + 1]
        coef = module.coefs[:, :, i:i + 1, :]
        y = y + coef * torch.relu(z + bias)
    if isinstance(module, LearnedMLPSoftmax):
        y = torch.maximum(y, module.min_value)
    else:
        y = torch.maximum(y, module.base * 0.5)
    return y / (y @ module.ones)


@pytest.mark.parametrize("kind", ["mlp", "pwl"])
def test_vectorized_hinges_match_sequential_reference_and_no_loop_in_trace(kind):
    torch.manual_seed(56)
    low = torch.full((4,), -6.0)
    high = torch.full((4,), 6.0)
    m = (LearnedMLPSoftmax(16, low, high, hidden=6) if kind == "mlp"
         else FixedPWLSoftmax(16, low, high, segments=12))
    scores = torch.randn(2, 4, 16, 16) * 1.2
    torch.testing.assert_close(m(scores), _legacy_hinge_reference(m, scores), atol=3e-6, rtol=2e-5)
    with torch.no_grad():
        traced = torch.jit.trace(m.eval(), scores)
    # There must be one vectorized hidden-layer ReLU, not one for each neuron.
    ops = set(torch.jit.export_opnames(traced))
    assert "aten::relu" in ops
    assert "aten::matmul" in ops
    assert not any("softmax" in op or "exp" in op for op in ops)


def test_vectorized_backward_gradients_match_reference():
    m = LearnedMLPSoftmax(16, torch.full((4,), -6.0), torch.full((4,), 6.0), hidden=6)
    scores = torch.randn(2, 4, 16, 16) * 1.8
    ref = _legacy_hinge_reference(m, scores).square().sum()
    ref_grads = torch.autograd.grad(ref, (m.base, m.biases, m.coefs))
    new = m(scores).square().sum()
    new_grads = torch.autograd.grad(new, (m.base, m.biases, m.coefs))
    for a, b in zip(ref_grads, new_grads):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=4e-4)


@pytest.mark.parametrize("kind", ["mlp", "pwl"])
def test_old_per_neuron_weights_upgrade_to_vectorized_state_dict(kind):
    low, high = torch.full((4,), -6.0), torch.full((4,), 6.0)
    m = (LearnedMLPSoftmax(16, low, high, hidden=6) if kind == "mlp"
         else FixedPWLSoftmax(16, low, high, segments=12))
    legacy = {k: v.clone() for k, v in m.state_dict().items()}
    packed_bias = legacy.pop("biases")
    packed_coef = legacy.pop("coefs")
    n = m.hidden if kind == "mlp" else m.segments
    for i in range(n):
        legacy[(f"biases.{i}" if kind == "mlp" else f"bias_{i}")] = packed_bias[..., i:i + 1].clone()
        legacy[(f"coefs.{i}" if kind == "mlp" else f"coef_{i}")] = packed_coef[:, :, i:i + 1, :].clone()
    restored = type(m)(16, low, high, **({"hidden": 6} if kind == "mlp" else {"segments": 12}))
    restored.load_state_dict(legacy, strict=True)
    x = torch.randn(2, 4, 16, 16)
    torch.testing.assert_close(restored(x), m(x), atol=1e-7, rtol=1e-6)
