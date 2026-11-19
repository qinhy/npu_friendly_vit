import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

from npu_friendly_vit import (
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


def test_patchify_matches_numpy():
    imgs = torch.randn(4, 3, 32, 32)
    pt = patchify(imgs, p=4)
    np_out = patchify_np(imgs.numpy(), p=4)
    assert pt.shape == (4, 64, 48)
    assert np_out.shape == (4, 64, 48)
    assert np.allclose(pt.numpy(), np_out)


def test_pwl_softmax_approximates_softmax():
    torch.manual_seed(0)
    s = torch.randn(256, 64) * 2.0
    pwl = PWLSoftmax(64)(s)
    exact = torch.softmax(s, dim=-1)
    assert torch.allclose(pwl.sum(dim=-1), torch.ones(256), atol=1e-5)
    assert (pwl - exact).abs().max() < 0.06
    assert (pwl - exact).abs().mean() < 0.002


def test_make_attn_norm_variants():
    assert isinstance(make_attn_norm("pwl", 16), PWLSoftmax)
    pwl_learn = make_attn_norm("pwl_learn", 16)
    assert isinstance(pwl_learn, PWLSoftmax)
    assert isinstance(pwl_learn.bias, nn.Parameter)
    assert isinstance(make_attn_norm("relu", 16), ReLUAttnNorm)
    assert isinstance(make_attn_norm("softmax", 16), nn.Softmax)
    with pytest.raises(ValueError):
        make_attn_norm("unknown", 16)


@pytest.mark.parametrize("attn", ["pwl", "pwl_learn", "relu", "softmax"])
def test_npu_vit_forward_shapes(attn):
    model = NPUViT(img_size=32, patch=4, num_classes=10, dim=64, depth=2, heads=4, attn=attn)
    patches = patchify(torch.randn(2, 3, 32, 32))
    logits = model(patches)
    assert logits.shape == (2, 10)


def test_fold_batchnorms_exact_equivalence():
    torch.manual_seed(0)
    model = NPUViT(dim=64, depth=2, heads=4)
    model.train()
    for _ in range(3):
        model(patchify(torch.randn(8, 3, 32, 32)))
    model.eval()

    folded = fold_batchnorms(model)
    assert isinstance(folded.patch_bn, nn.Identity)
    assert isinstance(folded.norm, nn.Identity)
    for blk in folded.blocks:
        assert isinstance(blk.norm1, nn.Identity)
        assert isinstance(blk.norm2, nn.Identity)
        assert isinstance(blk.bn_mid, nn.Identity)

    x = patchify(torch.randn(4, 3, 32, 32))
    diff = (model(x) - folded(x)).abs().max()
    assert diff < 1e-5


def test_export_onnx_audit_and_ort_parity(tmp_path):
    torch.manual_seed(0)
    model = NPUViT(dim=64, depth=2, heads=4, attn="pwl")
    model.train()
    for _ in range(2):
        model(patchify(torch.randn(8, 3, 32, 32)))
    model.eval()

    onnx_path = tmp_path / "model.onnx"
    export_onnx(model, onnx_path)
    assert onnx_path.exists()

    ops = audit_onnx(onnx_path)
    assert "Conv" not in ops
    assert "BatchNormalization" not in ops
    assert "Softmax" not in ops
    assert "MatMul" in ops

    x = patchify(torch.randn(1, 3, 32, 32))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": x.numpy()})[0]
    pt_out = model(x).detach().numpy()
    assert np.allclose(ort_out, pt_out, atol=1e-4)
