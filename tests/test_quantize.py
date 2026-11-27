"""Portable ORT quantization tests, with NPU disabled explicitly."""

import numpy as np
import pytest
import torch

from npu_friendly_vit import StandardViT, convert_to_npu, export_onnx, patchify_np
from npu_friendly_vit.quantize import NpyReader, accuracy, latency_ms, main, quantize_ort


def test_npy_reader_rewind_and_validation():
    arr = np.zeros((3, 16, 192), dtype=np.float32)
    reader = NpyReader(arr)
    assert reader.get_next()["input"].shape == (1, 16, 192)
    assert reader.get_next() is not None
    assert reader.get_next() is not None
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next() is not None
    with pytest.raises(ValueError):
        NpyReader(np.empty((0, 16, 192), dtype=np.float32))


@pytest.mark.parametrize("norm_type", ["rms", "harddyt"])
def test_quantize_ort_cpu_cli(tmp_path, norm_type):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    teacher = StandardViT(patch=8, dim=32, depth=1, heads=4,
                          norm_type=norm_type, qk_norm_type=norm_type).eval()
    calibration_images = [torch.randn(4, 3, 32, 32)]
    student = convert_to_npu(teacher, calibration_images, softmax_kind="mlp", mlp_fit_steps=5, mlp_rows=64, verbose=False)
    src = tmp_path / "vit_npu.onnx"
    export_onnx(student, src)
    rng = np.random.default_rng(7)
    calib = patchify_np(rng.normal(size=(8, 3, 32, 32)).astype(np.float32), p=8)
    test_x = patchify_np(rng.normal(size=(4, 3, 32, 32)).astype(np.float32), p=8)
    test_y = np.zeros(4, dtype=np.int64)
    np.save(tmp_path / "calib.npy", calib)
    np.save(tmp_path / "test_x.npy", test_x)
    np.save(tmp_path / "test_y.npy", test_y)
    dst = tmp_path / "vit_npu_ortint8.onnx"
    quantize_ort(src, dst, calib)
    assert dst.exists()
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    assert 0.0 <= accuracy(sess, test_x, test_y, 4) <= 1.0
    assert latency_ms(sess, test_x, runs=3) >= 0

    # CPU-only execution should be successful, even without a VitisAIExecutionProvider.
    report = main(["--model", str(src), "--data", str(tmp_path), "--quantizer", "ort",
                   "--npu", "off", "--eval-n", "4", "--latency-runs", "3"])
    assert report["cpu_int8_accuracy"] >= 0
    assert report["npu"] is None
