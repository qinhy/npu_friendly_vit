import numpy as np
import onnxruntime as ort
import pytest
import torch

from npu_friendly_vit import NPUViT, export_onnx, patchify, patchify_np
from npu_friendly_vit.quantize import NpyReader, accuracy, latency_ms, main, quantize_ort


def test_npy_reader_rewind():
    calib = np.zeros((3, 64, 48), dtype=np.float32)
    reader = NpyReader(calib)
    batches = []
    while (item := reader.get_next()) is not None:
        batches.append(item["input"])
    assert len(batches) == 3
    assert batches[0].shape == (1, 64, 48)
    reader.rewind()
    assert reader.get_next() is not None


def test_quantize_ort_pipeline(tmp_path):
    torch.manual_seed(0)
    model = NPUViT(dim=64, depth=1, heads=4, attn="pwl")
    model.train()
    model(patchify(torch.randn(8, 3, 32, 32)))
    model.eval()

    src = tmp_path / "model.onnx"
    dst = tmp_path / "model_ortint8.onnx"
    export_onnx(model, src)

    calib = patchify_np(np.random.randn(8, 3, 32, 32).astype(np.float32))
    test_x = patchify_np(np.random.randn(10, 3, 32, 32).astype(np.float32))
    test_y = np.random.randint(0, 10, size=(10,), dtype=np.int64)

    quantize_ort(str(src), str(dst), calib)
    assert dst.exists()

    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    acc = accuracy(sess, test_x, test_y, limit=10)
    assert 0.0 <= acc <= 1.0
    ms = latency_ms(sess, test_x, runs=5)
    assert ms >= 0.0


def test_quantize_cli_ort(tmp_path):
    torch.manual_seed(0)
    model = NPUViT(dim=64, depth=1, heads=4, attn="pwl")
    model.train()
    model(patchify(torch.randn(4, 3, 32, 32)))
    model.eval()

    src = tmp_path / "model.onnx"
    export_onnx(model, src)
    np.save(tmp_path / "calib.npy", patchify_np(np.random.randn(4, 3, 32, 32).astype(np.float32)))
    np.save(tmp_path / "test_x.npy", patchify_np(np.random.randn(4, 3, 32, 32).astype(np.float32)))
    np.save(tmp_path / "test_y.npy", np.zeros(4, dtype=np.int64))

    args = ["--model", str(src), "--data", str(tmp_path), "--quantizer", "ort", "--eval-n", "4"]
    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        with pytest.raises(SystemExit):
            main(args)
    else:
        main(args)
    assert (tmp_path / "model_ortint8.onnx").exists()
