"""Exercise the CLI stages without needing CIFAR downloads or an NPU."""

from npu_friendly_vit.train import load_student, load_teacher, main


def test_train_adapt_export_smoke(tmp_path):
    folder = tmp_path / "out"
    shared = [
        "--fake-data", "--fake-samples", "48", "--batch", "12", "--workers", "0",
        "--patch", "8", "--depth", "1", "--dim", "32", "--heads", "4",
        "--warmup", "0", "--calib-samples", "24", "--calib-batch", "8",
        "--val-samples", "8", "--out", str(folder),
    ]
    main(["--stage", "train", "--epochs", "1", *shared])
    assert (folder / "vit_teacher.pt").exists()
    teacher = load_teacher(folder / "vit_teacher.pt", "cpu")
    assert teacher.n_tokens == 16
    assert teacher.norm_type == "rms"
    assert teacher.qk_norm is True

    main(["--stage", "adapt", "--finetune-epochs", "1", "--softmax-segments", "8", *shared])
    assert (folder / "vit_npu.pt").exists()
    assert (folder / "calib.npy").exists()
    assert (folder / "test_x.npy").exists()
    assert (folder / "test_y.npy").exists()
    student = load_student(folder / "vit_npu.pt", "cpu")
    assert student.n_tokens == 16
    assert student.norm_type == "rms"
    assert student.qk_norm is True
    assert student.blocks[0].attn.norm.score_center == "mean"

    main(["--stage", "export", "--out", str(folder)])
    try:
        import onnx  # noqa: F401
    except ImportError:
        assert not (folder / "vit_npu.onnx").exists()
    else:
        assert (folder / "vit_npu.onnx").exists()
