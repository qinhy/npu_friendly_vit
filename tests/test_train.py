from npu_friendly_vit.train import main as train_main


def test_train_fake_data_smoke(tmp_path):
    out_dir = tmp_path / "out"
    train_main(
        [
            "--attn",
            "pwl",
            "--epochs",
            "1",
            "--batch",
            "128",
            "--workers",
            "0",
            "--out",
            str(out_dir),
            "--fake-data",
        ]
    )
    assert (out_dir / "npu_vit_pwl.pt").exists()
    assert (out_dir / "npu_vit_pwl.onnx").exists()
    assert (out_dir / "calib.npy").exists()
    assert (out_dir / "test_x.npy").exists()
    assert (out_dir / "test_y.npy").exists()
