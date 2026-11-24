"""Exercise the CLI stages without needing CIFAR downloads or an NPU."""

import torch

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

    main(["--stage", "adapt", "--finetune-epochs", "1", "--softmax-kind", "mlp", "--mlp-hidden", "6", "--mlp-fit-steps", "5", "--mlp-rows", "64", *shared])
    assert (folder / "vit_npu.pt").exists()
    assert (folder / "vit_npu_best.pt").exists()
    assert (folder / "vit_npu_latest.pt").exists()
    first = torch.load(folder / "vit_npu_latest.pt", weights_only=True)
    assert first["epoch"] == 1
    assert first["best_accuracy"] >= 0
    assert "optimizer" in first and first["optimizer"]["state"]
    assert (folder / "calib.npy").exists()
    assert (folder / "test_x.npy").exists()
    assert (folder / "test_y.npy").exists()
    student = load_student(folder / "vit_npu.pt", "cpu")
    assert student.n_tokens == 16
    assert student.norm_type == "rms"
    assert student.qk_norm is True
    assert student.blocks[0].attn.norm.score_center == "mean"
    assert student.blocks[0].attn.norm.hidden == 6

    # Resuming should complete epoch 2, keeping epoch 1 optimizer state.
    main(["--stage", "adapt", "--resume-finetune",
          "--finetune-epochs", "2", "--softmax-kind", "mlp", "--mlp-hidden", "6", "--mlp-fit-steps", "5", "--mlp-rows", "64", *shared])
    second = torch.load(folder / "vit_npu_latest.pt", weights_only=True)
    assert second["epoch"] == 2
    assert second["best_accuracy"] >= first["best_accuracy"]
    for key, value in first["optimizer"]["state"].items():
        assert second["optimizer"]["state"][key]["step"] > value["step"]
    assert (folder / "vit_npu_best.pt").exists()

    main(["--stage", "export", "--out", str(folder)])
    try:
        import onnx  # noqa: F401
    except ImportError:
        assert not (folder / "vit_npu.onnx").exists()
    else:
        assert (folder / "vit_npu.onnx").exists()


def test_data_only_cli(tmp_path):
    folder = tmp_path / "arrays"
    main(["--stage", "data", "--fake-data", "--fake-samples", "32",
          "--workers", "0", "--calib-samples", "8", "--test-samples", "6",
          "--out", str(folder)])
    import numpy as np
    assert np.load(folder / "calib.npy").shape == (8, 64, 48)
    assert np.load(folder / "test_x.npy").shape[0] == 6
    assert np.load(folder / "test_y.npy").shape == (6,)


def test_legacy_mlp_optimizer_state_migrates_exactly():
    """Old six ParameterList moments load into packed MLP biases/coefs."""
    from npu_friendly_vit.model import NPUViT, LearnedMLPSoftmax
    from npu_friendly_vit.train import _load_optimizer_compat

    model = NPUViT(dim=32, depth=1, heads=4, patch=8)
    model.blocks[0].attn.norm = LearnedMLPSoftmax(
        16, torch.full((4,), -6.0), torch.full((4,), 6.0), hidden=6
    )
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    opt.zero_grad(set_to_none=True)
    model(torch.randn(3, 16, 192)).square().sum().backward()
    opt.step()
    packed = opt.state_dict()
    names = [name for name, _ in model.named_parameters()]
    old_state, old_ids = {}, []
    old_idx = 0
    for idx, name in enumerate(names):
        state = packed["state"].get(idx, {})
        if name.endswith((".attn.norm.biases", ".attn.norm.coefs")):
            for i in range(6):
                old_ids.append(old_idx)
                old_state[old_idx] = {
                    k: (v.clone() if v.ndim == 0 else (
                        v[..., i:i + 1].clone() if name.endswith(".biases")
                        else v[:, :, i:i + 1, :].clone()))
                    for k, v in state.items()
                }
                old_idx += 1
        else:
            old_ids.append(old_idx)
            if state:
                old_state[old_idx] = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                      for k, v in state.items()}
            old_idx += 1
    old_group = dict(packed["param_groups"][0], params=old_ids)
    restored_opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    assert _load_optimizer_compat(restored_opt, {"param_groups": [old_group], "state": old_state}, model)
    got = restored_opt.state_dict()
    for idx, orig in packed["state"].items():
        for name, value in orig.items():
            torch.testing.assert_close(got["state"][idx][name], value)
