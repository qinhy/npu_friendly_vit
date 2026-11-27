"""Exercise the CLI stages without needing CIFAR downloads or an NPU."""

import pytest
import torch

from npu_friendly_vit.train import load_student, load_teacher, main


@pytest.mark.parametrize("norm_args, norm_type, qk_type, epochs", [
    ([], "rms", "rms", 1),
    (["--norm-type", "harddyt", "--qk-norm-type", "harddyt"], "harddyt", "harddyt", 1),
    (["--norm-type", "harddyt", "--qk-norm-type", "harddyt",
      "--dyt-hardening-fraction", "0.8"], "harddyt", "harddyt", 3),
    (["--norm-type", "harddyt", "--qk-norm-type", "harddyt",
      "--gelu-relu-ramp-fraction", "0.8"], "harddyt", "harddyt", 3),
])
def test_train_adapt_export_smoke(tmp_path, capsys, norm_args, norm_type, qk_type, epochs):
    folder = tmp_path / "out"
    shared = [
        "--fake-data", "--fake-samples", "48", "--batch", "12", "--workers", "0",
        "--patch", "8", "--depth", "1", "--dim", "32", "--heads", "4",
        "--warmup", "0", "--calib-samples", "24", "--calib-batch", "8",
        "--val-samples", "8", "--out", str(folder),
    ]
    main(["--stage", "train", "--epochs", str(epochs), *norm_args, *shared])
    assert (folder / "vit_teacher.pt").exists()
    teacher = load_teacher(folder / "vit_teacher.pt", "cpu")
    assert teacher.n_tokens == 16
    assert teacher.norm_type == norm_type
    assert teacher.qk_norm is True
    assert teacher.qk_norm_type == qk_type
    use_plain_relu = "--gelu-relu-ramp-fraction" in norm_args
    assert teacher.ffn_activation == ("relu" if use_plain_relu else "gelu")
    log = capsys.readouterr().out
    if "--dyt-hardening-fraction" in norm_args:
        assert "best=pending hard_fraction=0.3000" in log
        assert "best=pending hard_fraction=0.7000" in log
        assert "hard_fraction=1.0000" in log
        checkpoint = torch.load(folder / "vit_teacher.pt", weights_only=True)
        assert checkpoint["epoch"] == epochs
        assert teacher.blocks[0].attn.q_norm.hard_fraction == 1.0
    if use_plain_relu:
        assert "best=pending relu_fraction=0.3000" in log
        assert "best=pending relu_fraction=0.7000" in log
        assert "relu_fraction=1.0000" in log
        checkpoint = torch.load(folder / "vit_teacher.pt", weights_only=True)
        assert checkpoint["epoch"] == epochs
        assert isinstance(teacher.blocks[0].act, torch.nn.ReLU)

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
    assert student.norm_type == norm_type
    assert student.qk_norm is True
    assert student.qk_norm_type == qk_type
    assert student.blocks[0].attn.norm.score_center == "mean"
    assert student.blocks[0].attn.norm.hidden == 6
    if use_plain_relu:
        log = capsys.readouterr().out
        assert "plain ReLU" in log
        assert "ReLU gain" not in log
        assert isinstance(student.blocks[0].act, torch.nn.ReLU)
        assert not ({"relu_gain", "relu_bias"} & first["pwl_specs"][0].keys())

    if norm_type == "rms":
        # Older RMS checkpoints omit the Q/K type and FFN activation.
        first["config"].pop("qk_norm_type")
        first["config"].pop("ffn_activation")
        torch.save(first, folder / "vit_npu_latest.pt")

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
        import onnx
    except ImportError:
        assert not (folder / "vit_npu.onnx").exists()
    else:
        assert (folder / "vit_npu.onnx").exists()
        if norm_type == "harddyt":
            ops = {node.op_type for node in onnx.load(folder / "vit_npu.onnx").graph.node}
            assert "Clip" in ops and "Tanh" not in ops
            if use_plain_relu:
                assert "Relu" in ops and not ({"Gelu", "Erf"} & ops)


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
