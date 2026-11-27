"""CIFAR-10 ViT training and conversion to an NPU-oriented student.

Run from the repo (the package import is preferred):
  uv run npu-vit-train --stage all --epochs 100 --finetune-epochs 10

Or run alongside model.py:
  python train.py --stage all --fake-data --epochs 1 --finetune-epochs 1 \
      --dim 32 --depth 1 --heads 4 --patch 8 --workers 0

Stages:
  train   : train ViT with configurable token and Q/K norms, GELU and Softmax
  adapt   : load teacher, transfer norms, calibrate ReLU/MLP Softmax, distill,
            recalibrate, save NPU checkpoint/ONNX and test arrays
  export  : load already adapted NPU checkpoint and export ONNX
  data    : generate calib.npy/test_x.npy/test_y.npy without training
  all     : train + adapt (default)

Important: final ONNX is still floating point. INT8 quantization is a separate
Ryzen AI / Vitis AI calibration step using the generated calib.npy.
"""

import argparse
import copy
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

if __package__:
    from .model import (
        StandardViT, NPUViT, DynamicHardTanh, CalibratedReLU, FixedPWLSoftmax, LearnedMLPSoftmax,
        convert_to_npu, calibrate_npu, measure_saturation,
        audit_onnx, export_onnx, patchify, patchify_np, resolve_norm_mode,
    )
else:
    from model import (
        StandardViT, NPUViT, DynamicHardTanh, CalibratedReLU, FixedPWLSoftmax, LearnedMLPSoftmax,
        convert_to_npu, calibrate_npu, measure_saturation,
        audit_onnx, export_onnx, patchify, patchify_np, resolve_norm_mode,
    )

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_datasets(fake=False, data_dir="./data", fake_samples=256):
    """Train data with augmentation, separate unaugmented calibration data."""
    if fake:
        g = torch.Generator().manual_seed(123)
        train_x = torch.randn(fake_samples, 3, 32, 32, generator=g)
        train_y = torch.randint(0, 10, (fake_samples,), generator=g)
        test_x = torch.randn(max(32, fake_samples // 4), 3, 32, 32, generator=g)
        test_y = torch.randint(0, 10, (test_x.shape[0],), generator=g)
        train = TensorDataset(train_x, train_y)
        calib = TensorDataset(train_x, train_y)
        test = TensorDataset(test_x, test_y)
        return train, calib, test

    import torchvision
    import torchvision.transforms as T

    norm = T.Normalize(MEAN, STD)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(), T.ToTensor(), norm,
    ])
    clean_tf = T.Compose([T.ToTensor(), norm])
    train = torchvision.datasets.CIFAR10(data_dir, train=True, transform=train_tf, download=True)
    calib = torchvision.datasets.CIFAR10(data_dir, train=True, transform=clean_tf, download=True)
    test = torchvision.datasets.CIFAR10(data_dir, train=False, transform=clean_tf, download=True)
    return train, calib, test


def get_loaders(args):
    train_ds, calib_ds, test_ds = get_datasets(args.fake_data, args.data_dir, args.fake_samples)
    rng = np.random.default_rng(args.seed)
    all_indices = rng.permutation(len(train_ds))
    # Hold out validation data from training; use CIFAR test only for final reporting.
    val_count = min(args.val_samples, max(1, len(train_ds) // 5))
    if val_count >= len(train_ds):
        raise ValueError("Not enough training samples for a validation split")
    val_ids = all_indices[:val_count].tolist()
    train_ids = all_indices[val_count:].tolist()
    calib_ids = rng.choice(train_ids, size=min(args.calib_samples, len(train_ids)), replace=False)
    train_loader = DataLoader(Subset(train_ds, train_ids), args.batch, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(Subset(calib_ds, val_ids), min(256, args.batch * 2),
                            num_workers=args.workers)
    test_loader = DataLoader(test_ds, min(256, args.batch * 2), num_workers=args.workers)
    calib_loader = DataLoader(Subset(calib_ds, calib_ids.tolist()), batch_size=args.calib_batch,
                              shuffle=False, num_workers=args.workers)
    return train_loader, calib_loader, val_loader, test_loader, calib_ds, test_ds, calib_ids


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        logits = model(patchify(images.to(device), model.patch))
        correct += (logits.argmax(-1).cpu() == labels).sum().item()
        total += labels.numel()
    return correct / total


def _schedule(peak_lr, step, total, warmup):
    if warmup and step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * min(1, progress)))


def train_teacher(model, train_dl, val_dl, args, device, ckpt):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(train_dl)
    warm = args.warmup * len(train_dl)
    ramp_steps = min(total - 1, math.ceil(total * args.dyt_hardening_fraction))
    harden_layers = [m for m in model.modules() if isinstance(m, DynamicHardTanh)] if ramp_steps > 0 else []
    hard_fraction = 1.0
    best, step = -1, 0
    for epoch in range(args.epochs):
        model.train()
        loss_sum, start = 0, time.monotonic()
        for images, labels in train_dl:
            if harden_layers and step <= ramp_steps:
                hard_fraction = step / ramp_steps
                for layer in harden_layers:
                    layer.set_hard_fraction(hard_fraction)
            lr = _schedule(args.lr, step, total, warm)
            for group in opt.param_groups:
                group["lr"] = lr
            images, labels = images.to(device), labels.to(device)
            logits = model(patchify(images, model.patch))
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
            step += 1
        accuracy = evaluate(model, val_dl, device)
        # Saved teachers load as pure HardDyT, so select them only at that endpoint.
        if hard_fraction == 1.0 and accuracy > best:
            best = accuracy
            torch.save({"config": model.config, "model": model.state_dict(),
                        "epoch": epoch + 1, "accuracy": best}, ckpt)
        best_label = f"{best:.4%}" if best >= 0 else "pending"
        hardening_label = f" hard_fraction={hard_fraction:.4f}" if harden_layers else ""
        print(f"[FP32] epoch {epoch+1:3d}/{args.epochs} "
              f"loss={loss_sum/len(train_dl):.4f} val_acc={accuracy:.4%} "
              f"best={best_label}{hardening_label} time={time.monotonic()-start:.1f}s", flush=True)
    return best


def _checkpoint_model_config(ckpt):
    # Checkpoints predating RMSNorm used LayerNorm with no Q/K transform.
    cfg = dict(ckpt["config"])
    if "norm_type" not in cfg:
        cfg.update(norm_type="ln", qk_norm=False)
    cfg.setdefault("qk_norm_type", "rms")
    return cfg


def load_teacher(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = _checkpoint_model_config(ckpt)
    teacher = StandardViT(**cfg).to(device)
    teacher.load_state_dict(ckpt["model"])
    return teacher.eval()


def _student_pwl_specs(model):
    """Serialize network topology separately from weights, backward compatible."""
    specs = []
    for b in model.blocks:
        attn = b.attn.norm
        spec = dict(relu_gain=b.act.gain.detach().cpu(),
                    relu_bias=b.act.bias.detach().cpu())
        if isinstance(attn, LearnedMLPSoftmax):
            spec.update(attn_kind="mlp", mlp_hidden=attn.hidden,
                        score_low=attn.low.detach().reshape(-1).cpu(),
                        score_high=attn.high.detach().reshape(-1).cpu(),
                        score_center=attn.score_center,
                        knot_spacing=attn.knot_spacing,
                        mlp_trainable=any(p.requires_grad for p in attn.parameters()))
        elif isinstance(attn, FixedPWLSoftmax):
            spec.update(attn_kind="pwl", softmax_segments=attn.segments,
                        score_low=attn.low.detach().reshape(-1).cpu(),
                        score_high=attn.high.detach().reshape(-1).cpu(),
                        score_center=attn.score_center,
                        knot_spacing=attn.knot_spacing)
        elif isinstance(attn, nn.Softmax):
            spec.update(attn_kind="native")
        else:
            raise TypeError(f"Student attention is not calibrated: {type(attn)}")
        specs.append(spec)
    return specs


def _atomic_torch_save(payload, path):
    """Safely replace a checkpoint at epoch boundaries (also on Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_student(model, path, accuracy=None, *, extra=None):
    """Save a reloadable NPU student; optionally include training resume state."""
    payload = dict(config=model.config, norm_mode=model.norm_mode,
                   model=model.state_dict(),
                   pwl_specs=_student_pwl_specs(model), accuracy=accuracy)
    if extra:
        payload.update(extra)
    _atomic_torch_save(payload, path)


def load_student(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = _checkpoint_model_config(ckpt)
    model = NPUViT(**cfg, norm_mode=ckpt.get("norm_mode", "affine"))
    if len(ckpt["pwl_specs"]) != len(model.blocks):
        raise ValueError("Checkpoint attention specifications have wrong block count")
    for b, spec in zip(model.blocks, ckpt["pwl_specs"]):
        b.act = CalibratedReLU(spec["relu_gain"], spec["relu_bias"])
        kind = spec.get("attn_kind", "pwl")  # legacy fixed-PWL checkpoints
        if kind == "mlp":
            b.attn.norm = LearnedMLPSoftmax(
                model.n_tokens, spec["score_low"], spec["score_high"],
                hidden=spec["mlp_hidden"],
                score_center=spec.get("score_center", "mean"),
                knot_spacing=spec.get("knot_spacing", "geometric"),
            )
            if not spec.get("mlp_trainable", True):
                b.attn.norm.freeze()
        elif kind == "pwl":
            b.attn.norm = FixedPWLSoftmax(
                model.n_tokens, spec["score_low"], spec["score_high"],
                spec["softmax_segments"],
                score_center=spec.get("score_center", "none"),
                knot_spacing=spec.get("knot_spacing", "uniform"),
            )
        elif kind == "native":
            b.attn.norm = nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unknown attention kind in checkpoint: {kind}")
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def _load_optimizer_compat(opt, saved_optimizer, student):
    """Resume older per-neuron-ParameterList AdamW states after vectorization.

    Existing `vit_npu_latest.pt` files contain one parameter and optimizer slot
    per hidden unit. Our vectorized implementation packs them per head into two
    matrices; the Adam first/second moments can be concatenated exactly.
    No changes to the checkpoint are written by this helper.
    """
    new = opt.state_dict()
    if len(new["param_groups"]) != 1 or len(saved_optimizer["param_groups"]) != 1:
        # This training script uses a single group; do not guess multi-group mappings.
        opt.load_state_dict(saved_optimizer)
        return False
    old_ids = saved_optimizer["param_groups"][0]["params"]
    new_ids = new["param_groups"][0]["params"]
    if len(old_ids) == len(new_ids):
        opt.load_state_dict(saved_optimizer)
        return False

    names = [name for name, _ in student.named_parameters()]
    if len(names) != len(new_ids):
        raise ValueError("Current optimizer/model parameter counts do not match")
    old_names = []
    for name in names:
        if name.endswith((".attn.norm.biases", ".attn.norm.coefs")):
            path = name.rsplit(".", 1)[0]
            mod = student.get_submodule(path)
            if isinstance(mod, LearnedMLPSoftmax):
                old_names.extend(f"{name}.{i}" for i in range(mod.hidden))
                continue
        old_names.append(name)
    if len(old_names) != len(old_ids):
        raise ValueError(
            "Cannot restore optimizer: parameter layout does not match the old "
            "or new MLP format. Use vit_npu_best.pt for model-only recovery."
        )

    old_by_name = dict(zip(old_names, old_ids))
    migrated_state = {}
    for name, new_id in zip(names, new_ids):
        if name in old_by_name:
            slot = saved_optimizer["state"].get(old_by_name[name])
            if slot:
                migrated_state[new_id] = copy.deepcopy(slot)
            continue
        path = name.rsplit(".", 1)[0]
        mod = student.get_submodule(path)
        if not isinstance(mod, LearnedMLPSoftmax):
            raise ValueError(f"Cannot migrate optimizer parameter {name}")
        slots = [saved_optimizer["state"].get(old_by_name[f"{name}.{i}"])
                 for i in range(mod.hidden)]
        if all(not slot for slot in slots):
            continue  # Frozen parameters have no optimizer moments.
        if any(not slot for slot in slots):
            raise ValueError(f"Partial legacy Adam state for {name}")
        migrated = {}
        for key in slots[0]:
            vals = [slot[key] for slot in slots]
            if isinstance(vals[0], torch.Tensor) and vals[0].ndim > 0:
                if name.endswith(".biases"):
                    migrated[key] = torch.cat(vals, dim=-1)
                else:
                    migrated[key] = torch.cat(vals, dim=-1).transpose(-1, -2).contiguous()
            else:
                migrated[key] = copy.deepcopy(vals[0])
        migrated_state[new_id] = migrated
    migrated_groups = copy.deepcopy(saved_optimizer["param_groups"])
    migrated_groups[0]["params"] = new_ids
    opt.load_state_dict({"state": migrated_state, "param_groups": migrated_groups})
    return True


def finetune_student(student, teacher, train_dl, val_dl, args, device):
    """Fine-tune with best/latest checkpoints written at every epoch.

    Best: full reloadable student at the highest validation accuracy.
    Latest: full student + AdamW state + epoch/RNG state for --resume-finetune.
    """
    teacher.eval().requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=args.finetune_lr, weight_decay=args.wd)
    checkpoint_dir = Path(args.out)
    best_path = checkpoint_dir / "vit_npu_best.pt"
    latest_path = checkpoint_dir / "vit_npu_latest.pt"
    start_epoch = 0

    if args.resume_finetune:
        if not latest_path.is_file() or not best_path.is_file():
            raise FileNotFoundError(
                "Resume requires vit_npu_latest.pt and vit_npu_best.pt in --out"
            )
        # CUDA RNG byte tensors must be restored from CPU.
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
        if (_checkpoint_model_config(checkpoint) != student.config or
                checkpoint["norm_mode"] != student.norm_mode):
            raise ValueError("Saved fine-tuning checkpoint has an incompatible architecture")
        student.load_state_dict(checkpoint["model"])
        if _load_optimizer_compat(opt, checkpoint["optimizer"], student):
            print("[NPU] Upgraded legacy MLP optimizer moments to packed matrices", flush=True)
        start_epoch = int(checkpoint["epoch"])
        best_acc = float(checkpoint["best_accuracy"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        print(f"[NPU] Resuming at epoch {start_epoch}/{args.finetune_epochs}: "
              f"saved best={best_acc:.4%} (latest={latest_path})", flush=True)
    else:
        best_acc = evaluate(student, val_dl, device)
        print(f"[NPU] Before fine-tuning: accuracy={best_acc:.4%}", flush=True)
        save_student(student, best_path, best_acc)
        save_student(student, latest_path, best_acc, extra={
            "epoch": 0,
            "best_accuracy": best_acc,
            "optimizer": opt.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        })
        print(f"[NPU] Initial checkpoints saved: {best_path}, {latest_path}", flush=True)

    if start_epoch > args.finetune_epochs:
        raise ValueError(
            f"Checkpoint is at epoch {start_epoch}, but --finetune-epochs "
            f"is only {args.finetune_epochs}; set it to a larger total"
        )

    teacher_features, student_features = [], []
    hooks = []
    if args.feature_kd_weight > 0:
        for block in teacher.blocks:
            hooks.append(block.register_forward_hook(
                lambda _mod, _args, output: teacher_features.append(output.detach())))
        for block in student.blocks:
            hooks.append(block.register_forward_hook(
                lambda _mod, _args, output: student_features.append(output)))
    try:
        for epoch in range(start_epoch, args.finetune_epochs):
            student.train()
            running_loss, start = 0.0, time.monotonic()
            for images, labels in train_dl:
                images, labels = images.to(device), labels.to(device)
                patches = patchify(images, student.patch)
                teacher_features.clear()
                student_features.clear()
                with torch.no_grad():
                    teacher_logits = teacher(patches)
                student_logits = student(patches)
                t = args.kd_temperature
                kd = F.kl_div(
                    F.log_softmax(student_logits / t, dim=-1),
                    F.softmax(teacher_logits / t, dim=-1),
                    reduction="batchmean",
                ) * (t * t)
                ce = F.cross_entropy(student_logits, labels, label_smoothing=0.1)
                loss = (1.0 - args.kd_weight) * ce + args.kd_weight * kd
                if args.feature_kd_weight > 0:
                    if len(teacher_features) != len(student_features):
                        raise RuntimeError("Feature distillation layer count mismatch")
                    feat_loss = sum(F.mse_loss(s, t) for s, t in
                                    zip(student_features, teacher_features)) / len(teacher_features)
                    loss = loss + args.feature_kd_weight * feat_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                running_loss += loss.item()

            accuracy = evaluate(student, val_dl, device)
            teacher_features.clear()
            student_features.clear()
            improved = accuracy > best_acc
            if improved:
                best_acc = accuracy
                save_student(student, best_path, best_acc)
            # Save full resume state after *every* epoch; atomic replace prevents
            # half-written checkpoints when an interruption occurs during saving.
            save_student(student, latest_path, accuracy, extra={
                "epoch": epoch + 1,
                "best_accuracy": best_acc,
                "optimizer": opt.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            })
            print(f"[NPU] fine-tune {epoch+1:3d}/{args.finetune_epochs} "
                  f"loss={running_loss/len(train_dl):.4f} "
                  f"val_acc={accuracy:.4%} best={best_acc:.4%} "
                  f"time={time.monotonic()-start:.1f}s "
                  f"saved=latest{'+best' if improved else ''}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    # Final calibration/export starts from the best model, not the last epoch.
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    student.load_state_dict(best_checkpoint["model"])
    return best_acc


def save_quantizer_arrays(args, calib_ds, test_ds, indices, patch):
    """Input-ready patches; calibration arrays contain no augmentations."""
    np.save(Path(args.out) / "calib.npy", patchify_np(
        np.stack([calib_ds[int(i)][0].numpy() for i in indices], axis=0).astype(np.float32), patch))
    n_test = min(args.test_samples, len(test_ds))
    np.save(Path(args.out) / "test_x.npy", patchify_np(
        np.stack([test_ds[i][0].numpy() for i in range(n_test)]).astype(np.float32), patch))
    np.save(Path(args.out) / "test_y.npy", np.asarray(
        [int(test_ds[i][1]) for i in range(n_test)], dtype=np.int64))


def do_export(model, args):
    path = Path(args.out) / "vit_npu.onnx"
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("ONNX export skipped: install 'onnx' (and optionally 'onnxruntime').")
        return
    export_onnx(model.cpu().eval(), path)
    print(f"Exported: {path}")
    audit_onnx(path, allow_native_softmax=isinstance(model.blocks[0].attn.norm, nn.Softmax))
    # Other export failures deliberately propagate: do not silently claim success.


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["all", "train", "adapt", "export", "data"], default="all")
    p.add_argument("--epochs", type=int, default=10, help="standard ViT training epochs")
    p.add_argument("--finetune-epochs", type=int, default=3,
                   help="total fine-tuning epochs, including epochs already completed when resuming")
    p.add_argument("--resume-finetune", action="store_true",
                   help="resume model and AdamW state from --out/vit_npu_latest.pt")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--kd-weight", type=float, default=0.7)
    p.add_argument("--kd-temperature", type=float, default=2.0)
    p.add_argument("--feature-kd-weight", type=float, default=0.2,
                   help="intermediate block-output matching (MSE) during fine-tune")
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out", default="out")
    p.add_argument("--teacher-ckpt", default=None)
    p.add_argument("--student-ckpt", default=None)
    p.add_argument("--val-samples", type=int, default=5000)
    p.add_argument("--calib-samples", type=int, default=512)
    p.add_argument("--calib-batch", type=int, default=64)
    p.add_argument("--calib-batches", type=int, default=None)
    p.add_argument("--calib-tail", type=float, default=0.001, help="trim fraction each tail")
    p.add_argument("--calib-margin", type=float, default=0.05)
    p.add_argument("--softmax-kind", choices=["mlp", "pwl", "native"], default="mlp",
                   help="mlp: learned ReLU exponential; pwl: fixed geometric hinges; native: exact Softmax")
    p.add_argument("--mlp-hidden", type=int, default=6,
                   help="number of per-head ReLU neurons (default 6)")
    p.add_argument("--mlp-fit-steps", type=int, default=150,
                   help="standalone learned MLP distillation optimization steps per block")
    p.add_argument("--mlp-fit-lr", type=float, default=0.0003)
    p.add_argument("--mlp-rows", type=int, default=1536,
                   help="maximum captured attention rows per transformer block")
    p.add_argument("--mlp-freeze-after-fit", action="store_true",
                   help="do not train learned MLP coefficients in global student fine-tuning")
    p.add_argument("--softmax-segments", type=int, default=12)
    p.add_argument("--norm-type", choices=["rms", "ln", "dyt", "harddyt"], default="rms",
                   help="activation at block and final normalization sites during training")
    p.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True,
                   help="enable independent per-head transforms on Q and K")
    p.add_argument("--qk-norm-type", choices=["rms", "dyt", "harddyt"], default="rms",
                   help="per-head Q/K transform during training when --qk-norm is enabled")
    p.add_argument("--dyt-hardening-fraction", type=float, default=0.0,
                   help="fraction of teacher steps for HardDyT to ramp from tanh to clip")
    p.add_argument("--norm-mode", choices=["auto", "rms_matmul", "matmul", "affine", "pointwise"],
                   default="auto", help="student token norm implementation inferred from teacher by default")
    p.add_argument("--score-center", choices=["none", "mean", "max"], default="mean",
                   help="mean avoids ReduceMax; max is more numerically robust")
    p.add_argument("--knot-spacing", choices=["uniform", "geometric"],
                   default="geometric", help="geometric allocates knots near the exponent peak")
    p.add_argument("--test-samples", type=int, default=2000)
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--fake-samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--mlp-ratio", type=int, default=2)
    a = p.parse_args(argv)
    if a.feature_kd_weight < 0:
        p.error("--feature-kd-weight must be >= 0")
    if not (0 <= a.kd_weight <= 1) or a.kd_temperature <= 0:
        p.error("--kd-weight must be [0,1] and --kd-temperature > 0")
    if a.batch < 1 or a.calib_samples < 1 or a.calib_batch < 1:
        p.error("Batch and calibration sizes must be >=1")
    if a.stage in {"all", "train"} and a.epochs < 1:
        p.error("Training requires --epochs >= 1")
    if a.finetune_epochs < 0:
        p.error("--finetune-epochs must be >= 0")
    if a.mlp_hidden < 2 or a.mlp_rows < 2 or a.mlp_fit_steps < 0 or a.mlp_fit_lr <= 0:
        p.error("Invalid learned-MLP fitting settings")
    if a.resume_finetune and a.stage != "adapt":
        p.error("--resume-finetune is only valid for --stage adapt")
    if not 0.0 <= a.dyt_hardening_fraction < 1.0:
        p.error("--dyt-hardening-fraction must be in [0, 1)")
    if a.dyt_hardening_fraction > 0.0:
        if a.stage not in {"all", "train"}:
            p.error("--dyt-hardening-fraction requires --stage train or all")
        if a.norm_type != "harddyt" and not (a.qk_norm and a.qk_norm_type == "harddyt"):
            p.error("--dyt-hardening-fraction requires at least one HardDyT site")
    requested_norm_mode = None if a.norm_mode == "auto" else a.norm_mode
    if a.stage in {"all", "train"}:
        try:
            resolve_norm_mode(a.norm_type, requested_norm_mode)
        except ValueError as exc:
            p.error(str(exc))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_path = Path(a.teacher_ckpt) if a.teacher_ckpt else Path(a.out) / "vit_teacher.pt"
    student_path = Path(a.student_ckpt) if a.student_ckpt else Path(a.out) / "vit_npu.pt"

    if a.stage == "export":
        student = load_student(student_path, device)
        do_export(student, a)
        return

    train_dl, calib_dl, val_dl, test_dl, calib_ds, test_ds, indices = get_loaders(a)
    if a.stage == "data":
        patch = load_teacher(teacher_path, device).patch if teacher_path.is_file() else a.patch
        save_quantizer_arrays(a, calib_ds, test_ds, indices, patch)
        print(f"Calibration and evaluation arrays written to {a.out}/")
        return
    if a.stage in {"all", "train"}:
        teacher = StandardViT(dim=a.dim, depth=a.depth, heads=a.heads,
                              patch=a.patch, mlp_ratio=a.mlp_ratio,
                              norm_type=a.norm_type, qk_norm=a.qk_norm,
                              qk_norm_type=a.qk_norm_type)
        print(f"StandardViT: {sum(p.numel() for p in teacher.parameters()):,} parameters; "
              f"device={device}")
        train_teacher(teacher, train_dl, val_dl, a, device, teacher_path)
        print(f"Saved standard teacher: {teacher_path}")
        if a.stage == "train":
            return

    # Save calibration arrays before lengthy fitting/fine-tuning, not afterward.
    teacher = load_teacher(teacher_path, device)
    save_quantizer_arrays(a, calib_ds, test_ds, indices, teacher.patch)
    teacher_acc = evaluate(teacher, val_dl, device)
    norm_mode = resolve_norm_mode(teacher.norm_type, requested_norm_mode)
    qk_type = teacher.qk_norm_type.upper() if teacher.qk_norm else "off"
    print(f"Teacher validation ({teacher.norm_type.upper()}+Q/K={qk_type}"
          f"+GELU+Softmax): {teacher_acc:.4%}")
    if a.resume_finetune:
        latest_path = Path(a.out) / "vit_npu_latest.pt"
        if not latest_path.is_file():
            raise FileNotFoundError(f"Cannot resume; missing {latest_path}")
        student = load_student(latest_path, device)
        if student.config != teacher.config or student.norm_mode != norm_mode:
            raise ValueError("Teacher and resumed student architectures/norm modes differ")
        saved_attn = student.blocks[0].attn.norm
        saved_kind = ("mlp" if isinstance(saved_attn, LearnedMLPSoftmax)
                      else "pwl" if isinstance(saved_attn, FixedPWLSoftmax) else "native")
        if saved_kind != a.softmax_kind:
            raise ValueError(f"Resume attention kind mismatch: saved {saved_kind}; requested {a.softmax_kind}")
        if saved_kind != "native":
            if (saved_attn.score_center != a.score_center or
                    saved_attn.knot_spacing != a.knot_spacing):
                raise ValueError("Resume score-center/knot-spacing differ from checkpoint")
            if saved_kind == "mlp" and saved_attn.hidden != a.mlp_hidden:
                raise ValueError("Resume MLP hidden width differs from checkpoint")
            if saved_kind == "pwl" and saved_attn.segments != a.softmax_segments:
                raise ValueError("Resume PWL segments differ from checkpoint")
        print(f"Loaded NPU student from {latest_path} without recalibrating it.")
    else:
        print(f"Converting norm={norm_mode}, GELU -> calibrated ReLU, "
              f"attention {a.softmax_kind}(center={a.score_center}, knots={a.knot_spacing})...")
        student = convert_to_npu(
            teacher, calib_dl, max_batches=a.calib_batches,
            softmax_segments=a.softmax_segments, softmax_kind=a.softmax_kind,
            mlp_hidden=a.mlp_hidden, mlp_fit_steps=a.mlp_fit_steps,
            mlp_fit_lr=a.mlp_fit_lr, mlp_rows=a.mlp_rows,
            mlp_freeze=a.mlp_freeze_after_fit,
            tail=a.calib_tail, margin=a.calib_margin,
            norm_mode=norm_mode, score_center=a.score_center,
            knot_spacing=a.knot_spacing,
        )
        student_acc = evaluate(student, val_dl, device)
        print(f"Converted NPU ViT before fine-tune (validation): {student_acc:.4%}")
    finetune_student(student, teacher, train_dl, val_dl, a, device)
    print("Refreshing calibrated ReLU and attention PWL after fine-tuning...")
    pre_refresh_val = evaluate(student, val_dl, device)
    before_refresh_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    calibrate_npu(student, calib_dl, max_batches=a.calib_batches,
                  softmax_segments=a.softmax_segments, softmax_kind=a.softmax_kind,
                  mlp_hidden=a.mlp_hidden, mlp_fit_steps=a.mlp_fit_steps,
                  mlp_fit_lr=a.mlp_fit_lr, mlp_rows=a.mlp_rows,
                  mlp_freeze=a.mlp_freeze_after_fit,
                  tail=a.calib_tail, margin=a.calib_margin,
                  score_center=a.score_center, knot_spacing=a.knot_spacing)
    refreshed_val = evaluate(student, val_dl, device)
    if refreshed_val < pre_refresh_val:
        # Same buffers and shapes: safely restore the pre-refresh PWL coefficients.
        student.load_state_dict(before_refresh_state)
        print(f"Keeping pre-refresh coefficients: {pre_refresh_val:.4%} > {refreshed_val:.4%}")
    else:
        print(f"Keeping refreshed coefficients: {refreshed_val:.4%} >= {pre_refresh_val:.4%}")
    final_val = evaluate(student, val_dl, device)
    final_acc = evaluate(student, test_dl, device)
    teacher_test = evaluate(teacher, test_dl, device)
    print(f"Final test comparison: standard={teacher_test:.4%} "
          f"NPU-converted={final_acc:.4%} (delta={final_acc-teacher_test:+.2%}); "
          f"NPU validation={final_val:.4%}")
    print("PWL saturation by block (held-out validation):",
          measure_saturation(student, val_dl, max_batches=min(4, len(val_dl))))
    save_student(student, student_path, final_acc)
    print(f"Saved student checkpoint: {student_path}")
    save_quantizer_arrays(a, calib_ds, test_ds, indices, student.patch)
    print(f"Saved calib.npy, test_x.npy and test_y.npy in {a.out}/")
    do_export(student, a)


if __name__ == "__main__":
    main()
