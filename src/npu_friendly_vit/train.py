"""
train.py - train NPUViT on CIFAR-10, then export for the NPU.

Run with uv (a GPU machine is much faster; on the Ryzen 9 8945HS CPU expect a few
minutes per epoch):

    uv run npu-vit-train --attn pwl --epochs 100
    uv run npu-vit-train --attn softmax --epochs 100   # exact-softmax baseline to compare

Outputs (in --out):
    npu_vit_<attn>.pt      best checkpoint
    npu_vit_<attn>.onnx    conv-free, BN-folded, input = patches [1, 64, 48], opset 17
    calib.npy / test_x.npy / test_y.npy   normalized + patchified data for npu-vit-quantize
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from npu_friendly_vit.model import NPUViT, audit_onnx, export_onnx, patchify, patchify_np

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_loaders(batch, workers, fake=False, data_dir="./data"):
    if fake:  # smoke test without downloading CIFAR
        x, y = torch.randn(512, 3, 32, 32), torch.randint(0, 10, (512,))
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch, shuffle=True), DataLoader(ds, batch), ds, ds

    import torchvision
    import torchvision.transforms as T

    norm = T.Normalize(MEAN, STD)
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
            T.ToTensor(),
            norm,
        ]
    )
    test_tf = T.Compose([T.ToTensor(), norm])
    train = torchvision.datasets.CIFAR10(data_dir, True, train_tf, download=True)
    test = torchvision.datasets.CIFAR10(data_dir, False, test_tf, download=True)
    calib = torchvision.datasets.CIFAR10(data_dir, True, test_tf, download=True)  # no augmentation
    return (
        DataLoader(train, batch, shuffle=True, num_workers=workers, drop_last=True),
        DataLoader(test, 256, num_workers=workers),
        calib,
        test,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(patchify(x.to(device))).argmax(1).cpu() == y).sum().item()
        total += y.numel()
    return correct / total


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--attn", default="pwl", choices=["pwl", "pwl_learn", "relu", "softmax"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out", default="out")
    p.add_argument("--fake-data", action="store_true", help="smoke test with random data")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    train_dl, test_dl, calib_ds, test_ds = get_loaders(
        a.batch, a.workers, a.fake_data, data_dir=a.data_dir
    )

    model = NPUViT(attn=a.attn).to(device)
    print(
        f"NPUViT ({a.attn}): {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params, "
        f"device={device}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = len(train_dl)
    total, warm = a.epochs * steps_per_epoch, a.warmup * steps_per_epoch

    def lr_at(step):  # linear warmup, then cosine decay
        if step < warm:
            return a.lr * (step + 1) / max(1, warm)
        return 0.5 * a.lr * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    ckpt = os.path.join(a.out, f"npu_vit_{a.attn}.pt")
    best, step = -1.0, 0
    for epoch in range(a.epochs):
        model.train()
        t0, run_loss = time.time(), 0.0
        for x, y in train_dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(patchify(x)), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # BN-ViTs like this
            opt.step()
            run_loss += loss.item()
            step += 1
        acc = evaluate(model, test_dl, device)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), ckpt)
        print(
            f"epoch {epoch + 1:3d}/{a.epochs}  loss {run_loss / steps_per_epoch:.3f}  "
            f"test acc {acc * 100:.2f}%  best {best * 100:.2f}%  ({time.time() - t0:.0f}s)"
        )

    # --- export best model for the NPU ---
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model = model.cpu().eval()
    onnx_path = export_onnx(model, os.path.join(a.out, f"npu_vit_{a.attn}.onnx"))
    audit_onnx(onnx_path)

    # --- save preprocessed calibration + test data (no torchvision needed later) ---
    rng = np.random.default_rng(0)
    idx = rng.choice(len(calib_ds), size=min(500, len(calib_ds)), replace=False)
    np.save(
        os.path.join(a.out, "calib.npy"),
        patchify_np(np.stack([calib_ds[i][0].numpy() for i in idx]).astype(np.float32)),
    )
    n_test = min(2000, len(test_ds))
    np.save(
        os.path.join(a.out, "test_x.npy"),
        patchify_np(np.stack([test_ds[i][0].numpy() for i in range(n_test)]).astype(np.float32)),
    )
    np.save(
        os.path.join(a.out, "test_y.npy"),
        np.array([int(test_ds[i][1]) for i in range(n_test)], dtype=np.int64),
    )
    print(f"\nSaved {onnx_path}, calib.npy, test_x.npy, test_y.npy in {a.out}/")


if __name__ == "__main__":
    main()
