"""
model.py - a small, conv-free, NPU-friendly Vision Transformer for AMD Ryzen AI
(Hawk Point / XDNA1, XINT8 flow).

The exported graph contains only: MatMul, Add, Mul, Sub, Div, Max, ReduceMax,
Relu, Reshape, Transpose (+ constants). No Conv, no BatchNorm, no pooling op.

  * Patch embedding -> patchify outside the model (pure data rearrangement)
                       + Linear (MatMul)
  * LayerNorm       -> BatchNorm over channels, folded into the Linear layers at export
  * GELU            -> ReLU
  * Softmax         -> PWLSoftmax (ReduceMax, Sub, Max, Add, ReLU, Mul, MatMul, Div)
  * Mean pooling    -> MatMul with a constant [1, N] vector of 1/N
  * Tokens use [B, N, C] layout; nn.Linear on 3D input exports as MatMul + Add.

Self-test:  uv run npu-vit-selftest
"""

import collections
import copy
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn


# --- Patchify (done on the CPU, before the model) -----------------------------
def patchify(x, p=4):
    """[B, C, H, W] torch tensor -> [B, (H/p)*(W/p), C*p*p]"""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // p) * (W // p), C * p * p)


def patchify_np(x, p=4):
    """Same as patchify, for NumPy arrays (used at deployment)."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // p, p, W // p, p).transpose(0, 2, 4, 1, 3, 5)
    return np.ascontiguousarray(x.reshape(B, (H // p) * (W // p), C * p * p))


# --- Piecewise-linear exp(z) for z in [-5.5, 0] ------------------------------
KNOTS = [5.5, 3.5, 2.5, 1.75, 1.25, 0.75, 0.35]
COEFS = [0.0129, 0.0385, 0.0702, 0.1028, 0.1426, 0.2147, 0.2585]
Z_MIN = -5.5  # below this, exp(z) < 1/255 -> rounds to 0 in INT8 anyway
SHIFT = 2.75  # recentre [-5.5, 0] to [-2.75, 2.75] so symmetric INT8 uses all levels


class PWLSoftmax(nn.Module):
    """Softmax over the last dim, built only from NPU-friendly ops."""

    def __init__(self, n, learnable=False):
        super().__init__()
        bias = torch.tensor(KNOTS) - SHIFT
        coef = torch.tensor(COEFS)
        if learnable:
            self.bias, self.coef = nn.Parameter(bias), nn.Parameter(coef)
        else:
            self.register_buffer("bias", bias)
            self.register_buffer("coef", coef)
        self.register_buffer("z_min", torch.tensor(Z_MIN))
        self.register_buffer("ones", torch.ones(n, 1))  # row sum as a MatMul

    def forward(self, s):
        z = s - s.amax(dim=-1, keepdim=True)  # ReduceMax + Sub   -> z <= 0
        u = torch.maximum(z, self.z_min) + SHIFT  # Max + Add     -> [-2.75, 2.75]
        e = 0
        for i in range(len(KNOTS)):  # 7 x (Add, ReLU, Mul), summed
            e = e + self.coef[i] * torch.relu(u + self.bias[i])
        return e / (e @ self.ones)  # MatMul (row sum) + Div


class ReLUAttnNorm(nn.Module):
    """ReLU(s) / n - the simplest softmax-free alternative, for comparison."""

    def __init__(self, n):
        super().__init__()
        self.inv_n = 1.0 / n

    def forward(self, s):
        return torch.relu(s) * self.inv_n


def make_attn_norm(kind, n):
    if kind == "pwl":
        return PWLSoftmax(n)
    if kind == "pwl_learn":
        return PWLSoftmax(n, learnable=True)
    if kind == "relu":
        return ReLUAttnNorm(n)
    if kind == "softmax":  # exact baseline (not NPU-friendly)
        return nn.Softmax(dim=-1)
    raise ValueError(kind)


# --- BatchNorm over the last (channel) dim of [B, N, C] ---------------------------
class BNLast(nn.BatchNorm1d):
    """Per-channel BatchNorm for [B, N, C]; statistics over all B*N tokens."""

    def forward(self, x):
        return super().forward(x.reshape(-1, x.shape[-1])).reshape(x.shape)


# --- Transformer -------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads, n_tokens, attn="pwl"):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.hd = heads, dim // heads
        self.scale = self.hd**-0.5
        self.q = nn.Linear(dim, dim)  # three Linears: no Split op
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = make_attn_norm(attn, n_tokens)

    def forward(self, x):  # x: [B, N, C]
        B, N, C = x.shape
        h, d = self.heads, self.hd
        q = (self.q(x) * self.scale).reshape(B, N, h, d).transpose(1, 2)  # [B,h,N,d]
        k = self.k(x).reshape(B, N, h, d).permute(0, 2, 3, 1)  # [B,h,d,N]
        v = self.v(x).reshape(B, N, h, d).transpose(1, 2)  # [B,h,N,d]
        a = self.norm(q @ k)  # [B,h,N,N]
        o = (a @ v).transpose(1, 2).reshape(B, N, C)  # [B,N,C]
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, n_tokens, attn):
        super().__init__()
        hidden = dim * mlp_ratio
        self.norm1 = BNLast(dim)
        self.attn = Attention(dim, heads, n_tokens, attn)
        self.norm2 = BNLast(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.bn_mid = BNLast(hidden)  # extra BN in the MLP: helps BN-ViT training
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(self.act(self.bn_mid(self.fc1(self.norm2(x)))))


class NPUViT(nn.Module):
    """Input: patches [B, N, patch_dim] from patchify(); output: logits [B, classes]."""

    def __init__(
        self,
        img_size=32,
        patch=4,
        in_ch=3,
        num_classes=10,
        dim=128,
        depth=6,
        heads=4,
        mlp_ratio=2,
        attn="pwl",
    ):
        super().__init__()
        n = (img_size // patch) ** 2
        self.n_tokens, self.patch_dim = n, in_ch * patch * patch
        self.patch_embed = nn.Linear(self.patch_dim, dim)
        self.patch_bn = BNLast(dim)
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, n, attn) for _ in range(depth))
        self.norm = BNLast(dim)
        self.register_buffer("avg", torch.full((1, n), 1.0 / n))  # mean pool as MatMul
        self.head = nn.Linear(dim, num_classes)

    def forward(self, p):  # p: [B, N, patch_dim]
        x = self.patch_bn(self.patch_embed(p)) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.avg @ self.norm(x)  # [B, 1, C]
        return self.head(x).reshape(x.shape[0], -1)  # [B, classes]


# --- BatchNorm folding into Linear layers (exact, inference only) -------------------
def _bn_affine(bn):
    a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return a, bn.bias - bn.running_mean * a


@torch.no_grad()
def _fold_after(lin, bn):
    """linear -> bn   ==>   linear'"""
    a, b = _bn_affine(bn)
    lin.weight.mul_(a.view(-1, 1))
    lin.bias.copy_(lin.bias * a + b)


@torch.no_grad()
def _fold_before(bn, *lins):
    """bn -> linear(s)   ==>   linear'(s)"""
    a, b = _bn_affine(bn)
    for lin in lins:
        lin.bias.add_(lin.weight @ b)
        lin.weight.mul_(a.view(1, -1))


@torch.no_grad()
def fold_batchnorms(model):
    """Return a copy of the model with every BatchNorm folded into a Linear."""
    m = copy.deepcopy(model).eval()
    _fold_after(m.patch_embed, m.patch_bn)
    m.patch_bn = nn.Identity()
    for blk in m.blocks:
        _fold_before(blk.norm1, blk.attn.q, blk.attn.k, blk.attn.v)
        blk.norm1 = nn.Identity()
        _fold_after(blk.fc1, blk.bn_mid)
        blk.bn_mid = nn.Identity()
        _fold_before(blk.norm2, blk.fc1)
        blk.norm2 = nn.Identity()
    _fold_before(m.norm, m.head)
    m.norm = nn.Identity()  # BN commutes with mean pooling
    return m


# --- Export + op audit ----------------------------------------------------------
# Ops AMD's Ryzen AI 1.8 table lists for XINT8 (the format to use on Hawk Point).
XINT8_LISTED = {
    "Add",
    "AveragePool",
    "BatchNormalization",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Conv",
    "DepthToSpace",
    "Div",
    "Expand",
    "Flatten",
    "Gemm",
    "GlobalAveragePool",
    "Identity",
    "InstanceNormalization",
    "MatMul",
    "Max",
    "MaxPool",
    "Min",
    "Mul",
    "Pad",
    "Reciprocal",
    "ReduceMax",
    "Reshape",
    "Shape",
    "Slice",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
}


def export_onnx(model, path):
    folded = fold_batchnorms(model)
    dummy = torch.randn(1, model.n_tokens, model.patch_dim)
    kw = {"opset_version": 17, "input_names": ["input"], "output_names": ["logits"]}
    try:
        torch.onnx.export(folded, dummy, str(path), dynamo=False, **kw)
    except TypeError:  # torch < 2.5 has no `dynamo` arg
        torch.onnx.export(folded, dummy, str(path), **kw)
    return str(path)


def audit_onnx(path):
    import onnx

    ops = collections.Counter(n.op_type for n in onnx.load(str(path)).graph.node)
    print(f"\nOp audit for {path}:")
    for op, n in sorted(ops.items()):
        note = "XINT8 listed" if op in XINT8_LISTED else "NOT listed for XINT8 - check NPU log"
        if op == "Relu":
            note = "not in XINT8 column - check NPU log"
        print(f"  {op:20s} x{n:<4d} {note}")
    return ops


def selftest():
    """Run a quick self-test covering PWLSoftmax, patchify, BN folding, and ONNX export."""
    torch.manual_seed(0)

    s = torch.randn(1000, 64) * 2.0
    err = (PWLSoftmax(64)(s) - torch.softmax(s, -1)).abs()
    print(f"PWLSoftmax vs softmax: max abs err {err.max():.4f}, mean {err.mean():.5f}")

    imgs = torch.randn(4, 3, 32, 32)
    assert np.allclose(patchify(imgs).numpy(), patchify_np(imgs.numpy()))
    print("patchify (torch) == patchify_np (numpy): OK")

    model = NPUViT()
    model.train()
    for _ in range(3):  # populate BN running stats
        model(patchify(torch.randn(16, 3, 32, 32)))
    model.eval()
    x = patchify(imgs)
    diff = (model(x) - fold_batchnorms(model)(x)).abs().max()
    print(f"BN folding max diff: {diff:.2e}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_onnx(model, os.path.join(tmpdir, "npu_vit_selftest.onnx"))
        audit_onnx(path)
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            out = sess.run(None, {"input": x[:1].numpy()})[0]
            print(
                "\nONNX Runtime vs PyTorch max diff: "
                f"{abs(out - model(x[:1]).detach().numpy()).max():.2e}"
            )
        except ImportError:
            print("onnxruntime not installed - skipped runtime check")


if __name__ == "__main__":
    selftest()
