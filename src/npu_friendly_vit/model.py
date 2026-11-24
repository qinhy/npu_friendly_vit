"""Standard RMSNorm + QK-Norm ViT -> calibrated ReLU / PWL NPU ViT.

Teacher trains with dynamic RMSNorm, per-head Q/K RMSNorm, GELU and Softmax.
Student uses equivalent MatMul RMSNorm, calibrated GELU -> ReLU, and either
a learned six-neuron per-head ReLU MLP, fixed PWL, or native Softmax.
The preceding LayerNorm (ln) architecture remains available as an ablation.
FP32 graph suitability and real INT8 NPU partitioning are distinct checks.
"""

from __future__ import annotations

import collections
import copy
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# CPU-side patchification; exported model only receives [B,N,patch_dim] patches.
def patchify(x: torch.Tensor, p: int = 4) -> torch.Tensor:
    b, c, h, w = x.shape
    if h % p or w % p:
        raise ValueError("Image dimensions must be divisible by patch size")
    return x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(
        b, (h // p) * (w // p), c * p * p
    )


def patchify_np(x: np.ndarray, p: int = 4) -> np.ndarray:
    b, c, h, w = x.shape
    if h % p or w % p:
        raise ValueError("Image dimensions must be divisible by patch size")
    return np.ascontiguousarray(
        x.reshape(b, c, h // p, p, w // p, p)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(b, (h // p) * (w // p), c * p * p)
    )


class TokenRMSNorm(nn.Module):
    """Teacher RMSNorm: normalized over the channel dimension, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class MatMulRMSNorm(nn.Module):
    """Same RMSNorm math using constant MatMul instead of ReduceMean."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.register_buffer("avg", torch.full((dim, 1), 1.0 / dim), persistent=False)

    def forward(self, x):
        return x / torch.sqrt((x * x) @ self.avg + self.eps) * self.weight


class HeadRMSNorm(nn.Module):
    """Teacher QK-Norm, per head and per token over head_dim.

    Each head owns its gain vector; Q and K use separate instances.
    """

    def __init__(self, heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, head_dim))
        self.eps = eps

    def forward(self, x):  # [B, H, N, D]
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight[None, :, None, :]


class MatMulHeadRMSNorm(nn.Module):
    """NPU QK-Norm: a constant MatMul replaces ReduceMean over head_dim."""

    def __init__(self, heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, head_dim))
        self.eps = eps
        self.register_buffer("avg", torch.full((head_dim, 1), 1.0 / head_dim), persistent=False)

    def forward(self, x):  # [B, H, N, D]
        return x / torch.sqrt((x * x) @ self.avg + self.eps) * self.weight[None, :, None, :]


class StaticChannelNorm(nn.Module):
    """Trainable per-channel affine approximation to LayerNorm.

    No sample-wise mean/variance/reduction at runtime. Parameters are fitted by
    linear regression against the original LayerNorm outputs, then fine-tuned.
    Unlike LayerNorm this is NOT an exact transformation and needs adaptation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.weight + self.bias


class MatMulLayerNorm(nn.Module):
    """Exact LayerNorm in inference-friendly primitive ops, without LayerNormalization.

    Unlike a static channel affine map, this retains the input-dependent mean
    and variance. The reductions are fixed MatMuls; verify actual target NPU
    partitioning / INT8 support for Sqrt and Div before deployment.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.register_buffer("avg", torch.full((dim, 1), 1.0 / dim), persistent=False)

    def forward(self, x):
        center = x - (x @ self.avg)
        variance = (center * center) @ self.avg
        return (center / torch.sqrt(variance + self.eps)) * self.weight + self.bias


class CalibratedReLU(nn.Module):
    """Per-channel approximation GELU(x) ~= gain * ReLU(x) + bias.

    Only the *ordinary ReLU* remains after exporting: gain and bias are frozen
    calibration buffers and are folded exactly into the following fc2 Linear.
    """

    def __init__(self, gain: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        gain = torch.as_tensor(gain, dtype=torch.float32).detach().flatten()
        bias = torch.as_tensor(bias, dtype=torch.float32).detach().flatten()
        if gain.shape != bias.shape or gain.numel() == 0:
            raise ValueError("CalibratedReLU requires matching per-channel vectors")
        if not torch.isfinite(gain).all() or not torch.isfinite(bias).all():
            raise ValueError("Nonfinite ReLU calibration coefficients")
        self.register_buffer("gain", gain.clone())
        self.register_buffer("bias", bias.clone())

    def forward(self, x):
        return torch.relu(x) * self.gain + self.bias


def _upgrade_legacy_hinge_parameters(state_dict, prefix, count, old_bias, old_coef):
    """Convert old per-neuron checkpoint tensors to packed per-head tensors.

    This runs only while loading checkpoints; there is no loop in inference.
    """
    bkey, ckey = prefix + "biases", prefix + "coefs"
    if bkey in state_dict and ckey in state_dict:
        return
    bkeys = [prefix + old_bias.format(i) for i in range(count)]
    ckeys = [prefix + old_coef.format(i) for i in range(count)]
    if not all(k in state_dict for k in bkeys + ckeys):
        return
    state_dict[bkey] = torch.cat([state_dict.pop(k) for k in bkeys], dim=-1)
    state_dict[ckey] = torch.cat([state_dict.pop(k) for k in ckeys], dim=-1).transpose(-1, -2).contiguous()


class FixedPWLSoftmax(nn.Module):
    """Per-head fixed PWL approximation to Softmax with optional score centering.

    Calibration defines low_h, high_h and fits exp(s-high_h) with per-head
    knots. The per-head constant shift does not affect exact Softmax outputs.
    Uniform or geometric knot spacing is supported. This is range-observed
    calibration, not a temperature-changing RangeBN transform.

    IMPORTANT: unlike exact Softmax, the result is not shift invariant outside
    the fitted range. Clip rates and task accuracy must be measured.
    """

    def __init__(
        self,
        n_tokens: int,
        low: torch.Tensor,
        high: torch.Tensor,
        segments: int = 12,
        radius: float = 3.0,
        min_exp: float = 1e-8,
        score_center: str = "none",
        knot_spacing: str = "uniform",
    ):
        super().__init__()
        if segments < 2 or radius <= 0 or min_exp <= 0:
            raise ValueError("Invalid PWL settings")
        if score_center not in {"none", "mean", "max"}:
            raise ValueError("score_center must be none, mean, or max")
        if knot_spacing not in {"uniform", "geometric"}:
            raise ValueError("knot_spacing must be uniform or geometric")
        low = torch.as_tensor(low, dtype=torch.float32).flatten().detach()
        high = torch.as_tensor(high, dtype=torch.float32).flatten().detach()
        if low.shape != high.shape or not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("Invalid attention head ranges")
        if not (high > low).all():
            raise ValueError("Each attention high must exceed low")
        self.segments = segments
        self.radius = float(radius)
        self.score_center = score_center
        self.knot_spacing = knot_spacing
        self.register_buffer("low", low.view(1, -1, 1, 1).clone())
        self.register_buffer("high", high.view(1, -1, 1, 1).clone())
        # Fixed per-head knots in centered, calibrated raw-score coordinates.
        t = torch.linspace(0, 1, segments + 1, device=low.device)
        width = (high - low)[:, None]
        if knot_spacing == "uniform":
            knots = low[:, None] + width * t[None, :]
        else:
            # Most knots near the highest exponent values, where interpolation
            # error matters. Still fixed coefficients after calibration.
            knots = high[:, None] - torch.expm1(
                (1 - t[None, :]) * torch.log1p(width)
            )
        # Head-wise subtraction of high is an allowed shift (not a per-row max).
        values = torch.exp((knots - high[:, None]).clamp(min=-80)).clamp_min(min_exp)
        slopes = (values[:, 1:] - values[:, :-1]) / (knots[:, 1:] - knots[:, :-1])
        delta = torch.cat((slopes[:, :1], slopes[:, 1:] - slopes[:, :-1]), dim=1)
        self.register_buffer("base", values[:, :1].reshape(1, -1, 1, 1).clone())
        # Packed constant coefficients for a single batched MatMul, rather
        # than `segments` separate ReLU/Mul/Add nodes in the ONNX graph.
        # biases [1,H,1,M], coefs [1,H,M,1]
        self.register_buffer("biases", -knots[:, :segments].view(1, -1, 1, segments).clone())
        self.register_buffer("coefs", delta.view(1, -1, segments, 1).clone())
        self.register_buffer("ones", torch.ones(n_tokens, 1, device=low.device))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _upgrade_legacy_hinge_parameters(state_dict, prefix, self.segments,
                                         "bias_{}", "coef_{}")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def center_scores(self, scores):
        if self.score_center == "max":
            return scores - scores.amax(dim=-1, keepdim=True)
        if self.score_center == "mean":
            return scores - (scores @ self.ones) * (1.0 / self.ones.shape[0])
        return scores

    def forward(self, scores):
        centered = self.center_scores(scores)
        s = torch.minimum(torch.maximum(centered, self.low), self.high)
        b, h, q, k = s.shape
        # [B,H,Q*K,1] + [1,H,1,M] -> [B,H,Q*K,M]
        features = torch.relu(s.reshape(b, h, q * k, 1) + self.biases)
        # [B,H,Q*K,M] @ [1,H,M,1] -> [B,H,Q*K,1]
        e = (self.base + (features @ self.coefs)).reshape(b, h, q, k)
        exp_pwl = torch.maximum(e, self.base * 0.5)
        return exp_pwl / (exp_pwl @ self.ones)


class LearnedMLPSoftmax(nn.Module):
    """Per-head six-neuron scalar ReLU MLP for the exponential in Softmax.

    f_h(s) = base_h + sum_i coef_hi * ReLU(s - knot_hi)

    This is a constrained 1 -> hidden -> 1 ReLU MLP: the first-layer input
    weights are fixed at +1, while hidden biases (-knot), output weights and
    output bias are LEARNED from attention-score rows. Fixing the input weights
    is equivalent to a continuous piecewise-linear network and avoids extra
    Mul nodes in ONNX. Hidden neurons are evaluated together by one batched
    ReLU + MatMul (no per-neuron Python loop or multi-node chain).
    There are H * (2*hidden+1) trainable scalars.

    No exp or row-wise ReduceMax in inference with score_center='mean'. The
    row reduction and division remain necessary. All neuron weights can be
    kept after learning or frozen to constants by calling .freeze().
    """

    def __init__(self, n_tokens, low, high, hidden=6, score_center="mean",
                 knot_spacing="geometric", min_exp=1e-8):
        super().__init__()
        if hidden < 2:
            raise ValueError("hidden must be >= 2")
        if score_center not in {"none", "mean", "max"}:
            raise ValueError("invalid score_center")
        if knot_spacing not in {"uniform", "geometric"}:
            raise ValueError("invalid knot_spacing")
        low = torch.as_tensor(low, dtype=torch.float32).detach().flatten()
        high = torch.as_tensor(high, dtype=torch.float32).detach().flatten()
        if low.numel() == 0 or low.shape != high.shape or not torch.isfinite(low).all() or not torch.isfinite(high).all() or not (high > low).all():
            raise ValueError("invalid per-head score bounds")
        self.hidden = int(hidden)
        self.score_center = score_center
        self.knot_spacing = knot_spacing
        self.min_exp = float(min_exp)
        self.register_buffer("low", low.view(1, -1, 1, 1).clone())
        self.register_buffer("high", high.view(1, -1, 1, 1).clone())
        self.register_buffer("ones", torch.ones(n_tokens, 1))
        self.register_buffer("min_value", torch.tensor(float(min_exp)))
        # Warm start at the same six-segment interpolation of exp(s-high) used
        # by fixed PWL. The knot positions and slopes are then optimized.
        t = torch.linspace(0, 1, hidden + 1, device=low.device)
        width = (high - low)[:, None]
        if knot_spacing == "geometric":
            knots = high[:, None] - torch.expm1((1 - t[None, :]) * torch.log1p(width))
        else:
            knots = low[:, None] + width * t[None, :]
        values = torch.exp((knots - high[:, None]).clamp(min=-80)).clamp_min(min_exp)
        slopes = (values[:, 1:] - values[:, :-1]) / (knots[:, 1:] - knots[:, :-1])
        deltas = torch.cat((slopes[:, :1], slopes[:, 1:] - slopes[:, :-1]), dim=1)
        self.base = nn.Parameter(values[:, :1].view(1, -1, 1, 1).clone())
        # Two packed trainable tensors replace 2*hidden ParameterList entries.
        # Exactly 2*hidden+1 learnable scalars per head are retained.
        self.biases = nn.Parameter((-knots[:, :hidden]).view(1, -1, 1, hidden).clone())
        self.coefs = nn.Parameter(deltas.view(1, -1, hidden, 1).clone())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Supports vit_npu*.pt created by the original per-neuron for-loop model.
        _upgrade_legacy_hinge_parameters(state_dict, prefix, self.hidden,
                                         "biases.{}", "coefs.{}")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def center_scores(self, scores):
        if self.score_center == "max":
            return scores - scores.amax(dim=-1, keepdim=True)
        if self.score_center == "mean":
            return scores - (scores @ self.ones) * (1.0 / self.ones.shape[0])
        return scores

    def forward(self, scores):
        z = self.center_scores(scores)
        z = torch.minimum(torch.maximum(z, self.low), self.high)
        b, h, q, k = z.shape
        # A single batched hidden-layer ReLU and a single batched MatMul;
        # there is NO Python loop over neurons during training or inference.
        hidden = torch.relu(z.reshape(b, h, q * k, 1) + self.biases)
        values = (self.base + hidden @ self.coefs).reshape(b, h, q, k)
        positive = torch.maximum(values, self.min_value)
        return positive / (positive @ self.ones)

    def freeze(self):
        self.requires_grad_(False)
        return self


class _ScoreRowSamples:
    """Capture bounded representative attention rows for standalone MLP fitting."""

    def __init__(self, max_rows=1536, per_batch=256):
        if max_rows <= 1:
            raise ValueError("max_rows must exceed one")
        self.max_rows = int(max_rows)
        self.per_batch = int(per_batch)
        self.samples = []
        self.count = 0

    @torch.no_grad()
    def hook(self, module, args):
        if self.count >= self.max_rows:
            return
        x = args[0].detach()  # [B,H,Q,K]
        rows = x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])
        take = min(self.per_batch, self.max_rows - self.count, rows.shape[0])
        indices = torch.linspace(0, rows.shape[0] - 1, take, device=rows.device).long()
        self.samples.append(rows[indices].float().cpu())
        self.count += take

    def get(self):
        if not self.samples:
            raise ValueError("No attention rows collected")
        return torch.cat(self.samples, 0)


def _row_kl(model, rows):
    """KL(exact_softmax(scores) || learned_softmax(scores)), mean over rows/heads."""
    # rows [R,H,K] -> scores [1,H,R,K]
    scores = rows.permute(1, 0, 2).unsqueeze(0)
    target = torch.softmax(scores, dim=-1)
    predicted = model(scores).clamp_min(1e-9)
    return (target * (target.clamp_min(1e-9).log() - predicted.log())).sum(-1).mean()


def fit_mlp_softmax(module, rows, *, steps=150, lr=0.0003, batch_rows=128, verbose=False):
    """Learn per-head MLP coefficients from current exact attention distributions.

    Uses held-out CALIBRATION rows for early model selection (never the test set).
    Only module parameters are optimized. The ViT and source score rows stay
    frozen. The best validation-fit parameters are restored.
    """
    if steps < 0 or lr <= 0:
        raise ValueError("steps must be nonnegative and lr positive")
    if rows.ndim != 3 or rows.shape[0] < 2:
        raise ValueError("rows must have shape [R,H,K] with at least 2 rows")
    device = module.base.device
    rows = rows.to(device)
    # Random split drawn from a LOCAL generator (no global RNG perturbation).
    rng = torch.Generator(device="cpu").manual_seed(20261124)
    order = torch.randperm(rows.shape[0], generator=rng).to(device)
    n_val = max(1, int(0.15 * rows.shape[0]))
    train_rows, val_rows = rows[order[n_val:]], rows[order[:n_val]]
    with torch.no_grad():
        first = float(_row_kl(module, val_rows))
    best, best_state = first, {k: v.detach().clone() for k, v in module.state_dict().items()}
    if steps:
        opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=0.0)
        with torch.enable_grad():
            for step in range(steps):
                idx = torch.randint(0, len(train_rows),
                                    (min(batch_rows, len(train_rows)),), device=device)
                loss = _row_kl(module, train_rows[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite MLP approximation loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
                opt.step()
                if (step + 1) % 10 == 0 or step + 1 == steps:
                    with torch.no_grad():
                        current = float(_row_kl(module, val_rows))
                        if current < best:
                            best = current
                            best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    module.load_state_dict(best_state)
    if verbose:
        print(f"      6-neuron MLP fit: heldout attention KL {first:.6f} -> {best:.6f} ({steps} steps)", flush=True)
    return {"initial_kl": first, "best_kl": best, "train_rows": len(train_rows), "validation_rows": len(val_rows)}


class Attention(nn.Module):
    def __init__(self, dim, heads, n_tokens, *, qk_norm=True, adapted=False):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.hd = heads, dim // heads
        self.scale = self.hd ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.qk_norm = qk_norm
        if qk_norm:
            qkn = MatMulHeadRMSNorm if adapted else HeadRMSNorm
            self.q_norm = qkn(heads, self.hd)
            self.k_norm = qkn(heads, self.hd)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.norm = nn.Softmax(dim=-1)
        self.n_tokens = n_tokens

    def forward(self, x):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.heads, self.hd).transpose(1, 2)
        k = self.k(x).reshape(b, n, self.heads, self.hd).transpose(1, 2)
        v = self.v(x).reshape(b, n, self.heads, self.hd).transpose(1, 2)
        # Query and Key are normalized independently, prior to Q @ K^T.
        q = self.q_norm(q)
        k = self.k_norm(k)
        attention = self.norm((q @ k.transpose(-2, -1)) * self.scale)
        return self.proj((attention @ v).transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, n_tokens, *, adapted=False,
                 norm_type="rms", norm_mode="rms_matmul", qk_norm=True):
        super().__init__()
        if norm_type == "rms":
            norm = MatMulRMSNorm if adapted else TokenRMSNorm
        elif norm_type == "ln":
            norm = (StaticChannelNorm if norm_mode == "affine" else MatMulLayerNorm) if adapted else nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        self.norm1 = norm(dim)
        self.attn = Attention(dim, heads, n_tokens, qk_norm=qk_norm, adapted=adapted)
        self.norm2 = norm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class _ViT(nn.Module):
    def __init__(self, img_size=32, patch=4, in_ch=3, num_classes=10,
                 dim=128, depth=6, heads=4, mlp_ratio=2, adapted=False,
                 norm_type="rms", norm_mode=None, qk_norm=True):
        super().__init__()
        if img_size % patch:
            raise ValueError("img_size must be divisible by patch")
        if norm_type not in {"rms", "ln"}:
            raise ValueError("norm_type must be rms or ln")
        if norm_mode is None:
            norm_mode = "rms_matmul" if norm_type == "rms" else "matmul"
        if norm_mode not in {"affine", "matmul", "rms_matmul"}:
            raise ValueError("norm_mode must be affine, matmul or rms_matmul")
        if adapted and ((norm_type == "rms" and norm_mode != "rms_matmul") or
                        (norm_type == "ln" and norm_mode == "rms_matmul")):
            raise ValueError("rms requires rms_matmul; ln requires matmul or affine")
        self.config = dict(img_size=img_size, patch=patch, in_ch=in_ch,
                           num_classes=num_classes, dim=dim, depth=depth,
                           heads=heads, mlp_ratio=mlp_ratio,
                           norm_type=norm_type, qk_norm=qk_norm)
        self.norm_mode = norm_mode
        self.norm_type = norm_type
        self.qk_norm = qk_norm
        self.patch = patch
        self.n_tokens = (img_size // patch) ** 2
        self.patch_dim = in_ch * patch * patch
        self.patch_embed = nn.Linear(self.patch_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            Block(dim, heads, mlp_ratio, self.n_tokens, adapted=adapted,
                  norm_type=norm_type, norm_mode=norm_mode, qk_norm=qk_norm)
            for _ in range(depth)
        )
        if norm_type == "rms":
            norm = MatMulRMSNorm if adapted else TokenRMSNorm
        else:
            norm = (StaticChannelNorm if norm_mode == "affine" else MatMulLayerNorm) if adapted else nn.LayerNorm
        self.norm = norm(dim)
        self.register_buffer("avg", torch.full((1, self.n_tokens), 1.0 / self.n_tokens))
        self.head = nn.Linear(dim, num_classes)

    def forward(self, patches):
        x = self.patch_embed(patches) + self.pos
        for block in self.blocks:
            x = block(x)
        x = self.avg @ self.norm(x)
        return self.head(x).reshape(x.shape[0], -1)


class StandardViT(_ViT):
    """FP32 teacher: RMSNorm + QK RMSNorm + GELU + exact Softmax by default."""

    def __init__(self, **kwargs):
        super().__init__(adapted=False, **kwargs)


class NPUViT(_ViT):
    """Student: constant-MatMul RMSNorm/QK-Norm + calibrated ReLU and PWL."""

    def __init__(self, norm_mode=None, **kwargs):
        super().__init__(adapted=True, norm_mode=norm_mode, **kwargs)


# ---- Input and streaming observers: all executed only in calibration ----
def _as_patches(batch, model, device=None):
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    if not isinstance(batch, torch.Tensor):
        batch = torch.as_tensor(batch)
    if device is None:
        device = next(model.parameters()).device
    batch = batch.to(device, dtype=next(model.parameters()).dtype)
    if batch.ndim == 4:
        batch = patchify(batch, model.patch)
    if batch.ndim != 3 or batch.shape[1:] != (model.n_tokens, model.patch_dim):
        raise ValueError("Calibration expects images [B,C,H,W] or patches [B,N,D]")
    return batch


class _AffineRegression:
    """Collect per-channel moments for fitting LN(x) ~ gain*x + bias."""

    def __init__(self, channels):
        self.channels = channels
        self.n = 0
        self.sx = torch.zeros(channels, dtype=torch.float64)
        self.sy = torch.zeros(channels, dtype=torch.float64)
        self.sxx = torch.zeros(channels, dtype=torch.float64)
        self.sxy = torch.zeros(channels, dtype=torch.float64)

    @torch.no_grad()
    def hook(self, module, args, output):
        x = args[0].detach().reshape(-1, self.channels).double()
        y = output.detach().reshape(-1, self.channels).double()
        self.n += x.shape[0]
        self.sx += x.sum(0).cpu()
        self.sy += y.sum(0).cpu()
        self.sxx += (x * x).sum(0).cpu()
        self.sxy += (x * y).sum(0).cpu()

    def coefficients(self):
        if not self.n:
            raise RuntimeError("No LayerNorm samples received")
        vx = (self.sxx - self.sx * self.sx / self.n).clamp_min(0)
        cov = self.sxy - self.sx * self.sy / self.n
        gain = cov / vx.clamp_min(1e-10)
        bias = (self.sy - gain * self.sx) / self.n
        return gain.float(), bias.float()


class _ReLURegression(_AffineRegression):
    """Calibrate per-channel GELU targets against ordinary ReLU activations.

    Use a pre-hook so this works on both the initial nn.GELU and on an already
    converted CalibratedReLU during post-fine-tuning recalibration.
    """

    @torch.no_grad()
    def hook(self, module, args):
        preact = args[0].detach().reshape(-1, self.channels).float()
        x = torch.relu(preact).double()
        y = F.gelu(preact).double()
        self.n += x.shape[0]
        self.sx += x.sum(0).cpu()
        self.sy += y.sum(0).cpu()
        self.sxx += (x * x).sum(0).cpu()
        self.sxy += (x * y).sum(0).cpu()

    def coefficients(self):
        if not self.n:
            raise RuntimeError("No activation samples received")
        variance = (self.sxx - self.sx * self.sx / self.n).clamp_min(0)
        covariance = self.sxy - self.sx * self.sy / self.n
        # A channel with nearly constant ReLU output cannot identify a slope.
        gain = torch.where(variance > 1e-8, covariance / variance.clamp_min(1e-8),
                           torch.ones_like(variance)).clamp(0.0, 2.0)
        bias = (self.sy - gain * self.sx) / self.n
        return gain.float(), bias.float()


class _RangeSamples:
    """Lightweight representative quantile samples; never shipped in ONNX."""

    def __init__(self, heads=None, per_batch=2048, score_center="none"):
        self.heads = heads
        self.per_batch = per_batch
        self.score_center = score_center
        self.samples = []
        self.seen = 0

    @torch.no_grad()
    def hook(self, module, args):
        x = args[0].detach().float()
        if self.heads is not None:
            if self.score_center == "max":
                x = x - x.amax(dim=-1, keepdim=True)
            elif self.score_center == "mean":
                x = x - x.mean(dim=-1, keepdim=True)
        if self.heads is None:
            flat = x.reshape(-1)
            i = torch.linspace(0, flat.numel() - 1, min(self.per_batch, flat.numel()),
                               device=flat.device).long()
            self.samples.append(flat[i].cpu())
        else:
            # [B,H,Q,K] -> [H,B*Q*K], sample evenly per head.
            flat = x.permute(1, 0, 2, 3).reshape(self.heads, -1)
            i = torch.linspace(0, flat.shape[1] - 1, min(self.per_batch, flat.shape[1]),
                               device=flat.device).long()
            self.samples.append(flat[:, i].cpu())
        self.seen += 1

    def bounds(self, tail=0.001, margin=0.05):
        if not self.samples:
            raise RuntimeError("No calibration samples observed")
        values = torch.cat(self.samples, dim=-1)
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite calibration values")
        if not (0 <= tail < 0.5) or margin < 0:
            raise ValueError("Invalid calibration tail/margin")
        low = torch.quantile(values, tail, dim=-1)
        high = torch.quantile(values, 1.0 - tail, dim=-1)
        center = (low + high) / 2
        radius = (high - low).clamp_min(1e-4) * (0.5 + margin)
        return center - radius, center + radius


def _check_calib_loader(loader):
    if iter(loader) is loader:
        raise ValueError("Use a re-iterable DataLoader/list, not a one-shot iterator")


@torch.no_grad()
def fit_static_norms(teacher: StandardViT, calibration_loader, max_batches=None,
                     norm_mode=None):
    """Transfer trained RMSNorm weights exactly, or fit legacy LN affine."""
    _check_calib_loader(calibration_loader)
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    if norm_mode is None:
        norm_mode = "rms_matmul" if teacher.norm_type == "rms" else "matmul"
    if teacher.norm_type == "rms" and norm_mode != "rms_matmul":
        raise ValueError("RMSNorm teachers require norm_mode='rms_matmul'")
    if teacher.norm_type == "ln" and norm_mode not in {"matmul", "affine"}:
        raise ValueError("LayerNorm teachers require norm_mode='matmul' or 'affine'")
    device = next(teacher.parameters()).device
    teacher.eval()
    if norm_mode in {"matmul", "rms_matmul"}:
        student = NPUViT(**teacher.config, norm_mode=norm_mode).to(device)
        # avg MatMul buffers are nonpersistent; all trainable parameter keys match.
        student.load_state_dict(teacher.state_dict(), strict=True)
        return student.eval()
    stats = {}
    hooks = []
    for name, module in teacher.named_modules():
        if isinstance(module, nn.LayerNorm):
            stats[name] = _AffineRegression(module.normalized_shape[0])
            hooks.append(module.register_forward_hook(stats[name].hook))
    try:
        for idx, batch in enumerate(calibration_loader):
            if max_batches is not None and idx >= max_batches:
                break
            teacher(_as_patches(batch, teacher, device))
    finally:
        for hook in hooks:
            hook.remove()
    student = NPUViT(**teacher.config, norm_mode=norm_mode).to(device)
    student.load_state_dict(teacher.state_dict(), strict=True)
    for name, module in student.named_modules():
        if isinstance(module, StaticChannelNorm):
            gain, bias = stats[name].coefficients()
            module.weight.copy_(gain.to(device))
            module.bias.copy_(bias.to(device))
    return student.eval()


@torch.no_grad()
def calibrate_npu(
    student: NPUViT,
    calibration_loader,
    *,
    max_batches=None,
    softmax_segments=12,
    softmax_kind="mlp",
    mlp_hidden=6,
    mlp_fit_steps=150,
    mlp_fit_lr=0.0003,
    mlp_rows=1536,
    mlp_freeze=False,
    tail=0.001,
    margin=0.05,
    radius=3.0,
    score_center="mean",
    knot_spacing="geometric",
    verbose=True,
):
    """Convert GELU to calibrated ReLU; calibrate per-head Softmax approximation.

    mlp: warm-start a six-hinge ReLU network, fit its knot/slope coefficients to
         *current student attention scores* with an exact Softmax KL objective.
    pwl: keep the original fixed, 12-segment per-head approximation.
    native: keep exact Softmax for an NPU performance reference.
    """
    if softmax_kind not in {"mlp", "pwl", "native"}:
        raise ValueError("softmax_kind must be mlp, pwl, or native")
    _check_calib_loader(calibration_loader)
    student.eval()
    device = next(student.parameters()).device
    for idx, block in enumerate(student.blocks):
        relu_stats = _ReLURegression(block.fc1.out_features)
        score_stats = _RangeSamples(heads=block.attn.heads, score_center=score_center)
        row_stats = _ScoreRowSamples(max_rows=mlp_rows) if softmax_kind == "mlp" else None
        hooks = [block.act.register_forward_pre_hook(relu_stats.hook)]
        if softmax_kind != "native":
            hooks.append(block.attn.norm.register_forward_pre_hook(score_stats.hook))
        if row_stats is not None:
            hooks.append(block.attn.norm.register_forward_pre_hook(row_stats.hook))
        try:
            for bi, batch in enumerate(calibration_loader):
                if max_batches is not None and bi >= max_batches:
                    break
                student(_as_patches(batch, student, device))
        finally:
            for hook in hooks:
                hook.remove()
        gain, bias = relu_stats.coefficients()
        block.act = CalibratedReLU(gain, bias).to(device)
        if softmax_kind == "native":
            block.attn.norm = nn.Softmax(dim=-1)
            if verbose:
                print(f"  calibrated block {idx+1}/{len(student.blocks)}: ReLU only, native Softmax", flush=True)
            continue
        low_score, high_score = score_stats.bounds(tail, margin)
        if softmax_kind == "mlp":
            attn = LearnedMLPSoftmax(student.n_tokens, low_score, high_score,
                                     hidden=mlp_hidden, score_center=score_center,
                                     knot_spacing=knot_spacing).to(device)
            stats = fit_mlp_softmax(attn, row_stats.get(), steps=mlp_fit_steps,
                                    lr=mlp_fit_lr, verbose=verbose)
            if mlp_freeze:
                attn.freeze()
        else:
            attn = FixedPWLSoftmax(student.n_tokens, low_score, high_score,
                                   softmax_segments, radius, score_center=score_center,
                                   knot_spacing=knot_spacing).to(device)
        block.attn.norm = attn
        if verbose:
            print(f"  calibrated block {idx+1}/{len(student.blocks)}: "
                  f"{softmax_kind}({mlp_hidden if softmax_kind == 'mlp' else softmax_segments}), "
                  f"ReLU gain [{float(gain.min()):.3f}, {float(gain.max()):.3f}], "
                  f"attention widths {(high_score - low_score).tolist()}", flush=True)
    return student


@torch.no_grad()
def convert_to_npu(
    teacher: StandardViT,
    calibration_loader,
    *,
    max_batches=None,
    softmax_segments=12,
    softmax_kind="mlp",
    mlp_hidden=6,
    mlp_fit_steps=150,
    mlp_fit_lr=0.0003,
    mlp_rows=1536,
    mlp_freeze=False,
    tail=0.001,
    margin=0.05,
    norm_mode=None,
    score_center="mean",
    knot_spacing="geometric",
    verbose=True,
):
    """Transfer teacher weights, preserve RMSNorm, calibrate ReLU/Softmax."""
    student = fit_static_norms(teacher, calibration_loader, max_batches=max_batches,
                               norm_mode=norm_mode)
    return calibrate_npu(
        student, calibration_loader, max_batches=max_batches,
        softmax_segments=softmax_segments, softmax_kind=softmax_kind,
        mlp_hidden=mlp_hidden, mlp_fit_steps=mlp_fit_steps,
        mlp_fit_lr=mlp_fit_lr, mlp_rows=mlp_rows, mlp_freeze=mlp_freeze,
        tail=tail, margin=margin, score_center=score_center,
        knot_spacing=knot_spacing, verbose=verbose,
    )


@torch.no_grad()
def measure_saturation(student, validation_loader, max_batches=None):
    """Measure held-out PWL attention saturation and ReLU-vs-GELU MSE."""
    student.eval()
    data = {}
    handles = []
    for i, block in enumerate(student.blocks):
        if not isinstance(block.attn.norm, (FixedPWLSoftmax, LearnedMLPSoftmax, nn.Softmax)) or not isinstance(block.act, CalibratedReLU):
            raise ValueError("Model has not been fully calibrated")
        data[i] = {"attn_count": torch.zeros(block.attn.heads, dtype=torch.float64),
                   "attn_total": 0, "relu_sqerr": 0.0, "relu_total": 0}

        def score_hook(module, args, i=i):
            if isinstance(module, nn.Softmax):
                return
            x = module.center_scores(args[0])
            count = ((x < module.low) | (x > module.high)).sum(dim=(0, 2, 3))
            data[i]["attn_count"] += count.detach().double().cpu()
            data[i]["attn_total"] += x.shape[0] * x.shape[2] * x.shape[3]

        def relu_hook(module, args, output, i=i):
            target = F.gelu(args[0].detach())
            error = output.detach() - target
            data[i]["relu_sqerr"] += float(error.square().sum().item())
            data[i]["relu_total"] += error.numel()

        if not isinstance(block.attn.norm, nn.Softmax):
            handles.append(block.attn.norm.register_forward_pre_hook(score_hook))
        handles.append(block.act.register_forward_hook(relu_hook))
    try:
        for bi, batch in enumerate(validation_loader):
            if max_batches is not None and bi >= max_batches:
                break
            student(_as_patches(batch, student))
    finally:
        for handle in handles:
            handle.remove()
    if not data or not all(d["relu_total"] for d in data.values()):
        raise ValueError("No validation batches")
    return {
        i: {"attention_per_head": ((stats["attn_count"] / stats["attn_total"]).tolist()
                                   if stats["attn_total"] else None),
            "relu_vs_gelu_mse": stats["relu_sqerr"] / stats["relu_total"]}
        for i, stats in data.items()
    }


@torch.no_grad()
def fold_affine_norms(student: NPUViT):
    """Fold affine norms when available, always fold calibrated ReLU gain/bias.

    MatMulLayerNorm, MatMulRMSNorm and MatMulHeadRMSNorm remain dynamic.
    """
    if any(not isinstance(b.attn.norm, (FixedPWLSoftmax, LearnedMLPSoftmax, nn.Softmax)) or
           not isinstance(b.act, CalibratedReLU) for b in student.blocks):
        raise ValueError("Calibrate all ReLU and attention layers before folding")
    m = copy.deepcopy(student).eval()

    def fold(norm, *linears):
        w, b = norm.weight.detach(), norm.bias.detach()
        for linear in linears:
            old = linear.weight.detach().clone()
            linear.bias.add_(old @ b)
            linear.weight.mul_(w[None, :])

    for block in m.blocks:
        if isinstance(block.norm1, StaticChannelNorm):
            fold(block.norm1, block.attn.q, block.attn.k, block.attn.v)
            block.norm1 = nn.Identity()
        if isinstance(block.norm2, StaticChannelNorm):
            fold(block.norm2, block.fc1)
            block.norm2 = nn.Identity()
        # GELU -> calibrated ReLU, with gain+bias absorbed into the output Linear.
        gain, bias = block.act.gain.detach(), block.act.bias.detach()
        old_weight = block.fc2.weight.detach().clone()
        block.fc2.bias.add_(old_weight @ bias)
        block.fc2.weight.mul_(gain[None, :])
        block.act = nn.ReLU()
    # Mean over tokens commutes with any per-channel affine map.
    if isinstance(m.norm, StaticChannelNorm):
        fold(m.norm, m.head)
        m.norm = nn.Identity()
    return m


XINT8_LISTED = {
    "Add", "AveragePool", "BatchNormalization", "Concat", "Constant", "ConstantOfShape",
    "Conv", "DepthToSpace", "Div", "Expand", "Flatten", "Gemm", "GlobalAveragePool",
    "Identity", "InstanceNormalization", "MatMul", "Max", "MaxPool", "Min", "Mul",
    "Pad", "Reciprocal", "ReduceMax", "Reshape", "Shape", "Slice", "Sqrt",
    "Squeeze", "Sub", "Transpose", "Unsqueeze",
}


def export_onnx(model, path):
    """Export folded fixed-PWL student (never the normal teacher)."""
    folded = fold_affine_norms(model)
    parameter = next(folded.parameters())
    dummy = torch.randn(1, folded.n_tokens, folded.patch_dim, device=parameter.device,
                        dtype=parameter.dtype)
    kwargs = dict(opset_version=17, input_names=["input"], output_names=["logits"])
    try:
        torch.onnx.export(folded, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(folded, dummy, str(path), **kwargs)
    return str(path)


def audit_onnx(path, allow_native_softmax=False):
    import onnx
    ops = collections.Counter(node.op_type for node in onnx.load(str(path)).graph.node)
    print(f"\nExported ONNX operators in {path}:")
    for op, count in sorted(ops.items()):
        note = "listed XINT8" if op in XINT8_LISTED else "confirm NPU support"
        print(f"  {op:20s} {count:4d}   {note}")
    forbidden = {"LayerNormalization", "Softmax", "Exp", "BatchNormalization", "Gelu", "Erf"}
    if allow_native_softmax:
        forbidden.remove("Softmax")
    if ops.keys() & forbidden:
        raise AssertionError(f"Unexpected ops in exported NPU graph: {ops.keys() & forbidden}")
    return ops


def selftest():
    torch.manual_seed(42)
    images = torch.randn(5, 3, 32, 32)
    assert np.allclose(patchify(images, 8).numpy(), patchify_np(images.numpy(), 8))
    teacher = StandardViT(dim=32, depth=2, heads=4, patch=8)
    patches = patchify(images, 8)
    teacher.eval()
    calib = [images[:3], images[2:]]
    student = convert_to_npu(teacher, calib, softmax_kind="mlp", mlp_hidden=6,
                             mlp_fit_steps=15, mlp_rows=64, score_center="mean",
                             knot_spacing="geometric")
    original = teacher(patches)
    adapted = student(patches)
    folded = fold_affine_norms(student)(patches)
    print("teacher/student max logit difference:", (original - adapted).abs().max().item())
    print("folded/student max difference:", (folded - adapted).abs().max().item())
    assert torch.isfinite(adapted).all()
    assert (folded - adapted).abs().max() < 1e-5
    for block in student.blocks:
        assert not any(isinstance(m, nn.LayerNorm) for m in block.modules())
        assert isinstance(block.act, CalibratedReLU)
        assert isinstance(block.attn.norm, LearnedMLPSoftmax)
        assert block.attn.norm.hidden == 6
        test_s = torch.randn(2, block.attn.heads, student.n_tokens, student.n_tokens)
        a = block.attn.norm(test_s)
        assert torch.isfinite(a).all()
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
        assert not any(p.requires_grad for p in block.act.parameters())
        assert any(p.requires_grad for p in block.attn.norm.parameters())
    print("held-out saturation:", measure_saturation(student, calib))
    # A refresh must be repeatable after fine-tuning.
    calibrate_npu(student, calib, softmax_kind="mlp", mlp_fit_steps=10,
                  mlp_rows=64, verbose=False)
    assert all(isinstance(b.act, nn.ReLU) for b in fold_affine_norms(student).blocks)
    print("model self-test: OK")


if __name__ == "__main__":
    selftest()
