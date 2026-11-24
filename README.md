# RMSNorm + QK-Norm ViT for Ryzen AI experiments

Train a conventional transformer first, then calibrate and distill into a graph
with ordinary ReLU and **fixed per-head PWL Softmax**. This is a PyTorch / ONNX
research implementation, **not** a guarantee of hardware INT8 compilation.

## What changed from the LayerNorm prototype

| Stage | Original LayerNorm branch | New default architecture |
| --- | --- | --- |
| FP32 teacher normalization | `LayerNorm` | **RMSNorm** over embedding channels |
| Query / Key | unnormalized Q and K | **per-head QK-Norm** (separate Q and K RMSNorm) |
| Teacher activation | GELU | GELU |
| Teacher attention | exact Softmax | exact Softmax |
| Student normalization | MatMul-LayerNorm / static affine | **MatMul-RMSNorm** + **MatMul QK-Norm** |
| Student activation | calibrated ReLU | calibrated ReLU; gain/bias folded into `fc2` |
| Student attention | calibrated PWL | fixed per-head geometric-knot PWL |
| Score centering | none / mean / max | **mean (default)**; max optional for accuracy |

**Important:** RMSNorm and QK-Norm must be learned by a fresh FP32 training
run. Your old `out/vit_teacher.pt` contains a **LayerNorm teacher** and cannot
be relabeled or converted into a trained RMSNorm + QK-Norm teacher. The loader
recognizes legacy checkpoints, keeping them in the LN comparison branch.

## Architecture

Teacher: `patchify -> Linear -> [RMSNorm -> QK-Norm attention with
Softmax -> residual -> RMSNorm -> GELU MLP -> residual] x depth -> RMSNorm ->
mean pool -> classifier`.

Student: same learned projections and normalized Q/K, but reductions are
expressed via constant MatMul; GELU is calibrated into `gain * ReLU(x) + bias`
and gain/bias fold into `fc2`; Softmax uses fixed per-head PWL coefficients and
sum normalization via constant MatMul + Div.

For a tensor `x: [B,N,C]`, RMSNorm is:

```
x * gamma / sqrt( (x*x) @ ones(C,1)/C + eps )
```

QK-Norm applies the same idea to `Q,K: [B,H,N,D]` with **separate trainable
per-head `[H,D]` gains** before the score product. Q/K gains remain learnable
in the student; the calibration coefficients for ReLU/PWL are frozen.

**Hardware note:** MatMul-RMSNorm still uses `Sqrt`, `Div`, `Mul`, and the
constant reduction `MatMul`. They may be executed as NPU or CPU ops depending
on your quantizer/compiler and partitioning. This model removes the *native*
LayerNormalization, GELU and Softmax ONNX operators, but doesn't eliminate
dynamic RMS normalization.

## Install

Requirements: Python 3.10–3.13, PyTorch and NumPy. Use `uv` for convenience:

```bash
uv sync --extra train --extra export --extra test
uv run pytest -q
uv run npu-vit-selftest
```

For CPU-only tests without downloading CIFAR-10:

```bash
uv run npu-vit-train --stage all --fake-data --epochs 1 \
  --finetune-epochs 1 --dim 32 --depth 1 --patch 8 \
  --heads 4 --batch 16 --workers 0 --fake-samples 64 \
  --val-samples 8 --calib-samples 32 --out smoke
```

`--fake-data` checks the pipeline only; its accuracy is not meaningful.

## Train the new RMSNorm + QK-Norm teacher

```bash
uv run npu-vit-train --stage train --epochs 100 \
  --norm-type rms --qk-norm --out out_rms_qk
```

This trains `RMSNorm + QK-Norm + GELU + exact Softmax`, then saves
`out_rms_qk/vit_teacher.pt`.

## Calibrate and fine-tune the NPU student

```bash
uv run npu-vit-train --stage adapt \
  --teacher-ckpt out_rms_qk/vit_teacher.pt \
  --out out_rms_qk \
  --norm-mode rms_matmul \
  --score-center mean \
  --knot-spacing geometric \
  --softmax-segments 12 \
  --finetune-epochs 10
```

Conversion steps:

1. Copy teacher parameters into the student **without changing RMSNorm/QK-Norm
   mathematics**; their reductions are expressed through constant MatMul.
2. Observe per-channel GELU inputs; regress `GELU(x)` against `ReLU(x)` to
   obtain frozen per-channel gain and bias.
3. Observe attention scores *after QK-Norm* on calibration data, with the
   selected centering; fit per-head fixed PWL exponential coefficients.
4. Fine-tune using ground-truth cross-entropy and teacher-logit/feature KD.
5. Recalibrate PWL and ReLU coefficients after fine-tuning; **restore the
   previous coefficients if validation accuracy deteriorates**.
6. Save `vit_npu.pt`, `calib.npy`, `test_x.npy`, `test_y.npy`, and, when ONNX
   is installed, `vit_npu.onnx`.

To favor accuracy when mean-centering underperforms, repeat the adaptation with:

```bash
uv run npu-vit-train --stage adapt \
  --teacher-ckpt out_rms_qk/vit_teacher.pt --out out_rms_qk_max \
  --score-center max --knot-spacing geometric \
  --softmax-segments 12 --finetune-epochs 10
```

**Default `mean` score centering** computes a row-wise mean using a fixed
MatMul, rather than row-wise `ReduceMax`. Exact Softmax is shift invariant,
but clipped/approximated PWL is imperfect. The `max` fallback inserts a
`ReduceMax`; compare actual NPU partitioning and validation accuracy.

Calibrated bounds are quantiles, not guaranteed extrema. Validate held-out
out-of-range rates and classifier accuracy. Segments/placement are not yet
optimized for any specific NPU compiler.

## Export and quantize

```bash
uv run npu-vit-train --stage export --out out_rms_qk
uv run npu-vit-quantize --model out_rms_qk/vit_npu.onnx \
  --data out_rms_qk --quantizer ort --npu off
```

On a properly installed AMD Ryzen AI machine, use your matching Quark release:

```bash
uv run npu-vit-quantize --model out_rms_qk/vit_npu.onnx \
  --data out_rms_qk --quantizer quark --config XINT8 --npu require
```

The portable ORT INT8 path does **not** prove Ryzen AI `XINT8` support. Check
ONNX ops, compiler logs, NPU/CPU partitions, accuracy and latency.

## Compare against earlier architectures

```bash
# Legacy LayerNorm + no QK-Norm, with dynamic MatMul-LayerNorm
uv run npu-vit-train --stage train --norm-type ln --no-qk-norm \
  --epochs 100 --out out_ln
uv run npu-vit-train --stage adapt --teacher-ckpt out_ln/vit_teacher.pt \
  --norm-mode matmul --score-center max --out out_ln

# RMSNorm without QK-Norm (ablation)
uv run npu-vit-train --stage train --norm-type rms --no-qk-norm \
  --epochs 100 --out out_rms_only
```

An old LN checkpoint (without `norm_type` metadata) can still be loaded for
legacy comparisons; it is **never silently turned into RMSNorm/QK-Norm**.

## Files

```
src/npu_friendly_vit/
  model.py       teacher/student layers, calibration, folding, ONNX auditing
  train.py       CIFAR-10 training, adaptation and KD CLI
  quantize.py    ORT INT8 + AMD Quark NPU CLI
tests/
  test_model.py
  test_train.py
  test_quantize.py
pyproject.toml
```

## Testing status

Run `uv run pytest -q`. PyTorch tests verify exact FP32 RMSNorm/QK-Norm transfer,
PWL normalization, ReLU folding, checkpoint reload, and a fake-data train/adapt
smoke test. ONNX/ORT checks require optional `onnx` and `onnxruntime` packages.

No real CIFAR-10 benchmark or Ryzen AI NPU execution is claimed for this new
RMSNorm + QK-Norm branch. Earlier LayerNorm checkpoints/accuracy are not
comparable until both branches are trained and evaluated on the same split.
