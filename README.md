# DyT, HardDyT and RMSNorm ViT for Ryzen AI

Train a floating-point ViT with your choice of normalization, GELU and exact Softmax,
then convert it to an NPU-oriented student using **calibrated ReLU** and a
**learned per-head six-neuron ReLU MLP in place of the Softmax exponential**.

To evaluate the exported graph on AMD XDNA1, check ONNX validity, INT8
accuracy, and execution provider partitioning.

## Design

| Component | FP32 teacher | NPU-oriented student |
|---|---|---|
| Block and final normalization | RMSNorm (default), LayerNorm, DyT, or HardDyT | constant-MatMul RMSNorm, LayerNorm export, or the trained pointwise activation |
| Query and Key | per-head RMSNorm (default), DyT, HardDyT, or disabled | constant-MatMul RMSNorm, the trained pointwise activation, or passthrough |
| Activation | GELU | calibrated `gain * ReLU + bias`; gain/bias folded into `fc2` |
| Attention scores | `Q @ K.T / sqrt(d)` | unchanged |
| Attention normalization | exact Softmax | **learned six-neuron ReLU MLP + sum/division** (default) |
| Other attention choices | — | fixed 12-segment PWL (`pwl`) or native Softmax (`native`) |
| Row centering | internal to Softmax | mean via constant MatMul (default), optional max/none |

### Learned six-neuron MLP

For each head `h`, fit a scalar-output ReLU network:

```
f_h(z) = base_h + sum_{i=1}^6 coef_{h,i} * ReLU(z + bias_{h,i})
p_h(z_j) = max(f_h(z_j), eps) / sum_k max(f_h(z_k), eps)
z = scores - row_mean(scores)                    # the default
```

The input weight of each ReLU neuron is constrained to `+1` to reduce NPU
arithmetic. **The six hidden biases (knot positions), six output coefficients
and output bias ARE learned** per head: **13 parameters per head**, or 312
across 6 blocks × 4 heads. This is a constrained `1 → 6 → 1` ReLU MLP, and
is a learned continuous PWL function. It deliberately does not use a
full unconstrained `Linear(1, 6)` layer: that would require six additional
per-score input multiplications. A row-wise sum and division are still
required, just as with ordinary Softmax.

Each head is initialized from a six-segment geometric interpolation of
`exp(score - calibrated_high)`, then fitted by minimizing the **KL divergence
between exact Softmax and MLP probabilities over *real calibration attention
rows***. Score rows come from the student as preceding layers are converted,
and the target is exact Softmax **on those same rows**; afterward, full-model
teacher distillation helps recover classification accuracy. The fitting loop
retains its lowest held-out-calibration-row KL parameters. Coefficients are
learnable during global fine-tuning by default, or can be held fixed with
`--mlp-freeze-after-fit`.

**Inference does not calculate exponential or row-wise max** when using
`--softmax-kind mlp --score-center mean`. The exported ONNX model uses basic
`Add`, `Sub`, `Mul`, `Relu`, `Max`, `Min`, `MatMul`, `Div`, etc. It still has
`Sqrt`/`Div` when it includes dynamic RMSNorm or QK-RMSNorm. The DyT options
below change both normalization sites independently.

## Train and export with DyT or HardDyT

```bash
uv run npu-vit-train --stage all --norm-type harddyt --qk-norm-type harddyt \
  --softmax-kind mlp --score-center mean --out out_harddyt
```

`--norm-type` selects the transforms before attention, before the feed-forward
network, and before final pooling. `--qk-norm-type` selects the transforms on
the projected queries and keys. Each Q and K head learns an independent
input scale and channel affine. Pointwise layers initialize the input scale
to `0.5`, gain to `1`, and bias to `0`.

Use `--norm-type dyt --qk-norm-type rms` to test smooth DyT at the token sites.
Use `--norm-type harddyt --qk-norm-type rms` to test clipping at those same
sites. Compare against `--norm-type harddyt --qk-norm-type harddyt` to measure
the additional effect of changing Q/K. Choose `--softmax-kind native` to
hold exact Softmax fixed for those comparisons. `--no-qk-norm` bypasses the
Q/K transforms entirely.

Train a teacher for each normalization configuration. With separate `train`
and `adapt` runs, pass the teacher checkpoint to `adapt` and leave
`--norm-mode` at `auto`. The checkpoint records both normalization choices.

During export, Q/K input scales fold into the Q/K projection weights and
biases. The token transforms' output gains and offsets fold into their
consuming projections, including the final classifier. HardDyT uses ONNX
`Clip`. With HardDyT at both sites and mean-centered MLP attention, the
exported graph avoids `Tanh`, `Exp`, `Sqrt`, `Rsqrt`, `ReduceMean`, and
`ReduceMax`. Check `Clip` support and partitioning on the target NPU.

## Vectorized inference (no per-neuron `for` loop)

The previous implementation constructed six separate ReLU branches in a Python
`for` loop. This release packs each head's six learned biases and coefficients
into `[1,H,1,6]` and `[1,H,6,1]` tensors, respectively. A **single batched
MatMul** evaluates the six hidden-unit outputs:

```python
# scores: [B, H, Q, K]
z = self.center_scores(scores)
z = torch.minimum(torch.maximum(z, self.low), self.high)
b, h, q, k = z.shape
hidden = torch.relu(z.reshape(b, h, q * k, 1) + self.biases)
values = (self.base + hidden @ self.coefs).reshape(b, h, q, k)
e = torch.maximum(values, self.min_value)
return e / (e @ self.ones)
```

There is **one hidden-layer ReLU** and **one projection MatMul** for all six
neurons, followed by row normalization. There is no loop in `forward`. The
fixed PWL implementation is vectorized the same way. Training/calibration
loops still exist because gradient updates and dataset iteration are inherently
iterative.

The six-neuron learned function, parameters per head, and per-head calibration
are unchanged; floating-point reduction order may cause very small numerical
differences. This reduces explicit inference graph nodes but **does not prove
lower XDNA1 latency**. In particular, verify that the 4D batched MatMul is
assigned to the NPU after INT8 quantization.

### Reuse previous checkpoints without retraining

Old `vit_npu.pt` and `vit_npu_best.pt` files (with `biases.0`/`coefs.0` etc.)
are automatically upgraded when loading, without changing their learned values.
Legacy `vit_npu_latest.pt` AdamW moments are also migrated for
`--resume-finetune`. To export a previously adapted model using the vectorized
inference graph:

```bash
uv run npu-vit-train --stage export \
  --student-ckpt out_mlp6/vit_npu.pt \
  --out out_mlp6_vectorized

uv run npu-vit-quantize \
  --model out_mlp6_vectorized/vit_npu.onnx \
  --data out_mlp6 --quantizer ort --npu off
```

The original `out_mlp6/` retains your preprocessing arrays; no calibration or
fine-tuning is required for this *representation-only* conversion.

## Installation

Requires Python 3.10–3.13; the core package needs PyTorch and NumPy.

```bash
uv sync --extra train --extra export --extra test
uv run pytest -q
uv run npu-vit-selftest
```

## Stage 1 — Train the normal teacher (once)

```bash
uv run npu-vit-train --stage train \
  --norm-type rms --qk-norm \
  --epochs 100 --out out_rms_qk
```

Output: `out_rms_qk/vit_teacher.pt`. You can **reuse your already trained
RMSNorm + QK-Norm checkpoint**, so you do not need to train the teacher again.
An older LayerNorm-only teacher is loaded as a legacy LN model, not silently
reinterpreted as RMSNorm.

## Stage 2 — Fit six-neuron MLP attention and fine-tune

```bash
uv run npu-vit-train --stage adapt \
  --teacher-ckpt out_rms_qk/vit_teacher.pt \
  --out out_mlp6 \
  --norm-mode rms_matmul \
  --softmax-kind mlp --mlp-hidden 6 \
  --mlp-fit-steps 150 --mlp-fit-lr 0.0003 --mlp-rows 1536 \
  --score-center mean --knot-spacing geometric \
  --finetune-epochs 10
```

Conversion is blockwise: preserve RMSNorm/QK-Norm math; calibrate ReLU from
GELU preactivations; collect real attention score rows; initialize and learn
six-neuron ReLU coefficients against exact Softmax; then adapt the entire
student with cross-entropy, teacher-logit KL and optional block-feature MSE.

The PWL/MLP coefficients may change during the final fine-tuning stage.
After fine-tuning, the code tries refreshed ReLU/MLP calibration and keeps
**the better of the pre-refresh and refreshed validation accuracy**.

If you want to freeze the learned MLP during full-model fine-tuning, add
`--mlp-freeze-after-fit`.

### Checkpoint saving / resuming

- `out_mlp6/vit_npu_best.pt` — best student before final refresh.
- `out_mlp6/vit_npu_latest.pt` — saved after each epoch with optimizer,
  completed epoch, best accuracy and RNG state. Also saved at epoch 0.
- `out_mlp6/vit_npu.pt` — best final model after optional coefficient refresh.

Resume using the **same output folder and configuration**:

```bash
uv run npu-vit-train --stage adapt \
  --teacher-ckpt out_rms_qk/vit_teacher.pt \
  --out out_mlp6 --resume-finetune \
  --softmax-kind mlp --mlp-hidden 6 \
  --score-center mean --knot-spacing geometric \
  --finetune-epochs 10
```

`--finetune-epochs` is the total target, not additional epochs. Recovery is
at epoch boundaries, and a running process cannot pick up source edits until
restarted.

## Stage 3 — Export and quantize

```bash
uv run npu-vit-train --stage export --out out_mlp6

uv run npu-vit-quantize --model out_mlp6/vit_npu.onnx \
  --data out_mlp6 --quantizer ort --npu off
```

`calib.npy`, `test_x.npy`, and `test_y.npy` are produced **at the beginning of
`--stage adapt`**, before lengthy calibration/fine-tuning, and reproduced at
completion. If they are missing, generate them *without any retraining*:

```bash
uv run npu-vit-train --stage data \
  --teacher-ckpt out_rms_qk/vit_teacher.pt --out out_mlp6
```

Keep the same data split/preprocessing and input patch size. By default the
calibration array is an unaugmented subset of CIFAR-10 training images.
For quantization on AMD hardware, use a compatible Quark/Ryzen AI environment:

```bash
uv run npu-vit-quantize --model out_mlp6/vit_npu.onnx \
  --data out_mlp6 --quantizer quark --config XINT8 --npu require
```

The ORT INT8 CPU path is *not* proof of Ryzen AI NPU compatibility or speed.

## Controlled comparisons: keep the teacher fixed

**Fixed 12-segment PWL**, to compare with the learned six-neuron MLP:

```bash
uv run npu-vit-train --stage adapt \
  --teacher-ckpt out_rms_qk/vit_teacher.pt --out out_pwl12 \
  --softmax-kind pwl --softmax-segments 12 \
  --score-center mean --knot-spacing geometric --finetune-epochs 10
```

**Native Softmax**, as the hardware/compiler reference (exports a native
`Softmax` node; does NOT try to eliminate it):

```bash
uv run npu-vit-train --stage adapt \
  --teacher-ckpt out_rms_qk/vit_teacher.pt --out out_native \
  --softmax-kind native --finetune-epochs 10
```

Compare all three on the same validation/test split: teacher/student accuracy,
attention KL, clipped-score rates, post-quantization accuracy, median batch-1
latency, and NPU vs CPU assignment. A learned six-neuron model is **not
necessarily** faster or more accurate than an optimized native Softmax.

## Quick smoke tests (no dataset downloads)

```bash
uv run npu-vit-train --stage all --fake-data \
  --epochs 1 --finetune-epochs 1 --dim 32 --depth 1 --patch 8 \
  --heads 4 --batch 16 --workers 0 --fake-samples 64 \
  --val-samples 8 --calib-samples 32 --mlp-fit-steps 10 --mlp-rows 64 \
  --out smoke
```

This validates the pipeline, but **fake-data accuracy has no meaning**.

## Repo contents

```
src/npu_friendly_vit/
  __init__.py
  model.py        teacher/student, six-neuron ReLU MLP, calibration, folding, ONNX
  train.py        FP32 training, conversion, learned fitting, distillation, checkpoints
  quantize.py     ONNX Runtime INT8 / AMD Quark quantization and evaluation
tests/
  test_model.py
  test_train.py
  test_quantize.py
pyproject.toml
```

The test suite checks RMSNorm and QK-Norm equivalence, learned-MLP
probability normalization and fitting, old fixed-PWL and native branches,
ReLU folding, checkpoint persistence/resume, and smoke training. Optional
`onnx` and `onnxruntime` are required to validate the actual ONNX graph;
without them those tests skip. We have not measured real CIFAR-10 or NPU
inference results for the learned-MLP student.
