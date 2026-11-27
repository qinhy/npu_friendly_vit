# DyT, HardDyT and RMSNorm ViT for Ryzen AI

Train a floating-point ViT with configurable normalization and FFN
activations, then export it to ONNX. **The recommended starting point for
accelerator deployment is to use both training ramps and keep native
Softmax.** This recipe preserved accuracy with the default CPU INT8
quantization. Ryzen AI NPU execution and latency still need to be measured.

## Recommended export recipe

Ramp DyT to HardDyT at the block, final, and per-head Q/K sites. In parallel,
ramp GELU to plain ReLU in the FFNs. With both fractions set to `0.8`, the
last two of ten teacher epochs train with pure HardDyT and ReLU. Native
Softmax stays active throughout. The exported student carries those trained
activations through and skips ReLU and attention MLP calibration.

Train and export from scratch in Windows Command Prompt.

```cmd
uv run npu-vit-train --stage all --epochs 10 --finetune-epochs 0 ^
  --norm-type harddyt --qk-norm-type harddyt ^
  --dyt-hardening-fraction 0.8 --gelu-relu-ramp-fraction 0.8 ^
  --softmax-kind native --out out_harddyt_native
```

If you already trained a two-ramp teacher, reuse its checkpoint to export
the native Softmax variant.

```cmd
uv run npu-vit-train --stage adapt ^
  --teacher-ckpt out_harddyt_both_ramps/vit_teacher.pt ^
  --softmax-kind native --finetune-epochs 0 --out out_harddyt_native
```

Quantize and benchmark on CPU with ONNX Runtime.

```cmd
uv run npu-vit-quantize --model out_harddyt_native/vit_npu.onnx ^
  --data out_harddyt_native --quantizer ort --npu off --eval-n 2000 ^
  --latency-runs 200 --report out_harddyt_native/ort_cpu_benchmark.json
```

### CIFAR-10 results

The two-ramp teacher reached **62.32% validation accuracy**. On the full test
set, the native Softmax export matched the teacher at **61.41%** with zero
student fine-tuning epochs. The learned MLP student reached **62.31%** after
three fine-tuning epochs. Both variants used byte-identical calibration and
saved test arrays for the CPU quantization comparison.

| Attention | FP32 accuracy on 2,000 images | INT8 accuracy on 2,000 images | INT8 change |
|---|---:|---:|---:|
| Native Softmax | 60.65% | **60.65%** | 0.00 pp |
| Six-neuron learned MLP | 61.05% | 50.95% | −10.10 pp |

Native Softmax averaged **4.21 ms in FP32** and **4.99 ms in INT8** at batch 1
in this CPU run. The different fine-tuning budgets make the FP32-to-INT8
change within each variant the more informative comparison.

In the learned MLP, INT8 rounding erased small positive attention weights
before row normalization. A 128-image probe found zero denominators in
**9.7%** of final-block attention rows. Bypassing INT8 rounding for the small
weights and row sums recovered **61.15%** accuracy on the 2,000-image subset.

### Accelerator compatibility

The native export uses ONNX opset 17 and exposes attention as
`MatMul → scale → Softmax → MatMul`. HardDyT runs upstream on Q and K.
TensorRT can fuse compatible graphs with this pattern. Standard ONNX
introduced a dedicated `Attention` operator in opset 23, and the QNN
execution provider lists it for opsets 23–24. Ryzen AI lists native
`Softmax` for A16W8, A8W8, and XINT8 and documents a separate
`GroupQueryAttention` path for ONNX Runtime GenAI. Check the target
compiler's partition report for operator placement and its compiled graph
or build logs for attention fusion. [TensorRT fused attention](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/transformers-fused-attention.html),
[ONNX Attention](https://onnx.ai/onnx/operators/onnx__Attention.html#attention-23),
[QNN operator support](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md),
[Ryzen AI operator support](https://ryzenai.docs.amd.com/en/latest/ops_support.html)

## Design

| Component | FP32 teacher | NPU-oriented student |
|---|---|---|
| Block and final normalization | RMSNorm (default), LayerNorm, DyT, or HardDyT | constant-MatMul RMSNorm, LayerNorm export, or the trained pointwise activation |
| Query and Key | per-head RMSNorm (default), DyT, HardDyT, or disabled | constant-MatMul RMSNorm, the trained pointwise activation, or passthrough |
| FFN activation | GELU, or a ramp from GELU to ReLU | calibrated ReLU for GELU teachers, plain ReLU for ramp-trained teachers |
| Attention scores | `Q @ K.T / sqrt(d)` | unchanged |
| Attention normalization | exact Softmax | native Softmax (recommended here), learned six-neuron ReLU MLP (CLI default), or fixed 12-segment PWL |
| Row centering | internal to Softmax | approximate attention defaults to mean via constant MatMul, with optional max/none |

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

### Gradually switch from DyT to HardDyT

```bash
uv run npu-vit-train --stage all --epochs 10 --finetune-epochs 3 \
  --norm-type harddyt --qk-norm-type harddyt --dyt-hardening-fraction 0.8 \
  --softmax-kind mlp --score-center mean --out out_harddyt_ramp
```

At each selected HardDyT site, teacher training uses
`(1 - t) * tanh(alpha * x) + t * clip(alpha * x, -1, 1)` before the learned
output affine. The clip weight `t` increases linearly from 0 to 1 over the
specified fraction of optimizer steps. With 10 epochs and a fraction of
`0.8`, the last two epochs use pure HardDyT. Checkpoint selection starts
once the blend reaches 1. Student calibration, fine-tuning, and export then
use pure HardDyT. The default fraction of `0` uses HardDyT from the first step.

### Gradually switch the FFN from GELU to ReLU

```bash
uv run npu-vit-train --stage all --epochs 10 --finetune-epochs 3 \
  --norm-type harddyt --qk-norm-type harddyt --gelu-relu-ramp-fraction 0.8 \
  --softmax-kind mlp --score-center mean --out out_harddyt_relu_ramp
```

The teacher uses `(1 - t) * GELU(x) + t * ReLU(x)` in each FFN, increasing
`t` linearly from 0 to 1 over the first 80% of optimizer steps. With 10
epochs, it trains with plain ReLU for the final two epochs and selects
checkpoints from that phase. Adaptation keeps the teacher's plain ReLU
and calibrates the attention MLP using attention scores from the
ReLU model. The option can also be used with `--stage train` and a separate
`--stage adapt` run, which reads the FFN activation from the teacher checkpoint.

Omit the flag to train the GELU teacher and use the original calibrated
ReLU conversion. The example uses HardDyT throughout so it can be
compared with the all-HardDyT run that uses calibrated ReLU.

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

**Native Softmax**, as the hardware/compiler reference for an RMSNorm teacher.

```bash
uv run npu-vit-train --stage adapt \
  --teacher-ckpt out_rms_qk/vit_teacher.pt --out out_native \
  --softmax-kind native --finetune-epochs 10
```

Use the same validation and test splits to compare accuracy before and after
quantization, mean batch-1 latency, and operator placement across CPU and NPU.
For approximate attention, also compare attention KL and clipped-score rates.

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
`onnx` and `onnxruntime` enable ONNX graph and runtime validation. See
[CIFAR-10 results](#cifar-10-results) for the measured CPU comparison.
