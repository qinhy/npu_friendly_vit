# npu_friendly_vit

A small, conv-free, NPU-friendly Vision Transformer (`NPUViT`) designed for AMD Ryzen AI (Hawk Point / XDNA1, `XINT8` flow).

## Design

Every component is chosen so the exported ONNX graph contains only basic NPU-friendly tensor/arithmetic operations (`MatMul`, `Add`, `Mul`, `Sub`, `Div`, `Max`, `ReduceMax`, `Relu`, `Reshape`, `Transpose`, plus constants) with no `Conv`, no `BatchNormalization`, and no pooling op:

| Standard ViT | `NPUViT` replacement | Notes |
| --- | --- | --- |
| Conv patch embedding | CPU `patchify` / `patchify_np` + `nn.Linear` | Pure data rearrangement on CPU, `MatMul` on NPU |
| `LayerNorm` | `BNLast` (`BatchNorm1d` over channels) | Folded into adjacent `Linear` weights at ONNX export (`fold_batchnorms`) |
| `GELU` | `ReLU` | Elementwise activation |
| `Softmax` | `PWLSoftmax` (7-knot piecewise-linear `exp(z)` on `[-5.5, 0]`) | Built from `ReduceMax`, `Sub`, `Max`, `Add`, `ReLU`, `Mul`, `MatMul`, `Div`; also supports `pwl_learn`, `relu`, and exact `softmax` baselines |
| Mean pooling | Constant `MatMul` with `[1, N]` vector of `1/N` | Replaces `ReduceMean` / `GlobalAveragePool` |
| Fused QKV `Linear` | Separate `q`, `k`, `v` `nn.Linear` layers | Avoids `Split` ops in `[B, N, C]` token layout |

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python `>=3.10,<3.14` (pinned to `3.12` in `.python-version`)

## Installation

```bash
uv sync
```

## Quick Start

### 1. Run the self-test

Verifies `PWLSoftmax` accuracy, `patchify` vs `patchify_np` parity, exact BatchNorm folding, ONNX export, op audit, and ONNX Runtime parity:

```bash
uv run npu-vit-selftest
```

### 2. Train on CIFAR-10 and export ONNX

```bash
# Train NPU-friendly PWLSoftmax model
uv run npu-vit-train --attn pwl --epochs 100

# Optional: train exact-softmax baseline for comparison
uv run npu-vit-train --attn softmax --epochs 100

# Quick smoke test without downloading CIFAR-10
uv run npu-vit-train --fake-data --epochs 1
```

Outputs are written to `out/` by default (`npu_vit_<attn>.pt`, `npu_vit_<attn>.onnx`, `calib.npy`, `test_x.npy`, `test_y.npy`).

### 3. Quantize and evaluate

On a machine without Ryzen AI, test the INT8 quantization pipeline with ONNX Runtime (`--quantizer ort`):

```bash
uv run npu-vit-quantize --model out/npu_vit_pwl.onnx --data out --quantizer ort
```

On an AMD Ryzen AI machine (inside the Ryzen AI conda environment, e.g. `conda activate ryzen-ai-1.8.0` with `RYZEN_AI_INSTALLATION_PATH` set and AMD Quark installed):

```bash
python -m npu_friendly_vit.quantize --model out/npu_vit_pwl.onnx --data out --config XINT8
python -m npu_friendly_vit.quantize --model out/npu_vit_pwl.onnx --data out --config A8W8
```

## Development

```bash
# Run linter and formatter checks
uv run ruff check .
uv run ruff format --check .

# Run test suite
uv run pytest
```

## Project Structure

```text
.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/ci.yml
├── src/npu_friendly_vit/
│   ├── __init__.py
│   ├── model.py       # NPUViT, PWLSoftmax, BN folding, ONNX export & audit
│   ├── train.py       # CIFAR-10 training and preprocessing export
│   └── quantize.py    # AMD Quark / ORT INT8 quantization & NPU evaluation
├── tests/
│   ├── test_model.py
│   ├── test_quantize.py
│   └── test_train.py
├── .python-version
├── LICENSE
├── pyproject.toml
└── uv.lock
```

## License

[MIT](LICENSE)
