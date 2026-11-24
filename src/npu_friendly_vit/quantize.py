"""Quantize the *adapted* ViT ONNX graph and benchmark CPU / optional Ryzen AI NPU.

Examples:
    npu-vit-quantize --model out/vit_npu.onnx --data out --quantizer ort
    npu-vit-quantize --model out/vit_npu.onnx --data out --quantizer quark --config XINT8 --npu auto

The model must already be converted/fine-tuned using train.py. INT8 calibration
here is separate from the calibrated ReLU/attention PWL conversion in model.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import time

import numpy as np

try:
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader
except ImportError:  # Keep imports/reader unit tests usable without the quantization extra.
    ort = None

    class CalibrationDataReader:  # type: ignore[no-redef]
        pass


def _require_ort():
    if ort is None:
        raise RuntimeError("onnxruntime is required for quantization: pip install '.[quant]'")


class NpyReader(CalibrationDataReader):
    """Streaming reader of pre-patchified [S, N, patch_dim] FP32 arrays."""

    def __init__(self, calib: np.ndarray, input_name: str = "input"):
        self.calib = np.asarray(calib, dtype=np.float32)
        if self.calib.ndim != 3 or self.calib.shape[0] < 1:
            raise ValueError("Calibration data must be nonempty [samples, tokens, patch_dim]")
        if not np.isfinite(self.calib).all():
            raise ValueError("Calibration inputs must be finite")
        self.input_name = input_name
        self.rewind()

    def get_next(self):
        if self._index >= len(self.calib):
            return None
        item = {self.input_name: np.ascontiguousarray(self.calib[self._index : self._index + 1])}
        self._index += 1
        return item

    def rewind(self):
        self._index = 0


def quantize_quark(src, dst, calib, config="XINT8"):
    """Use the AMD Quark ONNX API present in the installed Ryzen AI environment."""
    _require_ort()
    try:
        from quark.onnx import ModelQuantizer, QConfig
    except ImportError:
        try:
            from quark.onnx import ModelQuantizer
            from quark.onnx.quantization.config import Config, get_default_config
        except ImportError as exc:
            raise RuntimeError("AMD Quark is not installed. Use --quantizer ort on a non-AMD machine.") from exc
        cfg = Config(global_quant_config=get_default_config(config))
    else:
        cfg = QConfig.get_default_config(config)
    ModelQuantizer(cfg).quantize_model(str(src), str(dst), NpyReader(calib))
    return str(dst)


def quantize_ort(src, dst, calib):
    """Portable static QDQ INT8 quantization for correctness testing, NOT an XINT8 guarantee."""
    _require_ort()
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    quantize_static(
        str(src), str(dst), NpyReader(calib),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    return str(dst)


def accuracy(sess, x: np.ndarray, y: np.ndarray, limit: int | None = None) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 3 or len(x) != len(y) or not len(x):
        raise ValueError("test_x must be [S,N,D] and test_y must have S labels")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    n = len(x) if limit is None else min(limit, len(x))
    name = sess.get_inputs()[0].name
    hits = 0
    for i in range(n):
        pred = sess.run(None, {name: x[i : i + 1]})[0].argmax(axis=-1).item()
        hits += int(pred == int(y[i]))
    return hits / n


def latency_ms(sess, x: np.ndarray, runs: int = 200, warmup: int = 10) -> float:
    if runs < 1 or warmup < 0 or not len(x):
        raise ValueError("runs >= 1, warmup >= 0 and nonempty test inputs are required")
    name = sess.get_inputs()[0].name
    inp = {name: x[:1]}
    for _ in range(warmup):
        sess.run(None, inp)
    start = time.perf_counter()
    for _ in range(runs):
        sess.run(None, inp)
    return (time.perf_counter() - start) * 1_000 / runs


def npu_session(model_path: str, cache_dir: str = "vaip_cache"):
    """Return (session, error); caller decides whether missing NPU is fatal."""
    _require_ort()
    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        return None, "VitisAIExecutionProvider not available"
    root = os.environ.get("RYZEN_AI_INSTALLATION_PATH")
    if not root:
        return None, "RYZEN_AI_INSTALLATION_PATH not set"
    hits = glob.glob(os.path.join(root, "voe-*", "xclbins", "phoenix", "4x4.xclbin"))
    if not hits:
        return None, "phoenix/4x4.xclbin not found in Ryzen AI installation"
    opts = {
        "target": "X1",
        "xclbin": hits[0],
        "cache_dir": os.path.abspath(cache_dir),
        "cache_key": Path(model_path).stem,
        "enable_cache_file_io_in_mem": "0",
    }
    so = ort.SessionOptions()
    so.log_severity_level = 1
    try:
        sess = ort.InferenceSession(
            str(model_path), sess_options=so,
            providers=["VitisAIExecutionProvider"], provider_options=[opts],
        )
    except Exception as exc:
        return None, f"NPU compilation/session failed: {exc}"
    if "VitisAIExecutionProvider" not in sess.get_providers():
        return None, "Vitis AI EP was not selected"
    return sess, None


def _load_arrays(data_dir: str):
    root = Path(data_dir)
    calib = np.load(root / "calib.npy", allow_pickle=False)
    tx = np.load(root / "test_x.npy", allow_pickle=False)
    ty = np.load(root / "test_y.npy", allow_pickle=False)
    NpyReader(calib)  # shape/finite checks
    if tx.ndim != 3 or ty.ndim != 1 or len(tx) != len(ty) or len(tx) == 0:
        raise ValueError("Expected nonempty test_x [S,N,D] and test_y [S]")
    if tx.shape[1:] != calib.shape[1:]:
        raise ValueError("Calibration and test patch dimensions do not match")
    if not np.isfinite(tx).all():
        raise ValueError("Test inputs must be finite")
    return calib, np.asarray(tx, dtype=np.float32), ty


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="out/vit_npu.onnx", help="adapted FP32 ONNX")
    p.add_argument("--data", default="out", help="directory with calibration/test .npy arrays")
    p.add_argument("--quantizer", choices=["ort", "quark"], default="ort")
    p.add_argument("--config", default="XINT8", help="Quark config, e.g. XINT8 or A8W8")
    p.add_argument("--npu", choices=["auto", "off", "require"], default="auto",
                   help="auto: skip if unavailable; off: CPU only; require: error if unavailable")
    p.add_argument("--eval-n", type=int, default=2000)
    p.add_argument("--latency-runs", type=int, default=50)
    p.add_argument("--report", default=None, help="optional JSON report path")
    args = p.parse_args(argv)
    if args.eval_n < 1 or args.latency_runs < 1:
        p.error("--eval-n and --latency-runs must be >= 1")
    _require_ort()
    src = Path(args.model)
    if not src.is_file():
        raise FileNotFoundError(f"Adapted ONNX file not found: {src}. Run --stage adapt/export first.")
    calib, tx, ty = _load_arrays(args.data)
    tag = args.config if args.quantizer == "quark" else "ortint8"
    dest = src.with_name(f"{src.stem}_{tag}.onnx")
    print(f"Quantizing {src} -> {dest} ({args.quantizer})", flush=True)
    if args.quantizer == "ort":
        quantize_ort(src, dest, calib)
    else:
        quantize_quark(src, dest, calib, args.config)

    cpu = ["CPUExecutionProvider"]
    f_sess = ort.InferenceSession(str(src), providers=cpu)
    q_sess = ort.InferenceSession(str(dest), providers=cpu)
    f_acc = accuracy(f_sess, tx, ty, args.eval_n)
    q_acc = accuracy(q_sess, tx, ty, args.eval_n)
    f_ms = latency_ms(f_sess, tx, runs=args.latency_runs)
    q_ms = latency_ms(q_sess, tx, runs=args.latency_runs)
    report = {
        "float_model": str(src), "quantized_model": str(dest),
        "quantizer": args.quantizer, "config": args.config,
        "cpu_float_accuracy": f_acc, "cpu_int8_accuracy": q_acc,
        "cpu_float_latency_ms": f_ms, "cpu_int8_latency_ms": q_ms,
        "npu": None,
    }
    print(f"CPU float={f_acc:.2%}, INT8={q_acc:.2%}, delta={q_acc-f_acc:+.2%}")
    print(f"CPU latency float={f_ms:.3f} ms, INT8={q_ms:.3f} ms")

    if args.npu != "off":
        n_sess, error = npu_session(str(dest))
        if error:
            if args.npu == "require":
                raise RuntimeError(error)
            print(f"NPU skipped: {error}")
            report["npu"] = {"status": "skipped", "reason": error}
        else:
            acc = accuracy(n_sess, tx, ty, args.eval_n)
            ms = latency_ms(n_sess, tx, runs=args.latency_runs)
            report["npu"] = {"status": "evaluated", "accuracy": acc, "latency_ms": ms}
            print(f"NPU INT8 accuracy={acc:.2%}, latency={ms:.3f} ms")
            print("Inspect the Vitis AI EP logs to confirm the NPU/CPU operator partition.")
    if args.report is not None:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report: {args.report}")
    return report


if __name__ == "__main__":
    main()
