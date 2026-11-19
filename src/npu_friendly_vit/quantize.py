"""
quantize.py - quantize the exported NPUViT and run it on the Hawk Point NPU.

Run inside the Ryzen AI conda env (conda activate ryzen-ai-1.8.0):

    python -m npu_friendly_vit.quantize --model out/npu_vit_pwl.onnx --data out --config XINT8
    python -m npu_friendly_vit.quantize --model out/npu_vit_pwl.onnx --data out --config A8W8

Or via uv (for ORT quantization testing on any machine):

    uv run npu-vit-quantize --model out/npu_vit_pwl.onnx --data out --quantizer ort

The model input is patches [1, 64, 48]; calib.npy / test_x.npy are already
patchified by train.py (use npu_friendly_vit.patchify_np for new images).

Steps: quantize with AMD Quark -> accuracy on CPU (float vs INT8) ->
compile + run on the NPU (target X1) -> accuracy, latency, and the
NPU/CPU operator split printed in the Vitis AI log.

--quantizer ort uses ONNX Runtime's quantize_static instead of Quark
(handy for testing the pipeline on a machine without Ryzen AI).
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader


class NpyReader(CalibrationDataReader):
    def __init__(self, calib):
        self.calib = calib
        self.rewind()

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter([{"input": x[None]} for x in self.calib])


def quantize_quark(src, dst, calib, config):
    try:  # newer Quark API
        from quark.onnx import ModelQuantizer, QConfig

        cfg = QConfig.get_default_config(config)
    except ImportError:  # older Quark API (Ryzen AI <= 1.5 era)
        from quark.onnx import ModelQuantizer
        from quark.onnx.quantization.config import Config, get_default_config

        cfg = Config(global_quant_config=get_default_config(config))
    ModelQuantizer(cfg).quantize_model(src, dst, NpyReader(calib))


def quantize_ort(src, dst, calib):
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    quantize_static(
        src,
        dst,
        NpyReader(calib),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )


def accuracy(sess, x, y, limit=None):
    n = len(x) if limit is None else min(limit, len(x))
    preds = [sess.run(None, {"input": x[i : i + 1]})[0].argmax() for i in range(n)]
    return float(np.mean(np.array(preds) == y[:n]))


def latency_ms(sess, x, runs=200):
    inp = {"input": x[:1]}
    for _ in range(10):
        sess.run(None, inp)
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, inp)
    return (time.perf_counter() - t0) / runs * 1000


def npu_session(model_path):
    root = os.environ.get("RYZEN_AI_INSTALLATION_PATH")
    if not root:
        return None, "RYZEN_AI_INSTALLATION_PATH not set (not in the Ryzen AI env)"
    hits = glob.glob(os.path.join(root, "voe-*", "xclbins", "phoenix", "4x4.xclbin"))
    if not hits:
        return None, "phoenix/4x4.xclbin not found"
    opts = {
        "target": "X1",
        "xclbin": hits[0],
        "cache_dir": os.path.abspath("vaip_cache"),
        "cache_key": os.path.splitext(os.path.basename(model_path))[0],
        "enable_cache_file_io_in_mem": "0",
    }
    so = ort.SessionOptions()
    so.log_severity_level = 1  # prints "[Vitis AI EP] No. of Operators : NPU .. CPU .."
    sess = ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=["VitisAIExecutionProvider"],
        provider_options=[opts],
    )
    return sess, None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="float ONNX from train.py")
    p.add_argument("--data", default="out", help="folder with calib.npy, test_x.npy, test_y.npy")
    p.add_argument("--config", default="XINT8", help="XINT8 | A8W8 | A16W8 (Quark)")
    p.add_argument("--quantizer", default="quark", choices=["quark", "ort"])
    p.add_argument("--eval-n", type=int, default=2000)
    a = p.parse_args(argv)

    calib = np.load(os.path.join(a.data, "calib.npy"))
    tx = np.load(os.path.join(a.data, "test_x.npy"))
    ty = np.load(os.path.join(a.data, "test_y.npy"))
    tag = a.config if a.quantizer == "quark" else "ortint8"
    stem = a.model[:-5] if a.model.endswith(".onnx") else a.model
    qpath = f"{stem}_{tag}.onnx"

    print(f"Quantizing {a.model} -> {qpath} ({a.quantizer}, {a.config}) ...")
    if a.quantizer == "quark":
        quantize_quark(a.model, qpath, calib, a.config)
    else:
        quantize_ort(a.model, qpath, calib)

    cpu = ["CPUExecutionProvider"]
    f_sess = ort.InferenceSession(a.model, providers=cpu)
    q_sess = ort.InferenceSession(qpath, providers=cpu)
    f_acc, q_acc = accuracy(f_sess, tx, ty, a.eval_n), accuracy(q_sess, tx, ty, a.eval_n)
    print(
        f"\nCPU  float acc {f_acc * 100:.2f}%   INT8 acc {q_acc * 100:.2f}%   "
        f"(drop {100 * (f_acc - q_acc):.2f} pts)"
    )
    print(
        f"CPU  float latency {latency_ms(f_sess, tx):.2f} ms   "
        f"INT8 latency {latency_ms(q_sess, tx):.2f} ms"
    )

    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        sys.exit("\nVitisAIExecutionProvider not available - stopping before the NPU step.")
    print("\nCompiling for the NPU (first run is slow, then cached) ...")
    n_sess, err = npu_session(qpath)
    if err:
        sys.exit(err)
    n_acc = accuracy(n_sess, tx, ty, a.eval_n)
    print(f"NPU  INT8 acc {n_acc * 100:.2f}%   latency {latency_ms(n_sess, tx):.2f} ms")
    print(
        "Check the log above for the NPU vs CPU operator counts. "
        "Fewer CPU ops and fewer NPU<->CPU switches = better."
    )


if __name__ == "__main__":
    main()
