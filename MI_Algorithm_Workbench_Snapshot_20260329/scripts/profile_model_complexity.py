"""
Model complexity profiler for the current MI workbench.

Outputs:
1. CSV summary under ../results_summaries/
2. Console summary for quick inspection

Notes:
- THOP reports MACs; we also report FLOPs as 2 * MACs for a common
  multiply-add convention.
- THOP may undercount some custom attention internals (einsum / softmax),
  so the MAC/FLOP values should be treated as comparative approximations.
- CPU latency is measured with batch size 1 on the local machine and is
  not a substitute for RK3568/NPU deployment latency.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import torch
from thop import profile

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from conformer_degradation import Conformer
from conformer_lowchannel_b1_poolhead import ConformerB1
from conformer_lowchannel_b2_diff import ConformerB2


@dataclass
class ModelSpec:
    tag: str
    family: str
    channel_config: str
    n_channels: int
    n_classes: int
    note: str


def build_model(spec: ModelSpec) -> torch.nn.Module:
    if spec.family == "baseline":
        return Conformer(
            emb_size=40,
            depth=6,
            n_classes=spec.n_classes,
            n_channels=spec.n_channels,
            window_size=8,
        )
    if spec.family == "lowch_b1":
        return ConformerB1(
            emb_size=40,
            depth=6,
            n_classes=spec.n_classes,
            n_channels=spec.n_channels,
            window_size=8,
        )
    if spec.family == "lowch_b2":
        return ConformerB2(
            emb_size=40,
            depth=6,
            n_classes=spec.n_classes,
            n_channels=spec.n_channels,
            window_size=8,
            use_diff_branch=True,
        )
    raise ValueError(f"Unsupported family: {spec.family}")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def serialized_model_size_mb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(model.state_dict(), tmp_path)
        return tmp_path.stat().st_size / (1024 * 1024)
    finally:
        tmp_path.unlink(missing_ok=True)


def benchmark_cpu_latency_ms(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
    warmup_runs: int = 10,
    timed_runs: int = 60,
    num_threads: int = 1,
) -> tuple[float, float]:
    old_threads = torch.get_num_threads()
    old_interop = torch.get_num_interop_threads()
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = model.cpu().eval()
    x = torch.randn(input_shape, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)

        measurements = []
        for _ in range(timed_runs):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            measurements.append((end - start) * 1000.0)

    torch.set_num_threads(old_threads)
    try:
        torch.set_num_interop_threads(old_interop)
    except RuntimeError:
        pass
    return mean(measurements), pstdev(measurements)


def profile_macs_flops(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
) -> tuple[float, float]:
    model = model.cpu().eval()
    dummy = torch.randn(input_shape, dtype=torch.float32)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    flops = macs * 2.0
    return macs, flops


def fmt_m(n: int) -> float:
    return n / 1e6


def fmt_g(n: float) -> float:
    return n / 1e9


def main() -> int:
    specs = [
        ModelSpec("baseline_full_4class", "baseline", "full", 22, 4, "Main full-channel baseline"),
        ModelSpec("baseline_central8_4class", "baseline", "central8", 8, 4, "8-channel degradation baseline"),
        ModelSpec("baseline_c3czc4_4class", "baseline", "c3czc4", 3, 4, "3-channel degradation baseline"),
        ModelSpec("baseline_c3c4_4class", "baseline", "c3c4", 2, 4, "2-channel degradation baseline"),
        ModelSpec("baseline_c3c4_2class", "baseline", "c3c4", 2, 2, "2-channel engineering baseline"),
        ModelSpec("lowch_b1_c3c4_2class", "lowch_b1", "c3c4", 2, 2, "Batch 1 attention-pooling variant"),
        ModelSpec("lowch_b2_c3c4_2class", "lowch_b2", "c3c4", 2, 2, "Batch 2 diff-branch variant"),
        ModelSpec("kd_student_c3c4_4class", "baseline", "c3c4", 2, 4, "Deploy-time student shares the same inference graph as baseline c3c4 / 4-class"),
    ]

    rows = []
    for spec in specs:
        model = build_model(spec)
        input_shape = (1, 1, spec.n_channels, 1000)
        total_params, trainable_params = count_parameters(model)
        macs, flops = profile_macs_flops(model, input_shape)
        model_size_mb = serialized_model_size_mb(model)
        latency_mean_ms, latency_std_ms = benchmark_cpu_latency_ms(model, input_shape)
        rows.append({
            "tag": spec.tag,
            "family": spec.family,
            "channel_config": spec.channel_config,
            "n_channels": spec.n_channels,
            "n_classes": spec.n_classes,
            "params_m": f"{fmt_m(total_params):.6f}",
            "trainable_params_m": f"{fmt_m(trainable_params):.6f}",
            "macs_g": f"{fmt_g(macs):.6f}",
            "flops_g": f"{fmt_g(flops):.6f}",
            "state_dict_size_mb": f"{model_size_mb:.6f}",
            "cpu_latency_mean_ms": f"{latency_mean_ms:.6f}",
            "cpu_latency_std_ms": f"{latency_std_ms:.6f}",
            "note": spec.note,
        })

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT_DIR / "results_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"model_complexity_profile_{ts}.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "family",
                "channel_config",
                "n_channels",
                "n_classes",
                "params_m",
                "trainable_params_m",
                "macs_g",
                "flops_g",
                "state_dict_size_mb",
                "cpu_latency_mean_ms",
                "cpu_latency_std_ms",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 88)
    print("Model complexity profile")
    print(f"Output: {out_path}")
    print("=" * 88)
    for row in rows:
        print(
            f"{row['tag']:<28} "
            f"params={float(row['params_m']):>7.3f}M  "
            f"macs={float(row['macs_g']):>7.3f}G  "
            f"flops={float(row['flops_g']):>7.3f}G  "
            f"size={float(row['state_dict_size_mb']):>6.3f}MB  "
            f"cpu={float(row['cpu_latency_mean_ms']):>8.3f}±{float(row['cpu_latency_std_ms']):<7.3f} ms"
        )

    print("RESULT_CSV:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
