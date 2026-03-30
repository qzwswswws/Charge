"""
P5-A3: multi-seed reproducibility for c3czc4 / 2-class on the 5-subject pilot set.

Strategy:
1. Reuse the existing seed=42 pilot rows from channel_combo_pilot_2class_latest.csv.
2. Run only the missing seeds.
3. Write a raw CSV and a subject/overall summary CSV.

Usage:
    python run_multiseed_c3czc4_2class_pilot.py
"""

from __future__ import annotations

import csv
import datetime
import math
import os
import shutil
import subprocess
from pathlib import Path

PYTHON = "/home/woqiu/anaconda3/envs/eegconformer/bin/python"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformer_degradation.py")

SUBJECTS = [1, 3, 5, 8, 9]
SEEDS = [42, 3407, 20260329]
CONFIG = "c3czc4"
N_CHANNELS = "3"
CLASSES = "1,2"
EPOCHS = 250
WINDOW_SIZE = 8


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_val = safe_mean(values)
    return math.sqrt(sum((v - mean_val) ** 2 for v in values) / (len(values) - 1))


def load_seed42_baseline(path: Path) -> dict[int, list[str]]:
    rows: dict[int, list[str]] = {}
    if not path.exists():
        return rows

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["channel_config"] != CONFIG:
                continue
            if row["classes"] != CLASSES:
                continue
            if row["n_channels"] != N_CHANNELS:
                continue
            subject = int(row["subject"])
            rows[subject] = [
                str(subject),
                str(42),
                CONFIG,
                N_CHANNELS,
                CLASSES,
                row["best_acc"],
                row["aver_acc"],
                "reused_p4_pilot",
                row.get("duration", ""),
            ]
    return rows


def run_subject(base_dir: str, subject: int, seed: int) -> list[str]:
    cmd = [
        PYTHON,
        "-u",
        SCRIPT,
        "--subject",
        str(subject),
        "--channel_config",
        CONFIG,
        "--classes",
        CLASSES,
        "--epochs",
        str(EPOCHS),
        "--window_size",
        str(WINDOW_SIZE),
        "--seed",
        str(seed),
    ]

    print(f">>> Running subject={subject}, seed={seed}, config={CONFIG}, classes={CLASSES}")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=base_dir,
    )

    if proc.stdout:
        print(proc.stdout[-1200:] if len(proc.stdout) > 1200 else proc.stdout)
    if proc.stderr:
        err_lines = proc.stderr.strip().split("\n")
        for line in err_lines[-5:]:
            print(f"  STDERR: {line}")

    result_row = None
    for line in proc.stdout.split("\n"):
        if line.startswith("RESULT_CSV:"):
            result_row = line.replace("RESULT_CSV: ", "").split(",")
            break

    if result_row is None:
        return [
            str(subject),
            str(seed),
            CONFIG,
            N_CHANNELS,
            CLASSES,
            "",
            "",
            "failed",
            "",
        ]

    subject_id, config_name, n_channels, n_classes, window_size, result_seed, epochs, best_acc, aver_acc, duration = result_row
    return [
        subject_id,
        result_seed,
        config_name,
        n_channels,
        CLASSES,
        best_acc,
        aver_acc,
        "fresh_run",
        duration,
    ]


def write_summary(raw_rows: list[list[str]], summary_path: Path) -> None:
    grouped: dict[int, dict[str, list[float]]] = {}
    by_seed: dict[int, dict[str, list[float]]] = {}
    for row in raw_rows:
        subject = int(row[0])
        seed = int(row[1])
        best_acc = float(row[5])
        aver_acc = float(row[6])
        grouped.setdefault(subject, {"best": [], "aver": []})
        grouped[subject]["best"].append(best_acc)
        grouped[subject]["aver"].append(aver_acc)
        by_seed.setdefault(seed, {"best": [], "aver": []})
        by_seed[seed]["best"].append(best_acc)
        by_seed[seed]["aver"].append(aver_acc)

    all_best_subject_means: list[float] = []
    all_aver_subject_means: list[float] = []

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "n_seeds",
                "best_acc_mean",
                "best_acc_std",
                "aver_acc_mean",
                "aver_acc_std",
            ]
        )

        for subject in sorted(grouped):
            best_vals = grouped[subject]["best"]
            aver_vals = grouped[subject]["aver"]
            best_mean = safe_mean(best_vals)
            aver_mean = safe_mean(aver_vals)
            all_best_subject_means.append(best_mean)
            all_aver_subject_means.append(aver_mean)
            writer.writerow(
                [
                    subject,
                    len(best_vals),
                    f"{best_mean:.6f}",
                    f"{safe_std(best_vals):.6f}",
                    f"{aver_mean:.6f}",
                    f"{safe_std(aver_vals):.6f}",
                ]
            )

        seed_best_means = [safe_mean(values["best"]) for values in by_seed.values()]
        seed_aver_means = [safe_mean(values["aver"]) for values in by_seed.values()]

        writer.writerow([])
        writer.writerow(
            [
                "overall_subject_means",
                len(SEEDS),
                f"{safe_mean(all_best_subject_means):.6f}",
                f"{safe_std(all_best_subject_means):.6f}",
                f"{safe_mean(all_aver_subject_means):.6f}",
                f"{safe_std(all_aver_subject_means):.6f}",
            ]
        )
        writer.writerow(
            [
                "overall_seed_means",
                len(SEEDS),
                f"{safe_mean(seed_best_means):.6f}",
                f"{safe_std(seed_best_means):.6f}",
                f"{safe_mean(seed_aver_means):.6f}",
                f"{safe_std(seed_aver_means):.6f}",
            ]
        )


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_dir = Path(base_dir).parent / "results_summaries"
    summary_dir.mkdir(exist_ok=True)

    seed42_rows = load_seed42_baseline(summary_dir / "channel_combo_pilot_2class_latest.csv")
    raw_rows: list[list[str]] = []

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = summary_dir / f"multi_seed_c3czc4_2class_pilot_raw_{ts}.csv"
    summary_path = summary_dir / f"multi_seed_c3czc4_2class_pilot_summary_{ts}.csv"
    latest_raw = summary_dir / "multi_seed_c3czc4_2class_pilot_raw_latest.csv"
    latest_summary = summary_dir / "multi_seed_c3czc4_2class_pilot_summary_latest.csv"
    total_start = datetime.datetime.now()

    print(f"{'=' * 72}")
    print("  P5-A3 multi-seed reproducibility")
    print(f"  Subjects: {SUBJECTS}")
    print(f"  Seeds:    {SEEDS}")
    print(f"  Config:   {CONFIG}")
    print(f"  Classes:  {CLASSES}")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  Start:    {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 72}\n")

    for seed in SEEDS:
        for subject in SUBJECTS:
            if seed == 42 and subject in seed42_rows:
                raw_rows.append(seed42_rows[subject])
                print(f">>> Reusing subject={subject}, seed=42 from channel_combo_pilot_2class_latest.csv")
                continue
            raw_rows.append(run_subject(base_dir, subject, seed))

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "seed",
                "channel_config",
                "n_channels",
                "classes",
                "best_acc",
                "aver_acc",
                "source",
                "duration",
            ]
        )
        writer.writerows(raw_rows)

    write_summary(raw_rows, summary_path)
    shutil.copyfile(raw_path, latest_raw)
    shutil.copyfile(summary_path, latest_summary)

    total_end = datetime.datetime.now()
    print(f"\n{'=' * 72}")
    print(f"  P5-A3 finished. Total duration: {total_end - total_start}")
    print(f"  Raw saved to:     {raw_path}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Latest raw:       {latest_raw}")
    print(f"  Latest summary:   {latest_summary}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
