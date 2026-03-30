"""
P6: training-data fraction sensitivity pilot for c3c4 / 2-class.

Strategy:
1. Reuse the existing 100% rows from degradation_2class.csv.
2. Run only 25% / 50% / 75% fractions on the 5-subject pilot set.
3. Write a raw CSV and a fraction-level summary CSV.

Usage:
    python run_data_fraction_pilot.py
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
FRACTIONS = [0.25, 0.50, 0.75, 1.00]
CONFIG = "c3c4"
CLASSES = "1,2"
EPOCHS = 250
WINDOW_SIZE = 8
SEED = 42


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_val = safe_mean(values)
    return math.sqrt(sum((v - mean_val) ** 2 for v in values) / (len(values) - 1))


def load_full_baseline(path: Path) -> dict[int, list[str]]:
    rows: dict[int, list[str]] = {}
    if not path.exists():
        return rows

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["channel_config"] != CONFIG:
                continue
            if row["classes"] != CLASSES:
                continue
            subject = int(row["subject"])
            if subject not in SUBJECTS:
                continue
            rows[subject] = [
                str(subject),
                f"{1.0:.2f}",
                CONFIG,
                "2",
                CLASSES,
                row["best_acc"],
                row["aver_acc"],
                "reused_baseline",
                "",
            ]
    return rows


def run_subject(base_dir: str, subject: int, train_fraction: float) -> list[str]:
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
        str(SEED),
        "--train_fraction",
        f"{train_fraction:.2f}",
    ]

    print(
        f">>> Running subject={subject}, fraction={train_fraction:.2f}, "
        f"config={CONFIG}, classes={CLASSES}"
    )
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
            f"{train_fraction:.2f}",
            CONFIG,
            "2",
            CLASSES,
            "",
            "",
            "failed",
            "",
        ]

    subject_id, config_name, n_channels, n_classes, window_size, result_seed, epochs, best_acc, aver_acc, duration = result_row
    return [
        subject_id,
        f"{train_fraction:.2f}",
        config_name,
        n_channels,
        CLASSES,
        best_acc,
        aver_acc,
        "fresh_run",
        duration,
    ]


def write_summary(raw_rows: list[list[str]], summary_path: Path) -> None:
    grouped: dict[str, dict[str, list[float]]] = {}

    for row in raw_rows:
        fraction = row[1]
        best_acc = float(row[5])
        aver_acc = float(row[6])
        grouped.setdefault(fraction, {"best": [], "aver": []})
        grouped[fraction]["best"].append(best_acc)
        grouped[fraction]["aver"].append(aver_acc)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "train_fraction",
                "n_subjects",
                "best_acc_mean",
                "best_acc_std",
                "aver_acc_mean",
                "aver_acc_std",
            ]
        )

        for fraction in sorted(grouped, key=float):
            best_vals = grouped[fraction]["best"]
            aver_vals = grouped[fraction]["aver"]
            writer.writerow(
                [
                    fraction,
                    len(best_vals),
                    f"{safe_mean(best_vals):.6f}",
                    f"{safe_std(best_vals):.6f}",
                    f"{safe_mean(aver_vals):.6f}",
                    f"{safe_std(aver_vals):.6f}",
                ]
            )


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_dir = Path(base_dir).parent / "results_summaries"
    summary_dir.mkdir(exist_ok=True)

    full_rows = load_full_baseline(summary_dir / "degradation_2class.csv")
    raw_rows: list[list[str]] = []

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = summary_dir / f"data_fraction_pilot_raw_{ts}.csv"
    summary_path = summary_dir / f"data_fraction_pilot_summary_{ts}.csv"
    latest_raw = summary_dir / "data_fraction_pilot_raw_latest.csv"
    latest_summary = summary_dir / "data_fraction_pilot_summary_latest.csv"
    total_start = datetime.datetime.now()

    print(f"{'=' * 72}")
    print("  P6 data-fraction sensitivity pilot")
    print(f"  Subjects:   {SUBJECTS}")
    print(f"  Fractions:  {FRACTIONS}")
    print(f"  Config:     {CONFIG}")
    print(f"  Classes:    {CLASSES}")
    print(f"  Seed:       {SEED}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Start:      {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 72}\n")

    for fraction in FRACTIONS:
        for subject in SUBJECTS:
            if fraction == 1.0 and subject in full_rows:
                raw_rows.append(full_rows[subject])
                print(f">>> Reusing subject={subject}, fraction=1.00 from degradation_2class.csv")
                continue
            raw_rows.append(run_subject(base_dir, subject, fraction))

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "train_fraction",
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
    print(f"  P6 pilot finished. Total duration: {total_end - total_start}")
    print(f"  Raw saved to:     {raw_path}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Latest raw:       {latest_raw}")
    print(f"  Latest summary:   {latest_summary}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
