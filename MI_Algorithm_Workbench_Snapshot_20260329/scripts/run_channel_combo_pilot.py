"""
Channel-combination pilot for P4.

Goal:
1. Compare a small set of 2-channel / 3-channel central combinations.
2. Prioritize 2-class left-vs-right evaluation.
3. Reuse the existing c3c4 2-class baseline rows when possible.

Usage:
    python run_channel_combo_pilot.py
"""

from __future__ import annotations

import csv
import datetime
import os
import shutil
import subprocess
from pathlib import Path

PYTHON = "/home/woqiu/anaconda3/envs/eegconformer/bin/python"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformer_degradation.py")

SUBJECTS = [1, 3, 5, 8, 9]
CONFIGS = ["c3c4", "c3cz", "czc4", "c3czc4", "c1czc2"]
CLASSES = "1,2"
EPOCHS = 250
WINDOW_SIZE = 8
SEED = 42


def load_existing_c3c4_baseline(summary_path: Path) -> dict[int, list[str]]:
    if not summary_path.exists():
        return {}

    rows: dict[int, list[str]] = {}
    with summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["channel_config"] != "c3c4":
                continue
            if row.get("classes", "") != "1,2":
                continue
            subject = int(row["subject"])
            if subject not in SUBJECTS:
                continue
            rows[subject] = [
                row["subject"],
                row["channel_config"],
                row["n_channels"],
                row["classes"],
                row["best_acc"],
                row["aver_acc"],
                "reused_baseline",
                "",
            ]
    return rows


def run_subject(base_dir: str, subject: int, config: str) -> list[str]:
    cmd = [
        PYTHON,
        "-u",
        SCRIPT,
        "--subject",
        str(subject),
        "--channel_config",
        config,
        "--classes",
        CLASSES,
        "--epochs",
        str(EPOCHS),
        "--window_size",
        str(WINDOW_SIZE),
        "--seed",
        str(SEED),
    ]

    print(f">>> Running subject={subject}, config={config}, classes={CLASSES}")
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
            config,
            "",
            CLASSES,
            "",
            "",
            "failed",
            "",
        ]

    subject_id, config_name, n_channels, n_classes, window_size, seed, epochs, best_acc, aver_acc, duration = result_row
    return [
        subject_id,
        config_name,
        n_channels,
        CLASSES,
        best_acc,
        aver_acc,
        "fresh_run",
        duration,
    ]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_dir = Path(base_dir).parent / "results_summaries"
    summary_dir.mkdir(exist_ok=True)

    existing_c3c4 = load_existing_c3c4_baseline(summary_dir / "degradation_2class.csv")
    rows: list[list[str]] = []

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = summary_dir / f"channel_combo_pilot_2class_{ts}.csv"
    latest_path = summary_dir / "channel_combo_pilot_2class_latest.csv"
    total_start = datetime.datetime.now()

    print(f"{'=' * 72}")
    print("  P4 channel-combination pilot")
    print(f"  Subjects: {SUBJECTS}")
    print(f"  Configs:  {CONFIGS}")
    print(f"  Classes:  {CLASSES}")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  Start:    {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 72}\n")

    for config in CONFIGS:
        for subject in SUBJECTS:
            if config == "c3c4" and subject in existing_c3c4:
                rows.append(existing_c3c4[subject])
                print(f">>> Reusing subject={subject}, config=c3c4 from degradation_2class.csv")
                continue
            rows.append(run_subject(base_dir, subject, config))

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject",
            "channel_config",
            "n_channels",
            "classes",
            "best_acc",
            "aver_acc",
            "source",
            "duration",
        ])
        writer.writerows(rows)

    shutil.copyfile(summary_path, latest_path)

    total_end = datetime.datetime.now()
    print(f"\n{'=' * 72}")
    print(f"  P4 pilot finished. Total duration: {total_end - total_start}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Latest copy:      {latest_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
