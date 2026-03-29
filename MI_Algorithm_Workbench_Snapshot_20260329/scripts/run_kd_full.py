"""
Expand KD from the existing 5-subject pilot to all 9 subjects.

Strategy:
1. Reuse the latest existing KD pilot summary if available.
2. Run only the missing subjects from 1..9.
3. Write a full 9-subject summary CSV.

Usage:
    python run_kd_full.py
"""

from __future__ import annotations

import csv
import datetime
import os
import subprocess
from pathlib import Path

PYTHON = "/home/woqiu/anaconda3/envs/eegconformer/bin/python"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformer_kd_student.py")

ALL_SUBJECTS = list(range(1, 10))
EPOCHS = 250
WINDOW_SIZE = 8
TEACHER_WINDOW_SIZE = 8
SEED = 42
TEMPERATURE = 2.0
ALPHA = 0.5


def latest_kd_pilot(summary_dir: Path) -> Path | None:
    matches = sorted(summary_dir.glob("kd_pilot_*.csv"))
    return matches[-1] if matches else None


def load_existing_rows(path: Path | None) -> dict[int, list[str]]:
    if path is None or not path.exists():
        return {}
    rows: dict[int, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["subject"])] = [
                row["model_tag"],
                row["subject"],
                row["channel_config"],
                row["classes"],
                row["best_acc"],
                row["aver_acc"],
                row["teacher_acc"],
                row["teacher_ckpt"],
                row["duration"],
            ]
    return rows


def run_subject(base_dir: str, subject: int) -> list[str]:
    cmd = [
        PYTHON, "-u", SCRIPT,
        "--subject", str(subject),
        "--channel_config", "c3c4",
        "--classes", "1,2,3,4",
        "--epochs", str(EPOCHS),
        "--window_size", str(WINDOW_SIZE),
        "--teacher_window_size", str(TEACHER_WINDOW_SIZE),
        "--temperature", str(TEMPERATURE),
        "--alpha", str(ALPHA),
        "--seed", str(SEED),
    ]

    print(f">>> Running subject={subject}, tag=kd_c3c4_cls4")
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

    teacher_ckpt = ""
    result_row = None
    for line in proc.stdout.split("\n"):
        if line.startswith("TEACHER_CKPT:"):
            teacher_ckpt = line.replace("TEACHER_CKPT: ", "").strip()
        if line.startswith("RESULT_CSV:"):
            result_row = line.replace("RESULT_CSV: ", "").split(",")
            break

    if result_row is None:
        return [
            "kdstudent",
            str(subject),
            "c3c4",
            "1,2,3,4",
            "",
            "",
            "",
            teacher_ckpt,
            "FAILED",
        ]

    subject_id, channel_config, n_channels, n_classes, window_size, seed, epochs, best_acc, aver_acc, teacher_acc, duration = result_row
    return [
        "kdstudent",
        subject_id,
        channel_config,
        "1,2,3,4",
        best_acc,
        aver_acc,
        teacher_acc,
        teacher_ckpt,
        duration,
    ]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_dir = Path(base_dir).parent / "results_summaries"
    summary_dir.mkdir(exist_ok=True)

    pilot_path = latest_kd_pilot(summary_dir)
    existing_rows = load_existing_rows(pilot_path)
    missing_subjects = [subject for subject in ALL_SUBJECTS if subject not in existing_rows]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = summary_dir / f"kd_full_9sub_{ts}.csv"
    total_start = datetime.datetime.now()

    print(f"{'=' * 72}")
    print("  KD full-run expansion")
    print(f"  Full subject set: {ALL_SUBJECTS}")
    print(f"  Existing pilot:   {pilot_path.name if pilot_path else 'None'}")
    print(f"  Reused subjects:  {sorted(existing_rows.keys()) if existing_rows else []}")
    print(f"  Missing subjects: {missing_subjects}")
    print("  Experiment: full teacher -> c3c4 student / 4-class")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Start: {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 72}\n")

    for subject in missing_subjects:
        existing_rows[subject] = run_subject(base_dir, subject)

    ordered_rows = [existing_rows[subject] for subject in ALL_SUBJECTS]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_tag",
            "subject",
            "channel_config",
            "classes",
            "best_acc",
            "aver_acc",
            "teacher_acc",
            "teacher_ckpt",
            "duration",
        ])
        writer.writerows(ordered_rows)

    total_end = datetime.datetime.now()
    print(f"\n{'=' * 72}")
    print(f"  KD full expansion finished. Total duration: {total_end - total_start}")
    print(f"  Summary saved to: {summary_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
