"""
Expand the existing BCI IV 2b 5-subject pilot to all 9 subjects.

Strategy:
1. Reuse the latest 5-subject pilot rows if available.
2. Run only the missing subjects from 1..9.
3. Save a full 9-subject summary CSV and a stable latest copy.

Usage:
    python run_2b_full.py
"""

from __future__ import annotations

import csv
import datetime
import os
import shutil
import subprocess
from pathlib import Path


PYTHON = "/home/woqiu/anaconda3/envs/eegconformer/bin/python"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformer_2b_baseline.py")

ALL_SUBJECTS = list(range(1, 10))
EPOCHS = 250
WINDOW_SIZE = 8
SEED = 42


def latest_2b_pilot(summary_dir: Path) -> Path | None:
    matches = sorted(summary_dir.glob("bciiv2b_pilot_2class_*.csv"))
    return matches[-1] if matches else None


def load_existing_rows(path: Path | None) -> dict[int, list[str]]:
    if path is None or not path.exists():
        return {}

    rows: dict[int, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subject = int(row["subject"])
            rows[subject] = [
                row["subject"],
                row["model_name"],
                row["n_channels"],
                row["n_classes"],
                row["window_size"],
                row["seed"],
                row["epochs"],
                row["best_acc"],
                row["aver_acc"],
                "reused_pilot",
                row["duration"],
            ]
    return rows


def run_subject(base_dir: str, subject: int) -> list[str]:
    cmd = [
        PYTHON,
        "-u",
        SCRIPT,
        "--subject",
        str(subject),
        "--epochs",
        str(EPOCHS),
        "--window_size",
        str(WINDOW_SIZE),
        "--seed",
        str(SEED),
    ]

    print(f">>> Running 2b subject={subject}, epochs={EPOCHS}, window={WINDOW_SIZE}, seed={SEED}")
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
            "b2_baseline",
            "3",
            "2",
            str(WINDOW_SIZE),
            str(SEED),
            str(EPOCHS),
            "",
            "",
            "failed",
            "",
        ]

    subject_id, model_name, n_channels, n_classes, window_size, seed, epochs, best_acc, aver_acc, duration = result_row
    return [
        subject_id,
        model_name,
        n_channels,
        n_classes,
        window_size,
        seed,
        epochs,
        best_acc,
        aver_acc,
        "fresh_run",
        duration,
    ]


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_dir = Path(base_dir).parent / "results_summaries"
    summary_dir.mkdir(exist_ok=True)

    pilot_path = latest_2b_pilot(summary_dir)
    existing_rows = load_existing_rows(pilot_path)
    missing_subjects = [subject for subject in ALL_SUBJECTS if subject not in existing_rows]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = summary_dir / f"bciiv2b_full_9sub_{ts}.csv"
    latest_path = summary_dir / "bciiv2b_full_9sub_latest.csv"
    total_start = datetime.datetime.now()

    print(f"{'=' * 72}")
    print("  P8 BCI IV 2b full expansion")
    print(f"  Full subject set: {ALL_SUBJECTS}")
    print(f"  Existing pilot:   {pilot_path.name if pilot_path else 'None'}")
    print(f"  Reused subjects:  {sorted(existing_rows.keys()) if existing_rows else []}")
    print(f"  Missing subjects: {missing_subjects}")
    print(f"  Epochs:           {EPOCHS}")
    print(f"  Window:           {WINDOW_SIZE}")
    print(f"  Seed:             {SEED}")
    print(f"  Start:            {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 72}\n")

    for subject in missing_subjects:
        existing_rows[subject] = run_subject(base_dir, subject)

    ordered_rows = [existing_rows[subject] for subject in ALL_SUBJECTS]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "model_name",
                "n_channels",
                "n_classes",
                "window_size",
                "seed",
                "epochs",
                "best_acc",
                "aver_acc",
                "source",
                "duration",
            ]
        )
        writer.writerows(ordered_rows)

    shutil.copyfile(summary_path, latest_path)

    total_end = datetime.datetime.now()
    print(f"\n{'=' * 72}")
    print(f"  P8 full expansion finished. Total duration: {total_end - total_start}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Latest copy:      {latest_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
