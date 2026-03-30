"""
P8 pilot for BCI Competition IV 2b.

Goal:
1. Verify that the external-dataset pipeline runs beyond a 1-epoch smoke test.
2. Produce an initial 5-subject pilot under the same window/seed conventions.
3. Save a compact CSV that can be referenced in follow-up notes and thesis drafts.

Usage:
    python run_2b_pilot.py
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

SUBJECTS = [1, 3, 5, 8, 9]
EPOCHS = 250
WINDOW_SIZE = 8
SEED = 42


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

    rows: list[list[str]] = []
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = summary_dir / f"bciiv2b_pilot_2class_{ts}.csv"
    latest_path = summary_dir / "bciiv2b_pilot_2class_latest.csv"
    total_start = datetime.datetime.now()

    print(f"{'=' * 72}")
    print("  P8 BCI IV 2b pilot")
    print(f"  Subjects: {SUBJECTS}")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  Window:   {WINDOW_SIZE}")
    print(f"  Seed:     {SEED}")
    print(f"  Start:    {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 72}\n")

    for subject in SUBJECTS:
        rows.append(run_subject(base_dir, subject))

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
        writer.writerows(rows)

    shutil.copyfile(summary_path, latest_path)

    total_end = datetime.datetime.now()
    print(f"\n{'=' * 72}")
    print(f"  P8 pilot finished. Total duration: {total_end - total_start}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Latest copy:      {latest_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
