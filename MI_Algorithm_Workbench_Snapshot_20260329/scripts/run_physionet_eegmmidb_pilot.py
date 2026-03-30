"""
Pilot runner for PhysioNet eegmmidb Conformer baseline.

Current pilot protocol:
- Subjects: 1, 3, 5, 8, 9
- Channels: C3/Cz/C4
- Task: imagined left vs right fist
- Train runs: R04, R08
- Test run: R12
"""

from __future__ import annotations

import csv
import datetime
import os
import shutil
import subprocess
from pathlib import Path


PYTHON = "/home/woqiu/anaconda3/envs/eegconformer/bin/python"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformer_physionet_eegmmidb_baseline.py")

SUBJECTS = [1, 3, 5, 8, 9]
CHANNEL_MODE = "c3czc4"
TRAIN_RUNS = "4,8"
TEST_RUNS = "12"
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
        "--channel_mode",
        CHANNEL_MODE,
        "--train_runs",
        TRAIN_RUNS,
        "--test_runs",
        TEST_RUNS,
        "--epochs",
        str(EPOCHS),
        "--window_size",
        str(WINDOW_SIZE),
        "--seed",
        str(SEED),
    ]

    print(
        f">>> Running PhysioNet subject={subject}, channel_mode={CHANNEL_MODE}, "
        f"train_runs={TRAIN_RUNS}, test_runs={TEST_RUNS}"
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=base_dir)

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
            "physionet_conformer",
            CHANNEL_MODE,
            "2",
            str(WINDOW_SIZE),
            str(SEED),
            str(EPOCHS),
            f"tr{TRAIN_RUNS.replace(',', '')}_te{TEST_RUNS.replace(',', '')}",
            "",
            "",
            "failed",
            "",
        ]

    (
        subject_id,
        model_name,
        channel_mode,
        n_classes,
        window_size,
        seed,
        epochs,
        run_tag,
        best_acc,
        aver_acc,
        duration,
    ) = result_row

    return [
        subject_id,
        model_name,
        channel_mode,
        n_classes,
        window_size,
        seed,
        epochs,
        run_tag,
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
    summary_path = summary_dir / f"physionet_eegmmidb_pilot_{ts}.csv"
    latest_path = summary_dir / "physionet_eegmmidb_pilot_latest.csv"
    total_start = datetime.datetime.now()

    print(f"{'=' * 72}")
    print("  PhysioNet eegmmidb pilot")
    print(f"  Subjects:      {SUBJECTS}")
    print(f"  Channel mode:  {CHANNEL_MODE}")
    print(f"  Train runs:    {TRAIN_RUNS}")
    print(f"  Test runs:     {TEST_RUNS}")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Window:        {WINDOW_SIZE}")
    print(f"  Seed:          {SEED}")
    print(f"  Start:         {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 72}\n")

    for subject in SUBJECTS:
        rows.append(run_subject(base_dir, subject))

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "model_name",
                "channel_mode",
                "n_classes",
                "window_size",
                "seed",
                "epochs",
                "run_tag",
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
    print(f"  PhysioNet pilot finished. Total duration: {total_end - total_start}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Latest copy:      {latest_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
