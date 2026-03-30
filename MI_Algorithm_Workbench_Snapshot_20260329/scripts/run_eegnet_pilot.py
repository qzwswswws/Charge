"""
Minimal EEGNet pilot for P7.

Goal:
1. Add a lightweight classical deep-learning baseline for thesis comparison.
2. Keep the scope intentionally narrow: full-channel 4-class and C3/C4 2-class.
3. Reuse the same 5 representative subjects used by prior pilot experiments.

Usage:
    python run_eegnet_pilot.py
"""

from __future__ import annotations

import csv
import datetime
import os
import shutil
import subprocess
from pathlib import Path


PYTHON = "/home/woqiu/anaconda3/envs/eegconformer/bin/python"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eegnet_baseline.py")

SUBJECTS = [1, 3, 5, 8, 9]
TASKS = [
    {"channel_config": "full", "classes": "1,2,3,4", "tag": "full_4class"},
    {"channel_config": "c3c4", "classes": "1,2", "tag": "c3c4_2class"},
]
EPOCHS = 250
WINDOW_SIZE = 8
SEED = 42


def load_completed_rows(base_dir: str) -> dict[tuple[int, str], list[str]]:
    """Reuse finished EEGNet runs when a full 250-epoch log already exists."""
    logs_root = Path(base_dir) / "logsEEGNet"
    results_root = Path(base_dir) / "results"
    completed: dict[tuple[int, str], list[str]] = {}

    if not logs_root.exists():
        return completed

    for task in TASKS:
        n_channels = 22 if task["channel_config"] == "full" else len(task["channel_config"].split("cz"))  # placeholder
        if task["channel_config"] == "full":
            n_channels = 22
        elif task["channel_config"] == "c3c4":
            n_channels = 2
        else:
            continue
        n_classes = 4 if task["classes"] == "1,2,3,4" else 2

        for subject in SUBJECTS:
            pattern = f"eegnet_subject_{subject}_ch{n_channels}_cls{n_classes}_frac100_*"
            candidates = sorted(logs_root.glob(pattern), reverse=True)
            if not candidates:
                continue

            log_txt = results_root / f"log_eegnet_subject{subject}_ch{n_channels}_cls{n_classes}_frac100.txt"
            if not log_txt.exists():
                continue

            txt = log_txt.read_text(encoding="utf-8")
            if "The average accuracy is:" not in txt or "The best accuracy is:" not in txt:
                continue

            best_acc = None
            aver_acc = None
            for line in txt.splitlines():
                if line.startswith("The average accuracy is:"):
                    aver_acc = line.split(":", 1)[1].strip()
                if line.startswith("The best accuracy is:"):
                    best_acc = line.split(":", 1)[1].strip()

            if best_acc is None or aver_acc is None:
                continue

            complete_dir = None
            for candidate in candidates:
                test_log = candidate / "test_log.csv"
                if not test_log.exists():
                    continue
                with test_log.open("r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)
                if line_count >= EPOCHS + 1:
                    complete_dir = candidate
                    break

            if complete_dir is None:
                continue

            completed[(subject, task["tag"])] = [
                str(subject),
                task["tag"],
                task["channel_config"],
                task["classes"],
                best_acc,
                aver_acc,
                "reused_complete_log",
                "",
            ]

    return completed


def run_subject(base_dir: str, subject: int, task: dict[str, str]) -> list[str]:
    cmd = [
        PYTHON,
        "-u",
        SCRIPT,
        "--subject",
        str(subject),
        "--channel_config",
        task["channel_config"],
        "--classes",
        task["classes"],
        "--epochs",
        str(EPOCHS),
        "--window_size",
        str(WINDOW_SIZE),
        "--seed",
        str(SEED),
    ]

    print(
        f">>> Running EEGNet subject={subject}, "
        f"config={task['channel_config']}, classes={task['classes']}"
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
            task["tag"],
            task["channel_config"],
            task["classes"],
            "",
            "",
            "failed",
            "",
        ]

    subject_id, config_name, n_channels, n_classes, window_size, seed, epochs, best_acc, aver_acc, duration = result_row
    return [
        subject_id,
        task["tag"],
        config_name,
        task["classes"],
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
    completed_rows = load_completed_rows(base_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = summary_dir / f"eegnet_pilot_{ts}.csv"
    latest_path = summary_dir / "eegnet_pilot_latest.csv"
    total_start = datetime.datetime.now()

    print(f"{'=' * 72}")
    print("  P7 EEGNet minimal pilot")
    print(f"  Subjects: {SUBJECTS}")
    print(f"  Tasks:    {[task['tag'] for task in TASKS]}")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  Seed:     {SEED}")
    print(f"  Start:    {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 72}\n")

    for task in TASKS:
        for subject in SUBJECTS:
            key = (subject, task["tag"])
            if key in completed_rows:
                print(f">>> Reusing EEGNet subject={subject}, task={task['tag']} from completed logs")
                rows.append(completed_rows[key])
                continue
            rows.append(run_subject(base_dir, subject, task))

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "task_tag",
                "channel_config",
                "classes",
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
    print(f"  P7 EEGNet pilot finished. Total duration: {total_end - total_start}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Latest copy:      {latest_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
