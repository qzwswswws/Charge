"""
本地训练监测脚本。

用途:
1. 周期性检查训练日志目录、结果目录和汇总 CSV。
2. 给出 RUNNING / STALLED / FINISHED / PENDING 状态。
3. 持续写出 Markdown 报告，方便隔夜查看。

默认参数面向 lowchannel Batch 2:
    python monitor_training_run.py

单次快照:
    python monitor_training_run.py --once
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


SUBJECT_PATTERN = re.compile(r"subject_(\d+)_")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    workbench_dir = base_dir.parent
    default_report = workbench_dir / "00_AI_Management" / "Session_Logs" / "Overnight_Monitor_B2.md"

    parser = argparse.ArgumentParser(description="Monitor long-running training jobs via local logs.")
    parser.add_argument("--logs-root", default=str(base_dir / "logsLowChB2"))
    parser.add_argument("--results-dir", default=str(base_dir / "results"))
    parser.add_argument("--summary-glob", default=str(workbench_dir / "results_summaries" / "lowchannel_b2_pilot_*.csv"))
    parser.add_argument("--subjects", default="1,3,5,8,9")
    parser.add_argument("--expected-epochs", type=int, default=250)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--stall-minutes", type=int, default=15)
    parser.add_argument("--report-path", default=str(default_report))
    parser.add_argument("--process-pattern", default="run_lowchannel_b2_pilot.py|conformer_lowchannel_b2_diff.py")
    parser.add_argument("--once", action="store_true", help="Capture one snapshot and exit.")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means unlimited in watch mode.")
    return parser.parse_args()


def parse_subjects(subject_str: str) -> List[int]:
    return [int(item.strip()) for item in subject_str.split(",") if item.strip()]


def run_pgrep(pattern: str) -> List[str]:
    if not pattern:
        return []
    try:
        proc = subprocess.run(
            ["pgrep", "-af", pattern],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    lines = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "monitor_training_run.py" in line or "pgrep -af" in line:
            continue
        lines.append(line)
    return lines


def read_test_log(test_log_path: Path) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {
        "last_epoch": None,
        "last_loss": None,
        "last_acc": None,
        "best_acc": None,
        "best_epoch": None,
        "rows": 0,
    }

    if not test_log_path.exists():
        return result

    best_acc = None
    best_epoch = None
    last_row = None

    with test_log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
            result["rows"] = int(result["rows"] or 0) + 1
            try:
                epoch = int(row["epoch"])
                acc = float(row["accuracy"])
                loss = float(row["loss"])
            except (KeyError, TypeError, ValueError):
                continue

            if best_acc is None or acc > best_acc:
                best_acc = acc
                best_epoch = epoch

    if last_row is None:
        return result

    try:
        result["last_epoch"] = int(last_row["epoch"])
        result["last_loss"] = float(last_row["loss"])
        result["last_acc"] = float(last_row["accuracy"])
    except (KeyError, TypeError, ValueError):
        pass

    result["best_acc"] = best_acc
    result["best_epoch"] = best_epoch
    return result


def age_minutes(path: Path, now_ts: float) -> Optional[float]:
    if not path.exists():
        return None
    return max(0.0, (now_ts - path.stat().st_mtime) / 60.0)


def scan_subject_runs(logs_root: Path, subjects: List[int], expected_epochs: int, stall_minutes: int) -> List[Dict[str, object]]:
    now_ts = time.time()
    subject_runs: List[Dict[str, object]] = []

    run_dirs = sorted(
        [p for p in logs_root.glob("*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    latest_by_subject: Dict[int, Path] = {}
    for run_dir in run_dirs:
        match = SUBJECT_PATTERN.search(run_dir.name)
        if not match:
            continue
        subject = int(match.group(1))
        if subject not in subjects or subject in latest_by_subject:
            continue
        latest_by_subject[subject] = run_dir

    for subject in subjects:
        run_dir = latest_by_subject.get(subject)
        if run_dir is None:
            subject_runs.append({
                "subject": subject,
                "status": "PENDING",
                "run_dir": "",
                "last_epoch": None,
                "last_acc": None,
                "best_acc": None,
                "best_epoch": None,
                "age_min": None,
            })
            continue

        test_log_path = run_dir / "test_log.csv"
        log_stats = read_test_log(test_log_path)
        current_age = age_minutes(test_log_path, now_ts)
        last_epoch = log_stats["last_epoch"]

        if last_epoch is None:
            status = "INIT"
        elif last_epoch >= expected_epochs - 1:
            status = "COMPLETE"
        elif current_age is not None and current_age > stall_minutes:
            status = "STALLED"
        else:
            status = "RUNNING"

        subject_runs.append({
            "subject": subject,
            "status": status,
            "run_dir": run_dir.name,
            "last_epoch": last_epoch,
            "last_acc": log_stats["last_acc"],
            "best_acc": log_stats["best_acc"],
            "best_epoch": log_stats["best_epoch"],
            "age_min": current_age,
        })

    return subject_runs


def glob_summary_files(summary_glob: str) -> List[Path]:
    return sorted((Path(p) for p in glob.glob(summary_glob)), key=lambda p: p.stat().st_mtime)


def determine_overall_status(subject_runs: List[Dict[str, object]], summary_files: List[Path], process_lines: List[str]) -> str:
    statuses = {row["status"] for row in subject_runs}
    if summary_files and all(row["status"] == "COMPLETE" for row in subject_runs):
        return "FINISHED"
    if "RUNNING" in statuses:
        return "RUNNING"
    if process_lines and ("INIT" in statuses or "PENDING" in statuses):
        return "RUNNING"
    if "STALLED" in statuses:
        return "STALLED"
    if all(status == "COMPLETE" for status in statuses):
        return "FINISHED"
    return "PENDING"


def latest_active_row(subject_runs: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    candidates = [row for row in subject_runs if row["status"] in {"RUNNING", "STALLED", "COMPLETE"} and row["last_epoch"] is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (int(row["subject"]), int(row["last_epoch"] or -1)))


def format_value(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(
    overall_status: str,
    subject_runs: List[Dict[str, object]],
    summary_files: List[Path],
    process_lines: List[str],
    report_path: Path,
) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    complete_count = sum(1 for row in subject_runs if row["status"] == "COMPLETE")
    total_count = len(subject_runs)
    active_row = latest_active_row(subject_runs)

    lines = [
        "# Overnight Monitor Report",
        "",
        f"- Updated: `{now}`",
        f"- Overall Status: `{overall_status}`",
        f"- Completed Subjects: `{complete_count}/{total_count}`",
    ]

    if active_row is not None:
        lines.extend([
            f"- Latest Subject: `{active_row['subject']}`",
            f"- Latest Epoch: `{format_value(active_row['last_epoch'], digits=0)}`",
            f"- Latest Accuracy: `{format_value(active_row['last_acc'])}`",
            f"- Best Accuracy So Far: `{format_value(active_row['best_acc'])}`",
        ])

    if summary_files:
        lines.append(f"- Latest Summary: `{summary_files[-1]}`")
    else:
        lines.append("- Latest Summary: `not generated yet`")

    if process_lines:
        lines.append(f"- Process Matches: `{len(process_lines)}`")
    else:
        lines.append("- Process Matches: `0`")

    lines.extend([
        "",
        "## Subject Status",
        "",
        "| Subject | Status | Last Epoch | Last Acc | Best Acc | File Age (min) | Run Dir |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])

    for row in subject_runs:
        lines.append(
            "| "
            + f"{row['subject']} | {row['status']} | "
            + f"{format_value(row['last_epoch'], digits=0)} | "
            + f"{format_value(row['last_acc'])} | "
            + f"{format_value(row['best_acc'])} | "
            + f"{format_value(row['age_min'], digits=2)} | "
            + f"{row['run_dir'] or '-'} |"
        )

    if process_lines:
        lines.extend([
            "",
            "## Process Snapshot",
            "",
        ])
        for line in process_lines:
            lines.append(f"- `{line}`")

    lines.extend([
        "",
        f"_Auto-written by `{report_path.name}` monitor._",
        "",
    ])
    return "\n".join(lines)


def write_report(report_path: Path, content: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root).resolve()
    report_path = Path(args.report_path).resolve()
    subjects = parse_subjects(args.subjects)

    cycle = 0
    while True:
        cycle += 1
        process_lines = run_pgrep(args.process_pattern)
        subject_runs = scan_subject_runs(
            logs_root=logs_root,
            subjects=subjects,
            expected_epochs=args.expected_epochs,
            stall_minutes=args.stall_minutes,
        )
        summary_files = glob_summary_files(args.summary_glob)
        overall_status = determine_overall_status(subject_runs, summary_files, process_lines)
        report = build_report(overall_status, subject_runs, summary_files, process_lines, report_path)
        write_report(report_path, report)

        now = dt.datetime.now().strftime("%H:%M:%S")
        complete_count = sum(1 for row in subject_runs if row["status"] == "COMPLETE")
        active_row = latest_active_row(subject_runs)
        active_desc = "no active subject"
        if active_row is not None:
            active_desc = (
                f"S{active_row['subject']} epoch={format_value(active_row['last_epoch'], digits=0)} "
                f"best={format_value(active_row['best_acc'])}"
            )
        print(f"[{now}] status={overall_status} complete={complete_count}/{len(subjects)} active={active_desc}")

        if args.once:
            break
        if overall_status == "FINISHED":
            break
        if args.max_cycles > 0 and cycle >= args.max_cycles:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
