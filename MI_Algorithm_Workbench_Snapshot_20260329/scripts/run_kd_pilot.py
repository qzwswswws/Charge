"""
Minimal KD pilot runner.

Focus:
- 22-channel teacher
- c3c4 2-channel student
- 4-class MI
- pilot subjects: 1,3,5,8,9
"""

import csv
import datetime
import os
import subprocess

PYTHON = "/home/woqiu/anaconda3/envs/eegconformer/bin/python"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conformer_kd_student.py")

SUBJECTS = [1, 3, 5, 8, 9]
EPOCHS = 250
WINDOW_SIZE = 8
TEACHER_WINDOW_SIZE = 8
SEED = 42
TEMPERATURE = 2.0
ALPHA = 0.5


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_dir = os.path.join(base_dir, "..", "results_summaries")
    os.makedirs(summary_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(summary_dir, f"kd_pilot_{ts}.csv")
    results = []
    total_start = datetime.datetime.now()

    print(f"{'=' * 72}")
    print("  Minimal KD pilot")
    print(f"  Subjects: {SUBJECTS}")
    print("  Experiment: full teacher -> c3c4 student / 4-class")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Start: {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 72}\n")

    for subject in SUBJECTS:
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
            results.append([
                "kdstudent",
                subject,
                "c3c4",
                "1,2,3,4",
                "",
                "",
                "",
                teacher_ckpt,
                "FAILED",
            ])
        else:
            subject_id, channel_config, n_channels, n_classes, window_size, seed, epochs, best_acc, aver_acc, teacher_acc, duration = result_row
            results.append([
                "kdstudent",
                subject_id,
                channel_config,
                "1,2,3,4",
                best_acc,
                aver_acc,
                teacher_acc,
                teacher_ckpt,
                duration,
            ])

    with open(summary_path, "w", newline="") as f:
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
        writer.writerows(results)

    total_end = datetime.datetime.now()
    print(f"\n{'=' * 72}")
    print(f"  KD pilot finished. Total duration: {total_end - total_start}")
    print(f"  Summary saved to: {summary_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
