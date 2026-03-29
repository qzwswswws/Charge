from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "baselines" / "results"
SUMM_DIR = ROOT_DIR / "results_summaries"
VIS_DIR = ROOT_DIR / "visualization"

LABELS_4 = ["Left", "Right", "Feet", "Tongue"]
LABELS_2 = ["Left", "Right"]
PILOT_SUBJECTS = {"1", "3", "5", "8", "9"}


@dataclass
class Selection:
    condition: str
    subject: str
    expected_acc: float
    selected_dir: str
    actual_acc: float
    acc_diff: float
    n_classes: int


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_matching(pattern: str) -> Path:
    matches = sorted(SUMM_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No summary matched {pattern}")
    return matches[-1]


def parse_ts_value(path: Path) -> float:
    return path.stat().st_mtime


def choose_result_dir(
    regex: str,
    expected_acc: float,
    n_classes: int,
) -> tuple[Path, np.ndarray, float, float]:
    compiled = re.compile(regex)
    candidates = []
    for path in RESULTS_DIR.iterdir():
        if not path.is_dir() or not compiled.match(path.name):
            continue
        matrix_path = path / "best_confusion_matrix.npy"
        if not matrix_path.exists():
            continue
        matrix = np.load(matrix_path)
        if matrix.shape != (n_classes, n_classes):
            continue
        actual_acc = float(np.trace(matrix) / matrix.sum())
        candidates.append(
            (
                abs(actual_acc - expected_acc),
                -parse_ts_value(path),
                path,
                matrix,
                actual_acc,
            )
        )

    if not candidates:
        raise FileNotFoundError(f"No confusion matrix matched {regex}")

    candidates.sort(key=lambda item: (item[0], item[1]))
    diff, _, path, matrix, actual_acc = candidates[0]
    return path, matrix, actual_acc, diff


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def summarize_matrix(condition: str, matrix: np.ndarray, labels: list[str]) -> tuple[list[dict[str, str]], dict[str, str]]:
    recalls = np.diag(matrix) / matrix.sum(axis=1)
    precisions = np.diag(matrix) / matrix.sum(axis=0)
    class_rows = []
    for idx, label in enumerate(labels):
        class_rows.append({
            "condition": condition,
            "class_index": str(idx),
            "class_label": label,
            "support": str(int(matrix[idx].sum())),
            "recall": f"{recalls[idx]:.6f}",
            "precision": f"{precisions[idx]:.6f}",
        })
    summary_row = {
        "condition": condition,
        "n_classes": str(len(labels)),
        "total_trials": str(int(matrix.sum())),
        "accuracy": f"{(np.trace(matrix) / matrix.sum()):.6f}",
    }
    return class_rows, summary_row


def write_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def plot_confusion(ax: plt.Axes, counts: np.ndarray, labels: list[str], title: str) -> None:
    norm = row_normalize(counts)
    im = ax.imshow(norm, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            pct = norm[i, j] * 100.0
            text_color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{pct:.1f}%\n({int(counts[i, j])})",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_recall_comparison(
    out_path: Path,
    full_recall: np.ndarray,
    c3c4_recall: np.ndarray,
    kd_base_recall: np.ndarray,
    kd_recall: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    x = np.arange(len(LABELS_4))
    width = 0.35

    axes[0].bar(x - width / 2, full_recall, width, label="22ch / 4-class")
    axes[0].bar(x + width / 2, c3c4_recall, width, label="2ch / 4-class")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LABELS_4)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Per-Class Recall: 22ch vs 2ch")
    axes[0].legend()

    axes[1].bar(x - width / 2, kd_base_recall, width, label="2ch baseline (pilot)")
    axes[1].bar(x + width / 2, kd_recall, width, label="2ch KD student (pilot)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LABELS_4)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Per-Class Recall: Baseline vs KD")
    axes[1].legend()

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    full_rows = read_csv(SUMM_DIR / "baseline_full_channel.csv")
    degr_rows = read_csv(SUMM_DIR / "degradation_4class.csv")
    two_class_rows = read_csv(SUMM_DIR / "degradation_2class.csv")
    kd_rows = read_csv(latest_matching("kd_pilot_*.csv"))

    selections: list[Selection] = []
    matrices: dict[str, np.ndarray] = {}

    def collect(
        condition: str,
        rows: Iterable[dict[str, str]],
        regex_builder,
        n_classes: int,
    ) -> np.ndarray:
        agg = np.zeros((n_classes, n_classes), dtype=int)
        for row in rows:
            subject = row["subject"]
            expected = float(row["best_acc"])
            regex = regex_builder(subject)
            selected_dir, matrix, actual_acc, diff = choose_result_dir(regex, expected, n_classes)
            agg += matrix
            selections.append(
                Selection(
                    condition=condition,
                    subject=subject,
                    expected_acc=expected,
                    selected_dir=selected_dir.name,
                    actual_acc=actual_acc,
                    acc_diff=diff,
                    n_classes=n_classes,
                )
            )
        matrices[condition] = agg
        return agg

    full_matrix = collect(
        "full_4class_9sub",
        full_rows,
        lambda subject: rf"^subject_{subject}_[0-9]{{8}}_[0-9]{{6}}$",
        4,
    )
    c3c4_4_matrix = collect(
        "c3c4_4class_9sub",
        [row for row in degr_rows if row["channel_config"] == "c3c4"],
        lambda subject: rf"^subject_{subject}_ch2_cls4_[0-9]{{8}}_[0-9]{{6}}$",
        4,
    )
    c3c4_2_matrix = collect(
        "c3c4_2class_9sub",
        two_class_rows,
        lambda subject: rf"^subject_{subject}_ch2_cls2_[0-9]{{8}}_[0-9]{{6}}$",
        2,
    )
    c3c4_4_pilot_base_matrix = collect(
        "c3c4_4class_pilot5_baseline",
        [row for row in degr_rows if row["channel_config"] == "c3c4" and row["subject"] in PILOT_SUBJECTS],
        lambda subject: rf"^subject_{subject}_ch2_cls4_[0-9]{{8}}_[0-9]{{6}}$",
        4,
    )
    c3c4_4_pilot_kd_matrix = collect(
        "c3c4_4class_pilot5_kd",
        kd_rows,
        lambda subject: rf"^kdstudent_subject_{subject}_ch2_cls4_[0-9]{{8}}_[0-9]{{6}}$",
        4,
    )

    selection_path = SUMM_DIR / "confusion_matrix_selection_manifest.csv"
    with selection_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "subject", "expected_acc", "selected_dir", "actual_acc", "acc_diff", "n_classes"])
        for item in selections:
            writer.writerow([
                item.condition,
                item.subject,
                f"{item.expected_acc:.6f}",
                item.selected_dir,
                f"{item.actual_acc:.6f}",
                f"{item.acc_diff:.6f}",
                item.n_classes,
            ])

    class_rows = []
    summary_rows = []
    for condition, matrix in matrices.items():
        labels = LABELS_4 if matrix.shape[0] == 4 else LABELS_2
        class_part, summary_part = summarize_matrix(condition, matrix, labels)
        class_rows.extend(class_part)
        summary_rows.append(summary_part)
        write_matrix_csv(SUMM_DIR / f"{condition}_counts.csv", labels, matrix)
        write_matrix_csv(SUMM_DIR / f"{condition}_row_norm.csv", labels, row_normalize(matrix))

    with (SUMM_DIR / "class_recall_precision_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "class_index", "class_label", "support", "recall", "precision"],
        )
        writer.writeheader()
        writer.writerows(class_rows)

    with (SUMM_DIR / "confusion_analysis_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "n_classes", "total_trials", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    plot_confusion(axes1[0], full_matrix, LABELS_4, "22ch / 4-class (9 subjects)")
    plot_confusion(axes1[1], c3c4_4_matrix, LABELS_4, "C3/C4 / 4-class (9 subjects)")
    plot_confusion(axes1[2], c3c4_2_matrix, LABELS_2, "C3/C4 / 2-class (9 subjects)")
    fig1.savefig(VIS_DIR / "confusion_matrix_key_conditions.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    plot_confusion(axes2[0], c3c4_4_pilot_base_matrix, LABELS_4, "C3/C4 / 4-class Baseline (pilot 5)")
    plot_confusion(axes2[1], c3c4_4_pilot_kd_matrix, LABELS_4, "C3/C4 / 4-class KD (pilot 5)")
    fig2.savefig(VIS_DIR / "confusion_matrix_kd_pilot.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    plot_recall_comparison(
        VIS_DIR / "class_recall_comparison.png",
        np.diag(full_matrix) / full_matrix.sum(axis=1),
        np.diag(c3c4_4_matrix) / c3c4_4_matrix.sum(axis=1),
        np.diag(c3c4_4_pilot_base_matrix) / c3c4_4_pilot_base_matrix.sum(axis=1),
        np.diag(c3c4_4_pilot_kd_matrix) / c3c4_4_pilot_kd_matrix.sum(axis=1),
    )

    print("Generated:")
    print(SUMM_DIR / "confusion_matrix_selection_manifest.csv")
    print(SUMM_DIR / "confusion_analysis_summary.csv")
    print(SUMM_DIR / "class_recall_precision_summary.csv")
    print(VIS_DIR / "confusion_matrix_key_conditions.png")
    print(VIS_DIR / "confusion_matrix_kd_pilot.png")
    print(VIS_DIR / "class_recall_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
