"""
Plot and summarize P4 channel-combination pilot results.

Inputs:
    results_summaries/channel_combo_pilot_2class_latest.csv

Outputs:
    results_summaries/channel_combo_pilot_2class_summary.csv
    visualization/channel_combo_pilot_2class.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "results_summaries" / "channel_combo_pilot_2class_latest.csv"
SUMMARY_CSV = ROOT / "results_summaries" / "channel_combo_pilot_2class_summary.csv"
OUTPUT_PNG = ROOT / "visualization" / "channel_combo_pilot_2class.png"


def load_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {"n_channels": None, "best": [], "aver": [], "subjects": 0}
    )

    for row in rows:
        cfg = row["channel_config"]
        grouped[cfg]["n_channels"] = int(row["n_channels"])
        grouped[cfg]["best"].append(float(row["best_acc"]))
        grouped[cfg]["aver"].append(float(row["aver_acc"]))
        grouped[cfg]["subjects"] += 1

    baseline_best = sum(grouped["c3c4"]["best"]) / len(grouped["c3c4"]["best"])
    baseline_aver = sum(grouped["c3c4"]["aver"]) / len(grouped["c3c4"]["aver"])

    summary: list[dict[str, object]] = []
    for cfg, values in grouped.items():
        avg_best = sum(values["best"]) / len(values["best"])
        avg_aver = sum(values["aver"]) / len(values["aver"])
        summary.append(
            {
                "channel_config": cfg,
                "n_channels": values["n_channels"],
                "subjects": values["subjects"],
                "avg_best_acc": avg_best,
                "avg_aver_acc": avg_aver,
                "delta_best_vs_c3c4": avg_best - baseline_best,
                "delta_aver_vs_c3c4": avg_aver - baseline_aver,
            }
        )

    summary.sort(key=lambda item: (-float(item["avg_best_acc"]), item["channel_config"]))
    return summary


def write_summary_csv(summary: list[dict[str, object]]) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "channel_config",
                "n_channels",
                "subjects",
                "avg_best_acc",
                "avg_aver_acc",
                "delta_best_vs_c3c4",
                "delta_aver_vs_c3c4",
            ],
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    "channel_config": row["channel_config"],
                    "n_channels": row["n_channels"],
                    "subjects": row["subjects"],
                    "avg_best_acc": f"{float(row['avg_best_acc']):.6f}",
                    "avg_aver_acc": f"{float(row['avg_aver_acc']):.6f}",
                    "delta_best_vs_c3c4": f"{float(row['delta_best_vs_c3c4']):+.6f}",
                    "delta_aver_vs_c3c4": f"{float(row['delta_aver_vs_c3c4']):+.6f}",
                }
            )


def plot(summary: list[dict[str, object]]) -> None:
    labels = [str(item["channel_config"]) for item in summary]
    best_vals = [float(item["avg_best_acc"]) for item in summary]
    aver_vals = [float(item["avg_aver_acc"]) for item in summary]
    colors = ["#c66b3d" if int(item["n_channels"]) == 3 else "#356c8c" for item in summary]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    fig.suptitle("P4 Channel-Combination Pilot (2-class, 5 subjects)", fontsize=13)

    for ax, values, title in zip(
        axes,
        [best_vals, aver_vals],
        ["Average Best Accuracy", "Average Mean Accuracy"],
    ):
        bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.7)
        ax.set_ylim(0.45, 0.92)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Accuracy")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#356c8c"),
        plt.Rectangle((0, 0), 1, 1, color="#c66b3d"),
    ]
    fig.legend(legend_handles, ["2-channel", "3-channel"], loc="upper right", frameon=False)
    fig.tight_layout(rect=[0, 0, 0.98, 0.94])
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    summary = build_summary(rows)
    write_summary_csv(summary)
    plot(summary)
    print(f"Summary saved to: {SUMMARY_CSV}")
    print(f"Figure saved to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
