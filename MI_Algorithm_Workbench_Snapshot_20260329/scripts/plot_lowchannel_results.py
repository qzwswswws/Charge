"""
生成低通道退化与二分类对比图。

输出:
- visualization/degradation_curve_4class.png
- visualization/c3c4_4class_vs_2class.png

用法:
    python visualization/plot_lowchannel_results.py
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/woqiu/下载/git/MI_Algorithm_Workbench")
RESULTS_DIR = ROOT / "results_summaries"
OUT_DIR = ROOT / "visualization"


def load_results():
    df4 = pd.read_csv(RESULTS_DIR / "degradation_4class.csv")
    df2 = pd.read_csv(RESULTS_DIR / "degradation_2class.csv")
    return df4, df2


def make_4class_curve(df4: pd.DataFrame):
    order = ["full", "central8", "c3czc4", "c3c4"]
    label_map = {
        "full": "22ch / 4-class",
        "central8": "8ch / 4-class",
        "c3czc4": "3ch / 4-class",
        "c3c4": "2ch / 4-class",
    }

    grouped = (
        df4.groupby(["channel_config", "n_channels"], as_index=False)[["best_acc", "aver_acc"]]
        .mean()
        .copy()
    )
    grouped["order"] = grouped["channel_config"].map({name: i for i, name in enumerate(order)})
    grouped = grouped.sort_values(["order", "n_channels"], ascending=[True, False])

    x_labels = [label_map[cfg] for cfg in grouped["channel_config"]]
    best_vals = grouped["best_acc"] * 100.0
    aver_vals = grouped["aver_acc"] * 100.0

    plt.figure(figsize=(9.5, 5.5))
    plt.plot(x_labels, best_vals, marker="o", linewidth=2.4, color="#1b4965", label="Avg Best Acc")
    plt.plot(x_labels, aver_vals, marker="s", linewidth=2.0, color="#ca6702", label="Avg Aver Acc")

    for i, value in enumerate(best_vals):
        plt.text(i, value + 0.8, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color="#1b4965")
    for i, value in enumerate(aver_vals):
        plt.text(i, value - 2.4, f"{value:.2f}", ha="center", va="top", fontsize=9, color="#ca6702")

    plt.title("4-Class Channel Degradation on BCI IV 2a", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.ylim(20, 80)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "degradation_curve_4class.png", dpi=220, bbox_inches="tight")
    plt.close()


def make_c3c4_compare(df4: pd.DataFrame, df2: pd.DataFrame):
    df4_c3c4 = df4[df4["channel_config"] == "c3c4"]
    best_4 = df4_c3c4["best_acc"].mean() * 100.0
    aver_4 = df4_c3c4["aver_acc"].mean() * 100.0
    best_2 = df2["best_acc"].mean() * 100.0
    aver_2 = df2["aver_acc"].mean() * 100.0

    labels = ["2ch / 4-class", "2ch / 2-class"]
    best_vals = [best_4, best_2]
    aver_vals = [aver_4, aver_2]
    x = [0, 1]
    width = 0.34

    plt.figure(figsize=(8.4, 5.4))
    plt.bar([i - width / 2 for i in x], best_vals, width=width, color="#1b4965", label="Avg Best Acc")
    plt.bar([i + width / 2 for i in x], aver_vals, width=width, color="#ca6702", label="Avg Aver Acc")

    for i, value in enumerate(best_vals):
        plt.text(i - width / 2, value + 0.8, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    for i, value in enumerate(aver_vals):
        plt.text(i + width / 2, value + 0.8, f"{value:.2f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(x, labels)
    plt.ylabel("Accuracy (%)")
    plt.ylim(20, 85)
    plt.title("C3/C4: 4-Class vs 2-Class", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "c3c4_4class_vs_2class.png", dpi=220, bbox_inches="tight")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df4, df2 = load_results()
    make_4class_curve(df4)
    make_c3c4_compare(df4, df2)
    print("Saved:", OUT_DIR / "degradation_curve_4class.png")
    print("Saved:", OUT_DIR / "c3c4_4class_vs_2class.png")


if __name__ == "__main__":
    main()
