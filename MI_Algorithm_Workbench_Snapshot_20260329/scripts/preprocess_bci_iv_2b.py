"""
Preprocess BCI Competition IV 2b raw GDF files into strict_TE-style MAT files.

This script mirrors the local MATLAB reference in spirit:
1. Read each GDF session.
2. Use event 768 as trial onset.
3. Extract a 4-second segment starting 750 samples after event 768.
4. Keep only C3 / Cz / C4.
5. Apply a 4-40 Hz Chebyshev type-II band-pass filter.
6. Save as .mat files with keys: data, label.

Expected label MAT files contain a `classlabel` array and share the same
session stem, e.g.:
    raw:   B0104E.gdf
    label: B0104E.mat
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import mne
import numpy as np
import scipy.io
from scipy.signal import cheby2, filtfilt


EEG_PICKS = ["EEG:C3", "EEG:Cz", "EEG:C4"]
EVENT_CODE = "768"
OFFSET_SAMPLES = 750
TRIAL_SAMPLES = 1000
FS = 250.0


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess BCI IV 2b GDF into MAT")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="/home/woqiu/下载/git/MI_Algorithm_Workbench/datasets/raw_2b_gdf/BCICIV_2b_gdf",
        help="directory containing B0xxxx?.gdf files",
    )
    parser.add_argument(
        "--label_root",
        type=str,
        default="/home/woqiu/下载/git/MI_Algorithm_Workbench/datasets/true_labels_2b",
        help="directory containing B0xxxx?.mat label files",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/home/woqiu/下载/git/MI_Algorithm_Workbench/datasets/standard_2b_strict_TE",
        help="output directory for strict_TE MAT files",
    )
    parser.add_argument("--subject", type=int, default=None, help="subject id 1-9; omit to process all")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing MAT files")
    return parser.parse_args()


def subject_stems(subject: int) -> list[str]:
    return [f"B0{subject}0{i}{kind}" for i, kind in ((1, "T"), (2, "T"), (3, "T"), (4, "E"), (5, "E"))]


def load_labels(label_path: Path) -> np.ndarray:
    mat = scipy.io.loadmat(label_path)
    if "classlabel" not in mat:
        raise KeyError(f"{label_path} does not contain 'classlabel'")
    labels = np.asarray(mat["classlabel"]).reshape(-1)
    return labels.astype(np.uint8)


def extract_trials(raw_path: Path, labels: np.ndarray) -> np.ndarray:
    raw = mne.io.read_raw_gdf(str(raw_path), preload=True, verbose="ERROR")
    data = raw.get_data(picks=EEG_PICKS)
    annotations = raw.annotations
    onset_samples = []
    for onset, desc in zip(annotations.onset, annotations.description):
        if desc == EVENT_CODE:
            onset_samples.append(raw.time_as_index(onset)[0])

    if len(onset_samples) != len(labels):
        raise ValueError(
            f"{raw_path.name}: 768 events ({len(onset_samples)}) do not match labels ({len(labels)})"
        )

    trials = np.zeros((TRIAL_SAMPLES, len(EEG_PICKS), len(labels)), dtype=np.float32)
    for idx, event_sample in enumerate(onset_samples):
        start = event_sample + OFFSET_SAMPLES
        stop = start + TRIAL_SAMPLES
        if stop > data.shape[1]:
            raise ValueError(f"{raw_path.name}: trial {idx} exceeds signal length")
        trial = data[:, start:stop].T  # (1000, 3)
        trials[:, :, idx] = np.nan_to_num(trial, nan=0.0)

    return trials


def bandpass_4_40(trials: np.ndarray) -> np.ndarray:
    wn = [4 * 2 / FS, 40 * 2 / FS]
    b, a = cheby2(6, 60, wn, btype="bandpass")
    filtered = np.zeros_like(trials, dtype=np.float32)
    for idx in range(trials.shape[2]):
        filtered[:, :, idx] = filtfilt(b, a, trials[:, :, idx], axis=0).astype(np.float32)
    return filtered


def process_one(stem: str, raw_root: Path, label_root: Path, out_root: Path, overwrite: bool) -> dict[str, str]:
    raw_path = raw_root / f"{stem}.gdf"
    label_path = label_root / f"{stem}.mat"
    out_path = out_root / f"{stem}.mat"

    if not raw_path.exists():
        return {"stem": stem, "status": "missing_raw", "trials": "", "out_path": str(out_path)}
    if not label_path.exists():
        return {"stem": stem, "status": "missing_label", "trials": "", "out_path": str(out_path)}
    if out_path.exists() and not overwrite:
        labels = load_labels(label_path)
        return {"stem": stem, "status": "exists", "trials": str(len(labels)), "out_path": str(out_path)}

    labels = load_labels(label_path)
    trials = extract_trials(raw_path, labels)
    filtered = bandpass_4_40(trials)

    scipy.io.savemat(
        out_path,
        {
            "data": filtered.astype(np.float32),
            "label": labels.reshape(-1, 1).astype(np.uint8),
        },
    )
    return {"stem": stem, "status": "ok", "trials": str(len(labels)), "out_path": str(out_path)}


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    label_root = Path(args.label_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = [args.subject] if args.subject else list(range(1, 10))
    rows = []

    print(f"{'=' * 72}")
    print("  Preprocess BCI IV 2b")
    print(f"  Raw root:   {raw_root}")
    print(f"  Label root: {label_root}")
    print(f"  Out root:   {out_root}")
    print(f"  Subjects:   {subjects}")
    print(f"  Overwrite:  {args.overwrite}")
    print(f"{'=' * 72}\n")

    for subject in subjects:
        for stem in subject_stems(subject):
            print(f">>> Processing {stem}")
            row = process_one(stem, raw_root, label_root, out_root, args.overwrite)
            rows.append(row)
            print(f"    status={row['status']} trials={row['trials']} out={row['out_path']}")

    manifest_path = out_root / "preprocess_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "status", "trials", "out_path"])
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(r["status"] == "ok" for r in rows)
    exists = sum(r["status"] == "exists" for r in rows)
    failed = len(rows) - ok - exists
    print(f"\n{'=' * 72}")
    print(f"  Done. ok={ok} exists={exists} failed={failed}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
