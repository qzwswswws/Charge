"""
Preprocess PhysioNet EEG Motor Movement/Imagery Dataset EDF files into
lightweight MAT files for downstream experiments.

Current default target:
1. Use imagined left/right fist runs: R04, R08, R12.
2. Extract 4-second epochs starting at T1/T2 onset.
3. Optionally keep only C3/Cz/C4 or retain all 64 channels.
4. Apply a 4-40 Hz Chebyshev type-II band-pass filter.
5. Save one MAT per run with keys: data, label, fs, run, subject, channels.

Notes:
- PhysioNet eegmmidb stores annotations directly in EDF. T1/T2 meanings depend
  on run type. For runs 4/8/12, T1=left fist imagery and T2=right fist imagery.
- The output layout is intentionally simple; training protocol can be decided
  later (for example, leave-one-run-out or run-level train/test splits).
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


DEFAULT_RUNS = [4, 8, 12]
FS = 160.0
TRIAL_SECONDS = 4.0
TRIAL_SAMPLES = int(FS * TRIAL_SECONDS)
EVENT_MAP = {"T1": 0, "T2": 1}
THREE_CH = ["C3", "CZ", "C4"]


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess PhysioNet eegmmidb EDF into MAT")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="/home/woqiu/下载/git/MI_Algorithm_Workbench/datasets/physionet_eegmmidb_raw",
        help="directory containing PhysioNet subject folders such as S001/",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/home/woqiu/下载/git/MI_Algorithm_Workbench/datasets/standard_physionet_eegmmidb",
        help="output directory for MAT files",
    )
    parser.add_argument("--subject", type=int, required=True, help="subject id, e.g. 1 for S001")
    parser.add_argument(
        "--runs",
        type=str,
        default="4,8,12",
        help="comma-separated run ids, default is imagined left/right runs 4,8,12",
    )
    parser.add_argument(
        "--channel_mode",
        choices=("c3czc4", "all64"),
        default="c3czc4",
        help="retain only C3/Cz/C4 or all 64 channels",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing MAT files")
    return parser.parse_args()


def parse_runs(run_arg: str) -> list[int]:
    runs = []
    for item in run_arg.split(","):
        item = item.strip()
        if not item:
            continue
        run = int(item)
        if run < 1 or run > 14:
            raise ValueError(f"Invalid run id: {run}")
        runs.append(run)
    if not runs:
        raise ValueError("No runs specified")
    return runs


def normalize_channel(name: str) -> str:
    return name.replace(".", "").replace(" ", "").upper()


def pick_indices(raw: mne.io.BaseRaw, channel_mode: str) -> tuple[list[int], list[str]]:
    if channel_mode == "all64":
        return list(range(raw.info["nchan"])), raw.ch_names

    normalized = [normalize_channel(ch) for ch in raw.ch_names]
    indices = []
    picked_names = []
    for target in THREE_CH:
        try:
            idx = normalized.index(target)
        except ValueError as exc:
            raise KeyError(f"Could not find channel {target} in {raw.ch_names}") from exc
        indices.append(idx)
        picked_names.append(raw.ch_names[idx])
    return indices, picked_names


def bandpass_4_40(data: np.ndarray) -> np.ndarray:
    wn = [4 * 2 / FS, 40 * 2 / FS]
    b, a = cheby2(6, 60, wn, btype="bandpass")
    filtered = np.zeros_like(data, dtype=np.float32)
    for idx in range(data.shape[2]):
        filtered[:, :, idx] = filtfilt(b, a, data[:, :, idx], axis=0).astype(np.float32)
    return filtered


def extract_trials(raw_path: Path, channel_mode: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw = mne.io.read_raw_edf(str(raw_path), preload=True, verbose="ERROR")
    pick_indices_list, picked_names = pick_indices(raw, channel_mode)
    data = raw.get_data(picks=pick_indices_list)

    epochs = []
    labels = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        if desc not in EVENT_MAP:
            continue
        start = raw.time_as_index(onset)[0]
        stop = start + TRIAL_SAMPLES
        if stop > data.shape[1]:
            continue
        trial = data[:, start:stop].T
        epochs.append(np.nan_to_num(trial, nan=0.0).astype(np.float32))
        labels.append(EVENT_MAP[desc])

    if not epochs:
        raise ValueError(f"{raw_path.name}: no T1/T2 epochs extracted")

    stacked = np.stack(epochs, axis=2)
    return stacked, np.asarray(labels, dtype=np.uint8), picked_names


def process_one(raw_path: Path, out_path: Path, subject: int, run: int, channel_mode: str, overwrite: bool):
    if out_path.exists() and not overwrite:
        mat = scipy.io.loadmat(out_path)
        n_trials = int(np.asarray(mat["label"]).reshape(-1).shape[0])
        return {"subject": subject, "run": run, "status": "exists", "trials": n_trials, "out_path": str(out_path)}

    epochs, labels, channels = extract_trials(raw_path, channel_mode)
    filtered = bandpass_4_40(epochs)

    scipy.io.savemat(
        out_path,
        {
            "data": filtered.astype(np.float32),
            "label": labels.reshape(-1, 1).astype(np.uint8),
            "fs": np.array([[FS]], dtype=np.float32),
            "run": np.array([[run]], dtype=np.int32),
            "subject": np.array([[subject]], dtype=np.int32),
            "channels": np.asarray(channels, dtype=object).reshape(-1, 1),
        },
    )
    return {"subject": subject, "run": run, "status": "ok", "trials": int(labels.shape[0]), "out_path": str(out_path)}


def main():
    args = parse_args()
    runs = parse_runs(args.runs)
    subject_tag = f"S{args.subject:03d}"

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root) / subject_tag / args.channel_mode
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 72}")
    print("  Preprocess PhysioNet eegmmidb")
    print(f"  Subject:      {subject_tag}")
    print(f"  Runs:         {runs}")
    print(f"  Channel mode: {args.channel_mode}")
    print(f"  Raw root:     {raw_root}")
    print(f"  Out root:     {out_root}")
    print(f"{'=' * 72}\n")

    rows = []
    for run in runs:
        stem = f"{subject_tag}R{run:02d}"
        raw_path = raw_root / subject_tag / f"{stem}.edf"
        out_path = out_root / f"{stem}.mat"
        if not raw_path.exists():
            rows.append(
                {"subject": args.subject, "run": run, "status": "missing_raw", "trials": "", "out_path": str(out_path)}
            )
            print(f">>> {stem}: missing raw")
            continue

        print(f">>> Processing {stem}")
        row = process_one(raw_path, out_path, args.subject, run, args.channel_mode, args.overwrite)
        rows.append(row)
        print(f"    status={row['status']} trials={row['trials']} out={row['out_path']}")

    manifest_path = out_root / "preprocess_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "run", "status", "trials", "out_path"])
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
