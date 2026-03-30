# P10 PhysioNet eegmmidb Preparation Note

## 1. Why this dataset is worth trying

PhysioNet `EEG Motor Movement/Imagery Dataset (eegmmidb)` is a reasonable next-step external dataset because it remains on the motor imagery mainline while adding:

- more subjects (`109`)
- a larger electrode montage (`64` channels)
- a public and well-documented official source

Compared with `OpenBMI`, it is easier to obtain from an official site and has a very clear run protocol.

Official dataset page:
- <https://physionet.org/content/eegmmidb/1.0.0/>

## 2. Official protocol facts confirmed

From the official PhysioNet page:

- sampling rate: `160 Hz`
- channels: `64`
- total runs per subject: `14`
- imagery-related left/right fist runs: `R04`, `R08`, `R12`
- imagery-related both-fists/both-feet runs: `R06`, `R10`, `R14`
- event labels:
  - `T0`: rest
  - `T1`: left fist for runs `3/4/7/8/11/12`, both fists for runs `5/6/9/10/13/14`
  - `T2`: right fist for runs `3/4/7/8/11/12`, both feet for runs `5/6/9/10/13/14`

This means the cleanest first experiment is:

- `2-class MI`
- use runs `R04/R08/R12`
- map `T1 -> left`, `T2 -> right`

## 2.5. Important difference from BCI Competition datasets

This point must be stated clearly in later writing:

- `eegmmidb` is **not** the same dataset as `BCI Competition IV 2a/2b`
- even when both are used for motor imagery, the **single-subject protocol length is different**

For the current PhysioNet pilot protocol:

- per run we use only `T1/T2`
- each selected run contributes `15` target trials
- current subject-level protocol uses `R04`, `R08`, `R12`
- therefore each subject contributes only `45` target trials in total
  - `30` training trials from `R04 + R08`
  - `15` test trials from `R12`

This is substantially shorter than the current `BCI Competition IV` mainline protocols used in this project.

So later in the thesis, PhysioNet results should be described as:

- an **external-dataset transfer check**
- a **shorter per-subject protocol**
- a result that is **not directly numerically equivalent** to `2a/2b`

In other words, its value is mainly:

- proving the pipeline can generalize beyond the competition datasets
- not replacing the main quantitative baseline built on `2a/2b`

## 3. Current local progress

Minimal download and read test has already passed for subject `S001`:

- raw root:
  [physionet_eegmmidb_raw](/home/woqiu/下载/git/MI_Algorithm_Workbench/datasets/physionet_eegmmidb_raw)
- downloaded files:
  - `S001R04.edf`
  - `S001R04.edf.event`
  - `S001R08.edf`
  - `S001R08.edf.event`
  - `S001R12.edf`
  - `S001R12.edf.event`

Local verification results:

- `EDF` download works from the official PhysioNet file server
- `MNE` can read the files successfully
- detected sample rate: `160.0`
- detected channels: `64`
- target central channels are present and readable:
  - `C3..`
  - `Cz..`
  - `C4..`
- annotations are readable and contain `T0/T1/T2`
- each selected run contributes `15` usable `T1/T2` target trials

## 4. New preprocessing entry

A minimal preprocessing script has been added:

- [preprocess_physionet_eegmmidb.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/preprocessing/preprocess_physionet_eegmmidb.py)

Current default behavior:

- target subject is specified by `--subject`
- target runs default to `4,8,12`
- default channel mode is `c3czc4`
- extract `4 s` epochs from `T1/T2` onset
- apply `4-40 Hz` band-pass filtering
- save one MAT file per run

Output root:

- [standard_physionet_eegmmidb](/home/woqiu/下载/git/MI_Algorithm_Workbench/datasets/standard_physionet_eegmmidb)

## 5. Recommended next step

The most sensible next move is not full-dataset download yet. It is:

1. preprocess `S001 / R04,R08,R12`
2. inspect the generated MAT files
3. decide the experiment protocol

Recommended first protocol:

- `C3/Cz/C4`
- `2-class`
- `R04 + R08` as training candidates
- `R12` as held-out run

This is not yet the final official benchmark protocol, but it is the fastest way to check whether the dataset can be aligned with the current low-channel MI pipeline.

This protocol is intentionally lightweight, but it also means the per-subject data volume is smaller than the current `BCI Competition IV` experiments. That caveat should be made explicit whenever PhysioNet results are compared with `2a/2b`.
