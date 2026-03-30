# MI Algorithm Workbench Snapshot 2026-03-29

This snapshot preserves the current MI experiment outcomes from the local `MI_Algorithm_Workbench`, including the later `P5/P6/P7/P8` extensions that were completed after the first low-channel milestone.

## Included Scope

- Full-channel baseline summary
- `22 -> 8 -> 3 -> 2` channel `4-class` degradation summary
- `2-channel / 2-class` baseline summary
- Exploratory low-channel pilot summaries:
  - `lowchannel_v1`
  - `Batch 1`
  - `Batch 2`
- Knowledge distillation pilot and full `9-subject` expansion
- Model complexity analysis
- Confusion matrix and class-level error analysis
- Channel-combination pilot for low-channel electrode selection
- Multi-seed reproducibility summaries for `2a`
- Training-data fraction sensitivity pilot
- `BCI Competition IV 2b` external validation (`9` subjects, multi-seed)
- Minimal `EEGNet` classical deep-learning comparison
- Core planning, rationale, and thesis-writing summary documents
- Key reproducibility scripts for degradation, KD, profiling, confusion analysis, channel-combination pilot, `2b`, and `EEGNet`

## Key Findings

### 4-class degradation

- `22ch`: `Avg Best = 0.7342`, `Avg Aver = 0.6005`
- `8ch`: `Avg Best = 0.6597`, `Avg Aver = 0.5517`
- `3ch`: `Avg Best = 0.5911`, `Avg Aver = 0.5170`
- `2ch`: `Avg Best = 0.4973`, `Avg Aver = 0.4441`

### 2-channel / 2-class baseline

- `Avg Best = 0.7076`
- `Avg Aver = 0.6488`

### Multi-seed reproducibility

- `C3/C4 / 2-class / 3 seeds`: `Best = 0.7081 ± 0.0054`, `Aver = 0.6467 ± 0.0063`
- `C3/C4 / 4-class / 3 seeds`: `Best = 0.4995 ± 0.0035`, `Aver = 0.4439 ± 0.0033`
- `C3/Cz/C4 / 2-class / 5 subjects / 3 seeds`: `Best = 0.8546 ± 0.0021`, `Aver = 0.7874 ± 0.0032`

### 2-channel / 4-class KD full expansion

- Baseline: `Avg Best = 0.4973`, `Avg Aver = 0.4441`
- KD student: `Avg Best = 0.5077`, `Avg Aver = 0.4525`
- Interpretation: KD remained directionally effective across `9` subjects, but the gains stayed mild and should be treated as supplementary compensation rather than a mainline replacement

### Channel-combination pilot (`2-class`, `5` subjects)

- `c3c4`: `Avg Best = 0.7500`, `Avg Aver = 0.6975`
- `c3cz`: `Avg Best = 0.7625`, `Avg Aver = 0.6881`
- `czc4`: `Avg Best = 0.7750`, `Avg Aver = 0.7084`
- `c3czc4`: `Avg Best = 0.8542`, `Avg Aver = 0.7836`
- `c1czc2`: `Avg Best = 0.8167`, `Avg Aver = 0.7306`
- Interpretation: `Cz` contributes meaningful value, especially when moving from `2` to `3` channels; `C3/Cz/C4` is the strongest current central combination

### Training-data fraction sensitivity (`C3/C4 / 2-class`, `5` subjects)

- `25%`: `Best = 0.6903`, `Aver = 0.6201`
- `50%`: `Best = 0.7250`, `Aver = 0.6628`
- `75%`: `Best = 0.7472`, `Aver = 0.6909`
- `100%`: `Best = 0.7500`, `Aver = 0.6975`
- Interpretation: around `75%` training data already approaches the full-data baseline on the pilot set

### `BCI Competition IV 2b` external validation

- `9 subjects / single seed / C3-Cz-C4 / 2-class`: `Avg Best = 0.8285`, `Avg Aver = 0.7816`
- `9 subjects / 3 seeds / C3-Cz-C4 / 2-class`: `Best = 0.8284 ± 0.0024`, `Aver = 0.7787 ± 0.0027`
- Interpretation: the central 3-channel 2-class route remains strong and statistically stable on an external MI dataset

### `EEGNet` minimal pilot comparison (`2a`, `5` subjects)

- `22ch / 4class`: `Avg Best = 0.7521`, `Avg Aver = 0.6653`
- `C3/C4 / 2class`: `Avg Best = 0.7222`, `Avg Aver = 0.6797`
- Interpretation: EEGNet forms a valid classical deep-learning comparison, but does not overturn the current Conformer-based low-channel mainline

### Complexity and error analysis

- Channel reduction decreases compute and latency much more than parameter count
- `2ch / 2class` and `2ch / 4class` have near-identical deployment complexity, so the main advantage of `2class` is task-match rather than model size
- In `2ch / 4class`, the most severe degradation occurs in `Feet` recall and in `Left-Right` / `Feet-Tongue` confusion

### Stepwise optimization pilot comparison

Pilot subjects: `S1 / S3 / S5 / S8 / S9`

- Pilot baseline: `Avg Best = 0.7500`, `Avg Aver = 0.6975`
- Batch 1: `Avg Best = 0.7444`, `Avg Aver = 0.6849`
- Batch 2: `Avg Best = 0.7389`, `Avg Aver = 0.6891`

## Current Conclusion

The lightweight low-channel modifications have moved closer to the original baseline, but have not yet surpassed it on the pilot set. At the current checkpoint:

- `Batch 1` should be treated as `not passed`
- `Batch 2` improved over `Batch 1` but still did not exceed the pilot baseline
- The original `c3c4 / 2-class` baseline remains the primary low-channel reference
- KD is worth retaining as a supplementary `22ch teacher -> 2ch student` compensation path for `4-class`
- If hardware can expand to `3` central electrodes, `C3/Cz/C4` is now the strongest candidate
- For strict `2-channel` settings, `Cz/C4` is worth further verification against `C3/C4`
- `2b` results now show that the central `3-channel / 2-class` route is not only a `2a` artifact, but a stable cross-dataset finding
- `EEGNet` now provides a valid horizontal comparison, and the mainline conclusion still stands: on the key `C3/C4 / 2-class` task, the current Conformer route remains stronger

## Notes

- This upload is a curated snapshot, not a full raw-log mirror.
- Large intermediate logs are intentionally omitted to keep the repository compact.
