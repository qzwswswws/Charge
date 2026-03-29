# MI Algorithm Workbench Snapshot 2026-03-29

This snapshot preserves the current low-channel MI experiment outcomes from the local `MI_Algorithm_Workbench`.

## Included Scope

- Full-channel baseline summary
- `22 -> 8 -> 3 -> 2` channel `4-class` degradation summary
- `2-channel / 2-class` baseline summary
- Exploratory low-channel pilot summaries:
  - `lowchannel_v1`
  - `Batch 1`
  - `Batch 2`
- Core planning and rationale documents
- Key reproducibility scripts for Batch 2 and monitoring

## Key Findings

### 4-class degradation

- `22ch`: `Avg Best = 0.7342`, `Avg Aver = 0.6005`
- `8ch`: `Avg Best = 0.6597`, `Avg Aver = 0.5517`
- `3ch`: `Avg Best = 0.5911`, `Avg Aver = 0.5170`
- `2ch`: `Avg Best = 0.4973`, `Avg Aver = 0.4441`

### 2-channel / 2-class baseline

- `Avg Best = 0.7076`
- `Avg Aver = 0.6488`

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

## Notes

- This upload is a curated snapshot, not a full raw-log mirror.
- Large intermediate logs are intentionally omitted to keep the repository compact.
