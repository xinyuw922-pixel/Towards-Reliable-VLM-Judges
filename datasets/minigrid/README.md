# Canonical Closeout Datasets

This directory is the clean authority root for current closeout-stage exams.

## Rules

1. One task, one subdirectory.
2. Future smoke and full runs should read from this root, not from legacy `datasets/exams_*` paths.
3. Historical datasets remain valid as provenance, but they are not the default run targets anymore.
4. Use `exam_authority.json` as the machine-readable source of truth.

## Layout

- `taska/`
- `taskb/`
- `taskc/`
- `taskd/`
- `taskd-f/`
- `taskd-r/`
- `taske/`

Each task directory contains:

- the default runnable exam file for that task
- task-local authority metadata
- any required images and supporting artifacts

## Default runnable exams

- Task A: `taska/task_a_exam.jsonl`
- Task B: `taskb/task_b_exam.jsonl`
- Task C: `taskc/task_c_exam.jsonl`
- Task D: `taskd-r/task_d_exam.jsonl`
- Task D legacy: `taskd/task_d_exam.jsonl`
- Task E: `taske/task_e_exam_v13.jsonl`

## Important caveats

- Task A: a fresh regeneration on 2026-04-03 produced 208 rows, not the historical 103-row canonical artifact. The clean root therefore keeps the current 103-row authority copy instead of silently upgrading Task A.
- Task B: the clean root now packages the 144-row reanchored candidate. The original blocked 72-row candidate and the older repaired 72-row rerun target remain historical provenance only.
  Operational running should use `taskb/task_b_exam_v71c_operational.jsonl` with prompt `v7.1c` and `max_tokens_B=4096`.
- Task C: the default runnable exam is the later 3-env probe-rich package (`keycorridor`, `multiroom`, `redblue`) with framing `v2-mid` added on the `orig.clean` slice. The older 324-row probe-rich baseline is preserved as `task_c_exam_probe_rich_baseline.jsonl`, and the older 6-env Path-B artifact is preserved as `task_c_exam_legacy_pathb.jsonl`.
- Task D: the mainline authority has been promoted to the expanded reference-aided `taskd-r/` package. It now includes `full / nocue / cf` query rows, a `style` visual probe, and `neu / pos / neg` prompt variants.
- Task D legacy: the earlier direct-judgment 144-row package remains in `taskd/` as a demoted historical/ablation line.
- Task D-F: appendix-style framing audit built from the direct-judgment Task D `full.orig.clean + cf.orig.clean` rows.
- Task D-R: now serves as the mainline Task D authority rather than a side candidate.
- Task E: regenerated as the current `v13` main candidate. Auxiliary slices such as strengthening, mandatory, and minismoke are intentionally kept out of the default full-run package.
