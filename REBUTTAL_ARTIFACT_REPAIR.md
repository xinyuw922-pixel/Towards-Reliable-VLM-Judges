# Rebuttal Artifact Repair

This branch repairs the public artifact without rewriting the historical
release. It is based on `main` at commit
`1e92e6fd2d5a75173d44f41a58fec39728268152`.

## Scope

The repair has four goals:

1. Restore the scoring and inference modules referenced by the public README.
2. Publish one internally consistent MiniGrid Task D-R bundle.
3. Make frozen responses reproducible as per-row scores and aggregate metrics.
4. Validate every published data dependency and UID join in a clean checkout.

No intermediate rebuttal experiments are deleted or folded into this branch.
Only files needed to audit or reproduce the released results are published.

## Frozen Data Contract

The authoritative MiniGrid Task D artifact is the reference-aided Task D-R
exam. Its row identifiers use the `DR.*` namespace.

| Artifact | Required condition |
|---|---|
| Exam | 144 JSONL rows |
| Independent groups | 12 total, two for each of six environments |
| Variants | `full`, `nocue`, and `cf` |
| Presentation conditions | four conditions per group and variant |
| Gold labels | 48 `Success` full, 48 `Success` nocue, 48 `Fail` cf |
| Model responses | 144 rows per model |
| UID join | response UID set exactly equals exam UID set |
| Scored rows | 144 rows per model, with no missing response |

Legacy `D.*` exam rows are not interchangeable with Task D-R responses. A
score file produced by joining `DR.*` responses to a `D.*` exam is invalid,
even if both files contain 144 rows.

## Source Provenance

The repair is assembled from the frozen experiment sources retained by the
authors:

- Task D-R exam and images:
  `datasets_canonical_closeout_2026-04-03/taskd-r/`
- Frozen paper-model responses:
  `FINAL/02_vlm_outputs/ssot_responses/minigrid/composite/taskD/`
- Current scoring implementation:
  `FINAL/04_code/scripts/`

The published branch records content hashes and row-level validation results;
local absolute source paths are provenance notes, not runtime dependencies.

## Known Defects on `main`

- The README references full-pipeline shell scripts that are not tracked.
- `scripts/04_inference/run_inference.py` imports a missing
  `scripts.schema` package.
- `compute_idr.py` and `compute_les.py` are referenced but absent.
- Released Task D per-row files were joined against legacy `D.*` gold while
  the frozen responses use `DR.*` UIDs, producing 144 missing responses.
- Task D per-row output does not preserve all presentation metadata required
  for presentation-aware aggregation.

## Acceptance Checks

Before this branch is considered repaired:

1. Inference and scoring entry points must import successfully.
2. Every Task D-R model bundle must pass exact UID-set validation.
3. Re-scoring must produce 144 rows and zero missing responses per model.
4. Aggregate results must be generated only from the published frozen bundle.
5. README commands must run in a clean clone without private absolute paths.
6. A strict validator must fail on namespace, count, hash, or join mismatch.

## Status

This document defines the repair contract. Implementation and data commits that
follow it are intentionally kept separate for auditability.
