# Task D-R Scoring Audit

## Result

The public `main` branch paired legacy `D.*` gold rows with `DR.*` responses.
The UID sets have no intersection, so the released Task D per-row files contain
144 `missing_response` rows per model. This repair publishes the matching
Task D-R exam and regenerates all scores from the raw responses.

Nine models reproduce the metrics in the internal FINAL table after the UID
repair. Kimi-K2.5 exposes an additional parser defect and therefore requires an
author decision before paper tables are updated.

## Kimi-K2.5 Difference

| Metric | Previous FINAL audit | Unified scorer |
|---|---:|---:|
| Full accuracy / `p_active` | 0.166667 | 0.833333 |
| NoCue accuracy | 0.000000 | 0.750000 |
| CF accuracy | 0.083333 | 0.666667 |
| `IDR_raw` | 0.000000 | 0.583333 |
| `IDR_cond` | 0.000000 | 0.700000 |
| Valid paired groups | 1 | 12 |

The previous formula-audit helper accepted a prediction only when the raw
response started with `Success` or `Fail`. Most Kimi responses contain an
explanation followed by a final standalone verdict, so that helper discarded
11 of 12 baseline pairs.

The shared verdict parser was intended and documented to use the final verdict
when both success and failure terms occur. Its implementation instead selected
the first keyword, which commonly described the successful reference
trajectory. This branch fixes the implementation to match the documented rule
and includes regression tests.

Under the corrected parser, 11 Kimi responses are strict, 132 are recoverable,
and one truncated response is unparsable. The unparsable response remains
incorrect, consistent with the benchmark policy.

## Author Review Required

Before using this branch for a rebuttal or paper revision, the authors should:

1. Confirm that the stated policy is final-verdict keyword parsing, with
   unparsable and truncated outputs counted as incorrect.
2. Approve replacing the previous Kimi Task D-R values with the unified-scorer
   values in `frozen_outputs/minigrid/taskd-r/taskdr_metrics.csv`.
3. Check any paper table or figure containing Kimi Task D values.

No new API call is needed for this correction; both values derive from the same
published raw responses.
