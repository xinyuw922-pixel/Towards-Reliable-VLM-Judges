# GridWM-Judge: SCI Diagnosis Framework

**Research:** Stable but Wrong — Diagnosing State-Conditional Invariance Violations in VLM Judges

This document explains the SCI (State-Conditional Invariance) diagnosis framework used to evaluate VLM world-model judges on embodied AI tasks.

---

## For Paper Writers

**Start here:** [`paper_results/2026-04-26_ssot_non_gpt5/`](paper_results/2026-04-26_ssot_non_gpt5/)

This bundle contains the canonical diagnosis results: main result tables, per-instance judgments, canonical API responses, and the full analysis narrative.

---

## SCI Framework Overview

We investigate whether VLM-based world-model judges are stable and causally sound across different visual representations of the same trajectory. The SCI framework decomposes failures into three axes:

| Axis | Violation | Description | Metrics |
|------|-----------|-------------|---------|
| **C1** | Presentation Sensitivity | Judge responds to non-causal presentation changes (temporal order, visual style, framing language) | LES_T, JCR_T |
| **C4** | Causal Insensitivity | Judge ignores counterfactual (CF) interventions — the core of causal reasoning | IDR_cond, JAccneu, Degenerate |
| **C3** | Representation Exchangeability | Verdicts shift across composite vs. split-image visual rendering of the same trajectory | VCC_rep |

### Key Terms

- **LES_T** (Log-Evidence Sensitivity, Temporal): Ratio of P(correct|orig) to P(correct|rev). LES_T > 1 means the judge is sensitive to temporal order, which is causally irrelevant. A value of 2.0 means the judge is twice as likely to be correct on original-order trajectories vs. reversed-order ones.

- **JCR_T** (Judgment Consistency Rate, Temporal): P(j_pred is correct|orig) / P(j_pred is correct|rev). Measures whether the judge's verdict quality is stable across temporal orderings.

- **IDR_cond** (Intervention Detection Rate, Conditional): P(verdict flips correctly | CF intervention applied). Measures whether the judge responds correctly to counterfactual interventions that should change the outcome.

- **JAccneu** (Judgment Accuracy on Neutralized trials): Fraction of CF/NoCue trials where the judge correctly detects that the evidence does not support a "success" verdict.

- **Degenerate**: Marks models that always predict the same label regardless of evidence. `all_fail` = always "fail", `all_success` = always "success". Both are pathological — they are maximally stable but causally blind.

- **VCC_rep** (Verdict Cross-Classification, Representation): P(y_comp = y_split | same UID, both responses are valid). Measures whether the judge's verdicts are consistent across two different visual representations (composite single-image vs. split multi-panel) of the same trajectory.

---

## The Three Failure Patterns

### 1. Stability Trap (C4: Causal Insensitivity)

Models with high JCR (stable verdicts) but zero causal sensitivity. They are stable not because they reliably identify causal structure, but because they default to a fixed verdict regardless of evidence.

**Example:** qwen3-32b and qwen3-235b-a22b achieve JCR_T = 1.0 (perfect stability) but show all-fail degenerate behavior — they always predict "fail" regardless of whether the agent actually succeeded.

### 2. Presentation Sensitivity (C1: Temporal/Framing Leakage)

Models whose verdicts change based on non-causal presentation details like temporal order or framing language.

**Example:** gpt-4o achieves LES_T = 2.079 — it is more than twice as likely to give a correct verdict when the trajectory is shown in original temporal order vs. reversed order. This means the judge is relying on temporal sequence heuristics rather than causal reasoning about the final state.

### 3. Representation Fragility (C3: Cross-Representation Divergence)

The same model, same gold labels, different visual rendering yields substantially different causal profiles.

**The strongest finding:** gpt-4o's JAccneu flips from 0.000 (all-fail, composite representation) to 0.778 (split-image representation). VCC_rep = 0.706 means ~29% of verdicts flip between the two representations.

---

## Diagnostic Matrix

| Model | Rep | C1 Violation | C4 Insensitivity | C3 Fragility | Overall Pattern |
|---|---|---|---|---|---|
| kimi-k2.5 | split | none | mild (IDR=1.0) | — | Near-ideal |
| gemini-3-flash-preview | both | moderate (LES=1.10) | mild (IDR=0.56) | moderate | Best performer |
| gpt-5.4 | both | mild-moderate | mild | moderate | Moderate |
| gpt-4o | split | moderate (LES=1.10) | moderate (IDR=0.93) | — | Improved with rep |
| gpt-4o | comp | severe (LES=2.08) | severe (all_fail) | severe | Multiple failures |
| gemini-2.0-flash | both | none | severe (all_fail) | severe | Representation Fragility |
| qwen3-8b | both | none | severe (flip) | severe | Representation Fragility |
| qwen3-32b | both | none | severe (all_fail) | moderate | Stability Trap |
| qwen3-235b-a22b | both | none | severe (all_fail) | moderate | Stability Trap |

---

## Reproducing the Results

```bash
# Step 1: Rebuild the SSOT main tables from raw responses
python scripts/build_non_gpt5_ssot_main_table.py
# Output: paper_results/2026-04-26_ssot_non_gpt5/main_tables/main_results.{md,csv,json} + task_b_results.csv + task_e_results.csv

# Step 2: Regenerate the full results bundle
python paper_results/2026-04-26_ssot_non_gpt5/scripts/generate_bundle.py
```

---

## Dataset and Setup

| Aspect | Detail |
|---|---|
| **Models** | 10 models (gpt-5 excluded per SSOT policy; gpt-5.4 is a distinct model and is included): claude-sonnet-4-6, gemini-2.0-flash, gemini-2.5-flash-lite, gemini-3-flash-preview, gpt-4o, gpt-5.4, kimi-k2.5, qwen3-8b, qwen3-32b, qwen3-235b-a22b |
| **Composite Representation** | phase71 canonical runs; Task A: n=103, Task B: n=144, Task C: n=486 probe rows, Task D: n=144, Task E: n=325 |
| **Split-Image Representation** | phase91 Task A + phase97 Task C split full matrix; C4 fallback from Task C baseline panel |
| **Task B Supplement** | phase71 canonical composite runs; weighted perception IoU scoring (agent_pos=0.3, carrying=0.15, front_cell=0.25, objects_jaccard=0.2; acc_threshold=0.9). No temporal/visual/framing variants. |
| **Task E Supplement** | phase71 canonical composite (n=325) + phase96 smoke split (n=24; only gemini-2.5-flash-lite and gemini-3-flash-preview have split-image data). Exact-match accuracy. |
| **SSOT Scoring** | All metrics computed via `export_unified_outputs._build_per_row_df()` on identical scoring layer |

---

## Known Limitations

- All metrics are **pilot-scale** (n ≈ 12 groups per JAccneu condition). Confidence intervals are wide.
- Split-Image C4 metrics use Task C baseline panel fallback (no direct split-image Task D run exists in the current benchmark line).
- gpt-5 results are excluded from the main tables.
