# Responses And Answers Index

This note records where canonical answers live and where copied model-response artifacts live inside the MiniGrid closeout root.

## Canonical Answers

Answers remain authoritative in the task-local exam JSONL files:

- Task A: `taska/task_a_exam.jsonl`
- Task B: `taskb/task_b_exam.jsonl`
- Task C: `taskc/task_c_exam.jsonl`
- Task D-R: `taskd-r/task_d_exam.jsonl`
- Task D legacy: `taskd/task_d_exam.jsonl`
- Task D-F: `taskd-f/task_d_exam.jsonl`
- Task E: `taske/task_e_exam_v13.jsonl`

Answer field notes:

- Task A: `answer`
- Task B: `answer`
- Task C: `answer`
- Task D / D-R / D-F: `answer`
- Task E: `answer`

## Copied Model Responses

The closeout root now includes a copied full-matrix response bundle:

- `model_runs/phase71_full_matrix_2026-04-04/`

This bundle was copied from:

- `/home/wxy/GridWM-Judge/runs/phase71_full_matrix_2026-04-04`

Typical contents per task/model directory:

- `job_summary.json`
- `score_report.json`
- `responses/openaicompatible_*/requests_*.jsonl`
- `responses/_requests_from_exam/requests_*.jsonl`

Examples:

- Task A / GPT-5.4 raw responses:
  - `model_runs/phase71_full_matrix_2026-04-04/A/gpt-5.4/responses/openaicompatible_zhizengzeng_gpt-5.4/requests_A.jsonl`
- Task C / Gemini-3-Flash-Preview raw responses:
  - `model_runs/phase71_full_matrix_2026-04-04/C/gemini-3-flash-preview/responses/openaicompatible_zhizengzeng_gemini-3-flash-preview/requests_C.jsonl`
- Task D / GPT-4o raw responses:
  - `model_runs/phase71_full_matrix_2026-04-04/D/gpt-4o/responses/openaicompatible_zhizengzeng_gpt-4o/requests_D.jsonl`

## Supplemental Reruns

The closeout root also includes a targeted MiniGrid Task A rerun for Gemini-3-Flash-Preview:

- `model_runs/phase71_rerun_2026-04-10/A/gemini-3-flash-preview_maxA128/`

Purpose:

- mitigate the heavy `finish_reason=length` issue observed in the original
  `phase71_full_matrix_2026-04-04/A/gemini-3-flash-preview` run by increasing
  `Task A` output budget from `--max_tokens_A 32` to `--max_tokens_A 128`

Key files:

- rerun score report:
  - `model_runs/phase71_rerun_2026-04-10/A/gemini-3-flash-preview_maxA128/score_report.json`
- rerun raw responses:
  - `model_runs/phase71_rerun_2026-04-10/A/gemini-3-flash-preview_maxA128/responses/openaicompatible_zhizengzeng_gemini-3-flash-preview_chat/requests_A.jsonl`

Observed outcome:

- accuracy improved from `0.184` in the original run to `0.233` in the rerun
- unique response finish reasons after deduplication were approximately:
  - `stop = 97`
  - `length = 5`
  - `error = 1`

## Scope

This copied bundle is intended as a closeout-local mirror for response inspection. Task-local exam JSONL files remain the authority for gold answers.
