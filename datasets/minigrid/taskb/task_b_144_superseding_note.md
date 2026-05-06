# Task B Canonical Closeout Authority

- Default exam: `task_b_exam.jsonl`
- Operational exam alias: `task_b_exam_v71c_operational.jsonl`
- Current packaged rows: `144`
- Source: `datasets/exams_taskb_v10b_144_reanchored_candidate/task_b_exam.jsonl`
- Operational prompt: `v7.1c`
- Operational budget: `max_tokens_B=4096`
- Audit support:
  - `task_b_v10b_manifest.json`
  - `task_b_v10b_qc_report.md`
  - `task_b_v10b_uniqueness_audit.json`
  - `task_b_review_144_reanchored_zh.pdf`

This replaces the earlier 72-row repaired copy inside the canonical-closeout package.

## Operational decision

- `v7_legend + 4096` smoke failed:
  - `fail_json = 2`
  - `strict + recoverable = 10/12`
  - `finish_reason=length = 2`
- `v7.1c + 4096` smoke passed:
  - `fail_json = 0`
  - `strict + recoverable = 12/12`
  - `finish_reason=length = 0`

Therefore, the current Task B running line is:

- exam: `task_b_exam_v71c_operational.jsonl`
- prompt: `v7.1c`
- budget: `4096`

`v7_legend` remains useful as an interpretability-side reference, but it is no longer the operational full-run prompt.
