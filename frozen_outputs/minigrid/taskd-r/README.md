# Frozen MiniGrid Task D-R Outputs

Each model directory contains:

- `responses.jsonl`: frozen raw model responses using `DR.*` UIDs.
- `score_per_row.jsonl`: scores regenerated from the published Task D-R exam.
- `score_report.json`: aggregate scorer output and coverage telemetry.

All three files are tied to:

`datasets/minigrid/taskd-r/task_d_exam.jsonl`

The frozen bundle contains ten paper models. Every model has 144 response UIDs,
an exact 144/144 join to the exam, and no missing response or API-error rows.
The response files are immutable evidence; reproduction writes fresh results to
a separate output directory.

The original internal bundle called the response-bearing files
`requests_D.jsonl`. They are published here as `responses.jsonl` because each
record contains the completed model response as well as request metadata.
