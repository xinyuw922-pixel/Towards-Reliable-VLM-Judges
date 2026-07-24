# Towards Reliable VLM Judges

Code and frozen artifacts for state-conditional invariance and
presentation-aware diagnostics in MiniGrid and MiniWorld.

This rebuttal repair branch prioritizes an auditable path from raw model
responses to MiniGrid Task D-R metrics. It preserves the historical release
while correcting missing code, a Task D/Task D-R UID mismatch, and a
recoverable-verdict parser defect.

## Reproduce Frozen Task D-R

Python 3.10 or newer is sufficient. The frozen reproduction path uses only the
standard library.

```bash
git clone --branch rebuttal-artifact-repair-2026-07-24 \
  https://github.com/xinyuw922-pixel/Towards-Reliable-VLM-Judges.git
cd Towards-Reliable-VLM-Judges

python scripts/validate_release.py
python scripts/reproduce_frozen_results.py \
  --output-dir /tmp/gridwm-taskdr-reproduced
```

The second command:

1. validates the exam, images, hashes, model set, and exact UID joins;
2. re-scores all ten frozen response files;
3. compares every regenerated per-row score with the published score;
4. regenerates the Task D-R metric table; and
5. compares it with the frozen metric table.

Successful completion ends with:

```text
PASS: reproduced 10 models and metrics
```

## Frozen Artifact

```text
datasets/minigrid/taskd-r/
├── task_d_exam.jsonl
├── task_d_r_exam.jsonl
├── manifest.json
├── authority_manifest.json
├── prompt_note.md
└── images/taskDR/

frozen_outputs/minigrid/taskd-r/
├── SHA256SUMS
├── taskdr_metrics.csv
└── <model>/
    ├── responses.jsonl
    ├── score_per_row.jsonl
    └── score_report.json
```

Task D-R contains 144 rows:

| Dimension | Distribution |
|---|---|
| Environments | 6, with 24 rows each |
| Independent query groups | 12, with 2 per environment |
| Variants | 48 Full, 48 NoCue, 48 CF |
| Gold labels | 96 Success, 48 Fail |
| Presentation conditions | clean-neutral, clean-positive, clean-negative, style-neutral |
| Models | 10 frozen response bundles |

Every model response set has an exact 144/144 UID join to the `DR.*` exam and
zero missing-response or API-error rows.

## Scoring

The SSOT scorer reads only an exam directory and raw response JSONL:

```bash
python scripts/05_scoring/score_exam.py \
  --exam_dir datasets/minigrid/taskd-r \
  --responses frozen_outputs/minigrid/taskd-r/gpt-4o/responses.jsonl \
  --out /tmp/gpt-4o-score-report.json \
  --per_row_out /tmp/gpt-4o-score-per-row.jsonl
```

Regenerate the Task D-R metric table from per-row scores:

```bash
python scripts/05_scoring/summarize_taskdr.py \
  --scores-root frozen_outputs/minigrid/taskd-r \
  --output /tmp/taskdr_metrics.csv
```

Task D-R baseline metrics use one neutral, clean, original row per group:

- `p_active`: fraction of groups judged Success on Full.
- `IDR_raw`: fraction judged Success on Full and Fail on matched CF.
- `IDR_cond`: `IDR_raw / p_active` when at least one Full group is active.
- `NS` and `JCR`: paired stability across framing or visual presentations.
- `LES`: the frozen Task D-R table uses the deterministic L3 McNemar
  effect-size rule.

Invalid, empty, truncated, refused, or otherwise unparsable outputs count as
incorrect. Recoverable responses use the model's concluding verdict when both
success and failure terms occur.

## Inference

The restored inference entry point is:

```bash
python scripts/04_inference/run_inference.py --help
```

Example API invocation:

```bash
python scripts/04_inference/run_inference.py \
  --requests path/to/requests.jsonl \
  --responses_dir runs/example \
  --backend openai_compatible \
  --provider zhizengzeng \
  --model gpt-4o \
  --temperature 0
```

Inference requires the dependencies used by the selected backend. API keys
must be supplied through environment variables or an untracked `.env` file.
No API call is required to reproduce the frozen results.

## Tasks

| Task | Diagnostic |
|---|---|
| A | Atomic next-state prediction |
| B | Structured scene perception |
| C | Trajectory outcome judgment under nuisance transformations |
| D / D-R | Paired Full/NoCue/CF outcome judgment; D-R includes a successful reference |
| E | Long-range discrete state tracking |

## Audit Notes

- [Repair contract](REBUTTAL_ARTIFACT_REPAIR.md)
- [Task D-R scoring audit](SCORING_AUDIT.md)
- [Frozen output notes](frozen_outputs/minigrid/taskd-r/README.md)

`SCORING_AUDIT.md` records a Kimi-K2.5 metric correction that requires author
review before updating paper tables. The raw responses are unchanged.

## Repository Scope

The repository also retains generators, inference utilities, MiniWorld
experiments, renderers, and general scoring modules. They are research code and
do not all form a single one-command pipeline. The commands above are the
strictly validated public reproduction path for this repair branch.

## License

MIT. See [LICENSE](LICENSE).
