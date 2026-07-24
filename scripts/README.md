# Scripts

## Validated Rebuttal Entry Points

| Script | Purpose |
|---|---|
| `validate_release.py` | Validate Task D-R structure, images, hashes, UID joins, scores, and reports |
| `reproduce_frozen_results.py` | Re-score all frozen Task D-R responses and compare per-row and aggregate outputs |
| `05_scoring/score_exam.py` | General SSOT scorer |
| `05_scoring/summarize_taskdr.py` | Generate the frozen Task D-R metric table |
| `04_inference/run_inference.py` | Unified VLM inference gateway |

Run commands from the repository root:

```bash
python scripts/validate_release.py
python scripts/reproduce_frozen_results.py \
  --output-dir /tmp/gridwm-taskdr-reproduced
```

## Modules

```text
scripts/
├── 01_generators/     Exam builders
├── 02_trajectories/   MiniGrid trajectory generators and validators
├── 03_postprocess/    Audit and framing transformations
├── 04_inference/      Request builders and inference gateway
├── 05_scoring/        SSOT and task-specific scoring
├── 06_rendering/      Audit and figure renderers
├── 08_export/         Export utilities
├── env_bootstrap.py   Project environment loading
└── exam_schema.py     UID metadata and shared answer parsing
```

The historical one-command pipeline shell scripts are not part of the tracked
public artifact. Use only commands that resolve to files present in the current
checkout.
