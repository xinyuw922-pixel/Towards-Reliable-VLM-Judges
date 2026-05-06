# Towards Reliable VLM Judges: State-Conditional Invariance and Presentation-Aware Diagnostics

Evaluating Vision-Language Models (VLMs) on causal reasoning tasks in grid worlds (MiniGrid) and 3D environments (MiniWorld).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/xinyuw922-pixel/Towards-Reliable-VLM-Judges.git
cd Towards-Reliable-VLM-Judges

# Install dependencies
pip install -r requirements_current.txt

# Run MiniGrid evaluation pipeline
./scripts/run_full_pipeline_minigrid.sh

# Run MiniWorld evaluation pipeline
./scripts/run_full_pipeline_miniworld.sh
```

---

## Overview

GridWM-Judge evaluates VLMs on causal reasoning by testing their ability to:

1. **Predict successor states** - Given an action and current state, predict the outcome
2. **Perceive environment state** - Understand spatial relationships and object properties
3. **Distinguish causal mechanisms** - Differentiate between intervention effects and nuisance variations
4. **Generalize across environments** - Transfer knowledge from training to novel settings

### Tasks

| Task | Description | Evaluates |
|------|-------------|-----------|
| **Task A** | Next state prediction | Causal reasoning: action -> consequence |
| **Task B** | Perception & spatial reasoning | Environment understanding |
| **Task C** | Trajectory outcome classification | Intervention discrimination |
| **Task D** | Cross-environment generalization | Transfer learning |
| **Task E** | Complex reasoning chains | Multi-step causal inference |

---

## Installation

### Dependencies

```bash
# Install all dependencies from requirements file
pip install -r requirements_current.txt

# Or minimal dependencies
pip install Pillow pandas numpy scipy python-dotenv
pip install statsmodels torch  # Optional
```

### API Configuration

Create a `.env` file in the repository root:

```bash
# Zhizengzeng API (supports GPT / Claude / Gemini / Qwen)
ZZZ_API_KEY=your_api_key
ZHIZENGZENG_API_KEY=your_api_key

# Alternative providers
GEMINI_API_KEY=your_gemini_key
MOONSHOT_API_KEY=your_moonshot_key
```

---

## Dataset Structure

### Pipeline Outputs (`outputs/`)

All generated data is stored in the `outputs/` directory. This directory is automatically created when running the pipeline scripts.

```
outputs/                        # Pipeline outputs (auto-generated)
├── raw_data/                   # Raw trajectory data
│   ├── doorkey/
│   ├── keycorridor/
│   ├── memory/
│   ├── multiroom/
│   ├── redblue/
│   └── lavagap/
├── exams_abc/                  # Task A/B/C/E exams + mock responses + scores
├── exams_taskd/                # Task D exam + scores
├── ablation_k4/               # Ablation: k=4 context frames
├── ablation_k12/              # Ablation: k=12 context frames
├── split/                     # Task A split/composite experiments
├── exams_taskc_split/         # Task C split/composite
├── ablation/                  # Action ablation (shuffle/mask)
└── miniworld/                  # MiniWorld pipeline outputs
    ├── taskA-MW-v2-prototype/
    ├── taskA-MW-v2-crossenv/
    ├── taskC-MW-family-prototypes/
    ├── taskC-MW-formal-bank/
    ├── taskD-MW-fourrooms/
    ├── taskD-MW-storyboard-prototype/
    ├── taskD-MW-highdisc-round1/
    ├── taskD-MW-highdisc-round2/
    ├── taskD-MW-highdisc-round3/
    └── taskD-MW-migration-showcase/
```
---

## Usage

### Full Pipeline (Recommended)

The easiest way to run evaluation:

#### MiniGrid Pipeline

```bash
# Full pipeline (generate trajectories -> exams -> VLM simulation -> scoring)
./scripts/run_full_pipeline_minigrid.sh

# Skip trajectory generation (use existing data)
./scripts/run_full_pipeline_minigrid.sh --skip-trajectories

# Skip sampling/exams generation (VLM simulation only)
./scripts/run_full_pipeline_minigrid.sh --skip-sampling --skip-exams

# Custom parameters
./scripts/run_full_pipeline_minigrid.sh --seed 42 --correct-rate 0.65 --c-k 8

# Ablation studies only
./scripts/run_full_pipeline_minigrid.sh --skip-trajectories --skip-sampling --skip-exams --skip-split --skip-vlm
```

#### MiniWorld Pipeline

```bash
# Full pipeline (generate tasks -> requests -> VLM simulation -> scoring)
./scripts/run_full_pipeline_miniworld.sh

# Skip specific tasks
./scripts/run_full_pipeline_miniworld.sh --skip-task-c --skip-task-d

# Custom parameters
./scripts/run_full_pipeline_miniworld.sh --seed 42 --correct-rate 0.65 --obs-width 640 --obs-height 480

# Skip VLM simulation (exam generation only)
./scripts/run_full_pipeline_miniworld.sh --skip-vlm
```

### Step-by-Step Evaluation (MiniGrid)

#### 1. Generate Trajectories

```bash
# Generate raw trajectory data
python scripts/02_trajectories/minigrid/tasks/doorkey/gen_doorkey_triplets.py --out-dir outputs/raw_data/doorkey --num 10
python scripts/02_trajectories/minigrid/tasks/keycorridor/gen_keycorridor_triplets.py --out-dir outputs/raw_data/keycorridor --num 10
# ... other tasks
```

#### 2. Sample Data

```bash
python scripts/sample_raw_data.py --src outputs/raw_data --dst scripts/outputs/raw_data --seed 42
```

#### 3. Build Exams

```bash
# Build Task A/B/C exams
python scripts/01_generators/minigrid/build_exam.py --root scripts/outputs/raw_data --seed 42 --c-k 8 --out-dir outputs/exams_abc

# Build Task D exam
python scripts/01_generators/minigrid/generate_task_d_v3_144.py --raw-root scripts/outputs/raw_data --out-dir outputs/exams_taskd --groups-per-env 8

# Ablation variants
python scripts/01_generators/minigrid/build_exam.py --root scripts/outputs/raw_data --seed 42 --c-k 4 --out-dir outputs/ablation_k4
python scripts/01_generators/minigrid/build_exam.py --root scripts/outputs/raw_data --seed 42 --c-k 12 --out-dir outputs/ablation_k12
```

#### 4. Run Inference

Core script: `scripts/04_inference/run_inference.py`

```bash
python scripts/04_inference/run_inference.py \
  --requests outputs/exams_abc/task_a_exam.jsonl \
  --responses_dir runs/my_run/taskA \
  --model gpt-4o \
  --backend openai_compatible \
  --provider zhizengzeng \
  --max_new_tokens 256 \
  --temperature 0.0
```

### Model Configuration

| Model Family | `--backend` | `--provider` |
|--------------|-------------|--------------|
| GPT (gpt-4o, gpt-5) | `openai_compatible` | `zhizengzeng` |
| Claude | `openai_compatible` | `zhizengzeng` |
| Gemini | `openai_compatible` | `zhizengzeng` or `gemini` |
| Qwen VL | `openai_compatible` | `zhizengzeng` |
| Kimi | `openai_compatible` | `moonshot` |

#### 5. Score Results

```bash
# Score Task A/B/C/E exams
python scripts/05_scoring/score_exam.py \
  --exam_dir outputs/exams_abc \
  --responses outputs/exams_abc/responses_mock.jsonl \
  --out outputs/exams_abc/score_mock.json

# Score Task D
python scripts/05_scoring/score_exam.py \
  --exam_dir outputs/exams_taskd \
  --responses outputs/exams_abc/responses_mock.jsonl \
  --out outputs/exams_taskd/score_mock.json

# Compute IDR (Intervention Discrimination Rate)
python scripts/05_scoring/compute_idr.py \
  --per-row-csv outputs/exams_abc/per_row.csv \
  --output-csv outputs/exams_abc/idr_results.csv

# Compute LES (Leakage Effect Size)
python scripts/05_scoring/compute_les.py \
  --per-row-csv outputs/exams_abc/per_row.csv \
  --output-csv outputs/exams_abc/les_results.csv
```

### Step-by-Step Evaluation (MiniWorld)

#### 1. Generate Tasks

```bash
# Task A: next-state prediction
python scripts/02_trajectories/miniworld/gen_taskA_MW_v2_prototype.py \
  --out_dir outputs/miniworld/taskA-MW-v2-prototype \
  --obs_width 320 --obs_height 240

# Task C: trajectory classification
python scripts/02_trajectories/miniworld/gen_taskC_MW_formal_exam.py \
  --out_dir outputs/miniworld/taskC-MW-formal-bank \
  --obs_width 320 --obs_height 240 --auto-deps

# Task D: cross-environment generalization
python scripts/02_trajectories/miniworld/gen_taskD_MW_fourrooms.py \
  --out_dir outputs/miniworld/taskD-MW-fourrooms \
  --obs_width 640 --obs_height 480 --num_questions 15
```

#### 2. Build API Requests

```bash
python scripts/04_inference/build_taskA_MW_v2_requests.py \
  --exam_jsonl outputs/miniworld/taskA-MW-v2-prototype/taskA-MW-v2-prototype.jsonl \
  --out outputs/miniworld/taskA-MW-v2-prototype/requests.jsonl

python scripts/04_inference/build_taskC_MW_formal_exam_requests.py \
  --exam_jsonl outputs/miniworld/taskC-MW-formal-bank/task_c_exam.jsonl \
  --out outputs/miniworld/taskC-MW-formal-bank/requests.jsonl
```

#### 3. Run Inference

```bash
python scripts/04_inference/run_inference.py \
  --requests outputs/miniworld/taskA-MW-v2-prototype/requests.jsonl \
  --responses_dir runs/miniworld/taskA \
  --model gpt-4o
```

#### 4. Score Results

```bash
python scripts/05_scoring/score_taskA_MW_v2_inference.py \
  --exam_jsonl outputs/miniworld/taskA-MW-v2-prototype/taskA-MW-v2-prototype.jsonl \
  --responses outputs/miniworld/taskA-MW-v2-prototype/responses.jsonl \
  --out outputs/miniworld/taskA-MW-v2-prototype/score_report.json

python scripts/05_scoring/score_taskC_MW_inference.py \
  --exam_jsonl outputs/miniworld/taskC-MW-formal-bank/task_c_exam.jsonl \
  --responses outputs/miniworld/taskC-MW-formal-bank/responses.jsonl \
  --out outputs/miniworld/taskC-MW-formal-bank/score_summary.json
```

---

## Metrics

| Metric | Description | Tasks |
|--------|-------------|-------|
| `Acc` | Accuracy (prediction = ground truth) | All |
| `IDR` | Intervention Discrimination Rate | C, D |
| `LES` | Leakage Effect Size | C |
| `JAccneu` | Neutral framing accuracy | C |
| `VCC` | Vision-Consistency Consistency | C |
| `Acc_A1` | Task A (successor state) accuracy | A |

---

## Project Structure

```
GridWM-Judge/
├── outputs/                  # Pipeline outputs (auto-generated)
│   ├── raw_data/            # Raw trajectory data
│   ├── exams_abc/           # Task A/B/C/E exams + mock responses + scores
│   ├── exams_taskd/         # Task D exam + scores
│   ├── ablation_k4/        # Ablation: k=4 context frames
│   ├── ablation_k12/        # Ablation: k=12 context frames
│   ├── split/              # Task A split/composite experiments
│   ├── exams_taskc_split/  # Task C split/composite
│   ├── ablation/           # Action ablation (shuffle/mask)
│   └── miniworld/          # MiniWorld pipeline outputs
├── scripts/
│   ├── run_full_pipeline_minigrid.sh    # MiniGrid one-click evaluation
│   ├── run_full_pipeline_miniworld.sh   # MiniWorld one-click evaluation
│   ├── run_full_pipeline.sh             # Legacy combined pipeline
│   ├── sample_raw_data.py               # Data sampling utility
│   ├── exam_schema.py                   # Exam schema definitions
│   ├── 01_generators/                  # Exam generation
│   │   └── minigrid/
│   │       ├── build_exam.py
│   │       ├── build_taskA_split_format_probe_requests.py
│   │       ├── build_taskce_api_smoke_requests.py
│   │       ├── build_exam_task_e_v12.py
│   │       ├── build_exam_task_e_v13_*.py
│   │       └── generate_task_d_v3_144.py
│   ├── 02_trajectories/                # Trajectory generation
│   │   ├── minigrid/
│   │   │   ├── config.py
│   │   │   ├── gen_action_ablation.py
│   │   │   ├── split_taskA_frames.py
│   │   │   └── tasks/
│   │   │       ├── doorkey/
│   │   │       ├── keycorridor/
│   │   │       ├── lavagap/
│   │   │       ├── memory/
│   │   │       ├── multiroom/
│   │   │       └── redblue/
│   │   └── miniworld/                  # MiniWorld trajectory generators
│   ├── 03_postprocess/                 # Post-processing
│   │   ├── audit/                       # Audit-related scripts
│   │   └── framing/                     # Framing transformation
│   ├── 04_inference/                    # VLM inference
│   │   ├── run_inference.py
│   │   ├── run_canonical_pipeline.py
│   │   ├── generate_mock_responses.py
│   │   ├── generate_mock_miniworld.py
│   │   ├── build_taskA_MW_v2_requests.py
│   │   ├── build_taskC_MW_formal_exam_requests.py
│   │   └── build_taskD_MW_*.py
│   ├── 05_scoring/                      # Evaluation & metrics
│   │   ├── score_exam.py
│   │   ├── score_miniworld_canonical.py
│   │   ├── score_taskA_MW_v2_inference.py
│   │   ├── score_taskC_MW_inference.py
│   │   └── score_taskD_MW_*.py
│   ├── 06_rendering/                   # PDF/figure rendering
│   │   ├── render_taskA_MW_*.py
│   │   ├── render_taskC_MW_*.py
│   │   ├── render_taskD_MW_*.py
│   │   ├── render_taskb_review_pdf.py
│   │   └── render_taskdr_review_pdf.py
│   └── 08_export/                      # Results export
├── datasets/                            # Static datasets (empty placeholder)
├── config.py                           # Configuration
├── requirements_current.txt            # Full dependencies
├── requirements_miniworld.txt          # MiniWorld dependencies
└── LICENSE
```

---

## Advanced Options

### Ablation Studies (MiniGrid)

```bash
# Test with different context sizes
./scripts/run_full_pipeline_minigrid.sh --c-k 4   # Fewer frames
./scripts/run_full_pipeline_minigrid.sh --c-k 12  # More frames

# Action ablation (shuffle/mask actions)
python scripts/02_trajectories/minigrid/gen_action_ablation.py \
  --exam outputs/exams_abc/task_a_exam.jsonl \
  --output-dir outputs/ablation
```

### VLM Provider Configuration

Edit `config.py` or set environment variables:

```bash
# Use specific provider
export VLM_PROVIDER=zhizengzeng  # or "gemini", "moonshot"
export VLM_MODEL=gpt-4o

# For MiniWorld with headless rendering
./scripts/run_full_pipeline_miniworld.sh --obs-width 640 --obs-height 480
```

### Oracle Inference (for VCC scoring)

Oracle is a text-only baseline (no images) used to measure pixel-text consistency.

```bash
# Build oracle requests
python scripts/build/build_oracle_requests.py \
  --exam outputs/exams_abc/task_c_exam.jsonl \
  --output runs/my_run/oracle/requests.jsonl

# Run oracle inference
python scripts/04_inference/run_oracle_inference.py \
  --exam runs/my_run/oracle/requests.jsonl \
  --output-dir runs/my_run/oracle
```

---

## FAQ

**Q: API quota exceeded?**
A: Rerun the same command - `run_inference.py` supports resume and skips completed rows.

**Q: How to speed up inference?**
A: Use `--workers N` for parallel requests (e.g., `--workers 8`).

**Q: Scoring error "no LES-eligible rows"?**
A: Verify the response JSONL contains temporal/visual probe data and `--per-row-csv` path is correct.

**Q: How are pixel and oracle responses aligned?**
A: Both scripts auto-align via UID (`exam_id` field). Ensure both response files exist.

---