# Towards-Reliable-VLM-Judges

Evaluating Vision-Language Models (VLMs) on causal reasoning tasks in grid worlds (MiniGrid) and 3D environments (MiniWorld).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Lucas-Jin-Qh/Towards-Reliable-VLM-Judges.git
cd Towards-Reliable-VLM-Judges

# Install dependencies
pip install Pillow pandas numpy scipy python-dotenv

# Run full pipeline (generate exams -> VLM responses -> scoring)
./scripts/run_full_pipeline.sh
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

```
datasets/
├── minigrid/                    # MiniGrid evaluation data
│   ├── taska/task_a_exam.jsonl  # Task A (next state)
│   ├── taskb/task_b_exam.jsonl  # Task B (perception)
│   ├── taskc/task_c_exam.jsonl  # Task C (trajectory)
│   └── taskd/task_d_exam.jsonl  # Task D (generalization)
├── global_views/                # Raw trajectory data
│   ├── doorkey/
│   ├── keycorridor/
│   ├── memory/
│   ├── multiroom/
│   ├── redblue/
│   └── lavagap/
└── miniworld/                   # MiniWorld evaluation data
```

---

## Usage

### Full Pipeline (Recommended)

The easiest way to run evaluation:

```bash
# Full pipeline with trajectory generation
./scripts/run_full_pipeline.sh

# Skip trajectory generation (use existing data)
./scripts/run_full_pipeline.sh --skip-trajectories

# MiniGrid only or MiniWorld only
./scripts/run_full_pipeline.sh --minigrid-only
./scripts/run_full_pipeline.sh --miniworld-only

# Custom parameters
./scripts/run_full_pipeline.sh --seed 42 --correct-rate 0.65 --c-k 8
```

### Step-by-Step Evaluation

#### 1. Generate Exams

```bash
# Build exams from trajectories
python scripts/build/build_exam.py \
  --root datasets/global_views \
  --seed 42 \
  --c-k 8 \
  --out-dir outputs/exams

# Generate ablation variants
python scripts/build/build_exam.py --root datasets/global_views --seed 42 --c-k 4 --out-dir outputs/ablation_k4
python scripts/build/build_exam.py --root datasets/global_views --seed 42 --c-k 12 --out-dir outputs/ablation_k12

# Generate split format
python scripts/build/split_taskA_frames.py --exam_dir outputs/exams --out_dir outputs/split --num 50
```

#### 2. Run Inference

Core script: `scripts/infer/run_inference.py`

```bash
python scripts/infer/run_inference.py \
  --requests datasets/minigrid/taska/task_a_exam.jsonl \
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

#### 3. Score Results

```bash
# Score MiniGrid
python scripts/score/score_exam.py \
  --exam_dir outputs/exams \
  --responses runs/my_run/responses.jsonl \
  --out outputs/scores/score.json

# Compute IDR (Intervention Discrimination Rate)
python scripts/score/compute_idr.py \
  --per-row-csv outputs/scores/per_row.csv \
  --output-csv outputs/scores/idr_results.csv

# Compute LES (Leakage Effect Size)
python scripts/score/compute_les.py \
  --per-row-csv outputs/scores/per_row.csv \
  --output-csv outputs/scores/les_results.csv
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
Towards-Reliable-VLM-Judges/
├── datasets/
│   ├── minigrid/            # MiniGrid exam data
│   ├── miniworld/           # MiniWorld exam data
│   └── global_views/        # Raw trajectory data
├── scripts/
│   ├── run_full_pipeline.sh # One-click evaluation
│   ├── build/               # Exam generation
│   │   ├── build_exam.py
│   │   └── split_taskA_frames.py
│   ├── infer/              # VLM inference
│   │   ├── run_inference.py
│   │   └── run_canonical_pipeline.py
│   ├── score/              # Evaluation & metrics
│   │   ├── score_exam.py
│   │   ├── compute_idr.py
│   │   ├── compute_les.py
│   │   └── compute_vcc_oracle.py
│   └── schema/             # Shared utilities
└── outputs/                 # Generated exams & results
```

---

## Advanced Options

### Ablation Studies

```bash
# Test with different context sizes
./scripts/run_full_pipeline.sh --c-k 4   # Fewer frames
./scripts/run_full_pipeline.sh --c-k 12  # More frames
```

### Oracle Inference (for VCC scoring)

Oracle is a text-only baseline (no images) used to measure pixel-text consistency.

```bash
# Build oracle requests
python scripts/build/build_oracle_requests.py \
  --exam datasets/minigrid/taskc/task_c_exam.jsonl \
  --output runs/my_run/oracle/requests.jsonl

# Run oracle inference
python scripts/infer/run_oracle_inference.py \
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

## License

MIT License - see [LICENSE](LICENSE) for details.
# Towards-Reliable-VLM-Judges
# Towards-Reliable-VLM-Judges
# Towards-Reliable-VLM-Judges
# Towards-Reliable-VLM-Judges
