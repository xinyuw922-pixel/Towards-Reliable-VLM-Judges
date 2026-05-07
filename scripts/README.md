# GridWM-Judge Scripts

This directory contains the complete pipeline scripts for the GridWM-Judge benchmark, organized into the following modules.

---

## Directory Structure

```
scripts/
├── README.md                        # This file
├── exam_schema.py                   # Unified schema definition (shared by Task A/B/C/D/E)
├── config.py                        # Repository-level configuration
├── run_full_pipeline.sh              # Unified pipeline entry point (MiniGrid + MiniWorld)
├── run_full_pipeline_minigrid.sh     # MiniGrid full pipeline
├── run_full_pipeline_miniworld.sh   # MiniWorld full pipeline
├── run_with_headless_env.sh         # Headless environment helper script
├── sample_raw_data.py               # Raw data sampling utility
│
├── 01_generators/                   # Exam question generators
│   ├── minigrid/                   # MiniGrid environment
│   │   ├── build_exam.py           # [Main Entry] Task A/B/C unified generation
│   │   ├── build_exam_task_e_*.py  # Task E generation (complex reasoning)
│   │   ├── build_taskA_*.py        # Task A probe request construction
│   │   └── generate_task_d_*.py    # Task D generation
│   └── miniworld/                  # MiniWorld environment
│       └── build_miniworld_tier1_paper_pack.py  # Tier1 bundled generation
│
├── 02_trajectories/                # Trajectory data generation
│   ├── minigrid/                   # MiniGrid trajectory generation
│   │   ├── gen_action_ablation.py  # Action ablation data
│   │   ├── split_taskA_frames.py   # Task A frame splitting
│   │   ├── config.py               # MiniGrid trajectory configuration
│   │   └── tasks/                 # Per-environment triplet generators (Full/NoCue/CF)
│   │       ├── doorkey/           # DoorKey task
│   │       ├── keycorridor/       # KeyCorridor task
│   │       ├── lavagap/           # LavaGap task
│   │       ├── memory/            # Memory task
│   │       ├── multiroom/         # MultiRoom task
│   │       └── redblue/           # RedBlueDoors task
│   └── miniworld/                 # MiniWorld trajectory generation
│       ├── build_miniworld_exam.py # MiniWorld exam construction
│       ├── gen_task*_MW_*.py      # Per-Task MiniWorld version generation
│       ├── build_task*_MW_requests.py  # MiniWorld request construction
│       └── common.py              # MiniWorld common utilities
│
├── 03_postprocess/                 # Post-processing
│   ├── audit/                      # Audit variant generation (NoCue/CF enhancement)
│   │   ├── rebuild_taskb_72_repaired.py   # Task B repair
│   │   ├── build_taskd_memory_full_cf_redesign.py  # Task D CF variant
│   │   └── apply_memory_hybrid_to_taskd.py # Task D hybrid audit
│   └── framing/                   # Framing variants (tone/wording variations)
│       ├── freeze_taskC_MW_framing_smoke.py  # Task C Framing
│       └── freeze_taskD_MW_*.py             # Task D Framing
│
├── 04_inference/                  # VLM inference engine
│   ├── run_inference.py           # [Main Entry] Unified inference gateway
│   ├── run_canonical_pipeline.py  # Canonical pipeline
│   ├── generate_mock_*.py          # Mock response generation
│   └── build_*_requests.py        # Per-Task request construction
│
├── 05_scoring/                    # Scoring engine
│   ├── score_exam.py              # [Main Entry] SSOT scorer
│   ├── score_miniworld_canonical.py  # MiniWorld canonical scoring
│   └── score_task*_MW_*.py        # Per-Task specialized scoring
│
├── 06_rendering/                  # PDF/visual rendering
│   ├── render_taskA_*.py         # Task A visualization
│   ├── render_taskB_*.py         # Task B visualization
│   ├── render_taskC_*.py         # Task C visualization
│   ├── render_taskD_*.py         # Task D visualization
│   └── render_miniworld_*.py      # MiniWorld visualization
│
├── 08_export/                     # Export utilities
│   └── export_demo.py            # Dataset export
│
└── schema/                       # Schema definitions
    ├── exam_schema.py            # Exam schema (UID parsing, verdict extraction)
    └── env_bootstrap.py          # Environment variable loading
```

---

## Module Details

### 01_generators - Exam Question Generators

**Purpose**: Transform raw trajectory data into standardized VLM evaluation questions.

**Core Scripts**:
| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `build_exam.py` (MiniGrid) | Task A/B/C unified generation entry | Raw trajectories | `.jsonl` exam files + images |
| `build_exam_task_e_*.py` | Task E generation | Trajectories | Complex reasoning questions |
| `generate_task_d_*.py` | Task D generation | Trajectories | Trajectory classification questions |

**Task Mapping**:

| Task | Description | Core Capability | MiniGrid | MiniWorld |
|------|-------------|----------------|----------|-----------|
| A | Atomic State Transition Prediction | Physical dynamics understanding | `build_exam.py` | `build_*_MW_*.py` |
| B | Structured Scene Perception | Visual parsing, spatial reasoning | `build_exam.py` | - |
| C | Temporal Judgment Reasoning | Sequence understanding, counterfactual reasoning | `build_exam.py` | `build_*_MW_*.py` |
| D | Trajectory Outcome Classification | Trajectory interpretation | `generate_task_d_*.py` | `gen_taskD_MW_*.py` |
| E | Complex Reasoning | Multi-step planning | `build_exam_task_e_*.py` | - |

---

### 02_trajectories - Trajectory Data Generation

**Purpose**: Generate deterministic trajectories in MiniGrid/MiniWorld environments, including Full/NoCue/CF triplet variants.

**Core Concepts**:
- **Full**: Complete trajectory where the agent successfully completes the task
- **NoCue**: Key visual cues removed (e.g., key), testing reasoning under information absence
- **CF (Counterfactual)**: Counterfactual intervention by moving the goal position, testing whether the agent relies on surface features

**Core Scripts**:
| Script | Purpose |
|--------|---------|
| `gen_*_triplets.py` | Generate Full/NoCue/CF triplets |
| `validate_*.py` | Validate triplet consistency |
| `gen_action_ablation.py` | Action space ablation data |

**Planning Algorithm**: BFS (Breadth-First Search) computes shortest paths, ensuring trajectory optimality.

---

### 03_postprocess - Post-processing

**Purpose**: Augment and variant-generate the produced exam data.

**Sub-modules**:
- **audit/**: Audit variant generation
  - Build NoCue/CF combination variants
  - Repair problematic exam items
- **framing/**: Tone/wording variants
  - Generate questions with different phrasing (e.g., "succeed" vs "success")
  - Test model's phrasing robustness

---

### 04_inference - VLM Inference Engine

**Purpose**: Unify calls to different VLM backends for inference.

**Supported Backends**:
| Backend | Type | Examples |
|---------|------|---------|
| `openai_compatible` | API | OpenAI, SiliconFlow, Zhizengzeng |
| `gemini` | API | Google Gemini |
| `local` | Local | Qwen2.5-VL, InternVL2.5, LLaVA |

**Core Parameters** (Greedy Decoding):
```python
temperature = 0.0      # Greedy decoding
max_tokens_A = 512    # Task A
max_tokens_B = 2048   # Task B (JSON output)
max_tokens_C = 256    # Task C
```

**Core Scripts**:
| Script | Purpose |
|--------|---------|
| `run_inference.py` | [Main Entry] Unified inference gateway |
| `run_canonical_pipeline.py` | Canonical inference pipeline |

---

### 05_scoring - Scoring Engine

**Purpose**: Score VLM responses against ground truth.

**Design Principles**:
- **SSOT (Single Source of Truth)**: Only rely on ground truth in the exam directory
- **Capability-Fidelity**: Distinguish format noise from real capability
- **Strict vs Recoverable**: Report both strict and recoverable scores

**Scoring Rules**:

| Task | Scoring Method | Parsing Rules |
|------|----------------|---------------|
| Task A | Multiple choice | Extract first A/B/C/D |
| Task B | JSON Exact-Match | Strict field matching |
| Task C/D | Binary classification | Regex extract Yes/No, Success/Fail |
| Task E | JSON Exact-Match | Complex structure matching |

**Core Scripts**:
| Script | Purpose |
|--------|---------|
| `score_exam.py` | [Main Entry] SSOT scorer |
| `score_miniworld_canonical.py` | MiniWorld scoring |

---

### 06_rendering - PDF Visualization

**Purpose**: Render exam data to PDF for human review and visualization analysis.

**Output Formats**:
- Single-question PDF: One question per page
- Summary PDF: Multiple questions merged for quick review
- Showcase PDF: High-quality figures for papers

---

### 08_export - Data Export

**Purpose**: Export processed data to standard formats for sharing and reproducibility.

---

## Usage

### Full Pipeline

```bash
# MiniGrid full pipeline
./scripts/run_full_pipeline_minigrid.sh --seed 42 --num-trajs 50

# MiniWorld full pipeline
./scripts/run_full_pipeline_miniworld.sh --seed 42 --num-trajs 50

# Unified pipeline (supports both)
./scripts/run_full_pipeline.sh --seed 42
```

### Step-by-Step Execution

```bash
# Step 1: Generate trajectories
cd scripts/02_trajectories/minigrid
python -m tasks.doorkey.gen_doorkey_triplets --num 50 --out-dir ../../../outputs/raw_data

# Step 2: Generate exams
cd scripts/01_generators/minigrid
python build_exam.py --root ../../outputs/raw_data --out-dir ../../outputs/exams

# Step 3: VLM inference
cd scripts/04_inference
python run_inference.py --model gpt-4o --backend openai_compatible --provider zhizengzeng

# Step 4: Scoring
cd scripts/05_scoring
python score_exam.py --responses outputs/runs/responses/
```

---

## Requirements

- Python 3.10+
- PyTorch (CUDA 12.8)
- MiniGrid 3.0.0
- See `requirements_current.txt` for dependencies
