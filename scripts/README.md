# GridWM-Judge Scripts

## 目录结构

```
scripts/
├── README.md                    # 本文件
├── run_full_pipeline.sh        # 完整流水线
│
├── 01_generators/              # 试卷生成（主入口）
│   ├── minigrid/              # MiniGrid 环境
│   │   ├── build_exam.py              # A+B+C 统一入口
│   │   ├── generate_task_d_v3_144.py # D 生成
│   │   ├── build_exam_task_e_*.py    # E 生成
│   │   └── build_taskA_*.py          # A 探测请求
│   └── miniworld/              # MiniWorld 环境
│       ├── build_exam.py               # 统一入口
│       └── build_miniworld_tier1_paper_pack.py
│
├── 02_trajectories/            # 轨迹数据生成
│   ├── minigrid/               # MiniGrid 轨迹生成
│   │   ├── gen_action_ablation.py
│   │   ├── split_taskA_frames.py
│   │   └── tasks/              # 各环境生成器
│   │       ├── doorkey/
│   │       ├── keycorridor/
│   │       ├── lavagap/
│   │       ├── memory/
│   │       ├── multiroom/
│   │       └── redblue/
│   └── miniworld/              # MiniWorld 轨迹生成
│       ├── gen_taskA_MW_*.py
│       ├── gen_taskD_MW_*.py
│       └── build_task*_MW_requests.py
│
├── 03_postprocess/             # 后处理
│   ├── audit/                  # 审计变体生成
│   │   ├── apply_memory_hybrid_to_taskd.py
│   │   ├── rebuild_taskb_72_repaired.py
│   │   └── build_taskd_memory_full_cf_redesign.py
│   └── framing/                 # 语气变体
│       ├── freeze_taskC_MW_framing_smoke.py
│       └── freeze_taskD_MW_*.py
│
├── 04_inference/               # 推理
│   ├── run_canonical_pipeline.py
│   └── run_inference.py
│
├── 05_scoring/                 # 评分
│   ├── score_exam.py
│   ├── compute_idr.py
│   ├── compute_les.py
│   └── compute_*.py
│
├── 06_rendering/               # PDF/可视化渲染
│   ├── render_taskb_review_pdf.py
│   ├── render_taskdr_review_pdf.py
│   └── render_*.py
│
├── 07_utils/                   # 工具函数
│   ├── anonymize.py
│   ├── generate_oracle_text.py
│   └── render_*.py
│
├── 08_export/                  # 导出
│   └── export_demo.py
│
└── schema/                     # Schema 定义
    ├── exam_schema.py
    └── env_bootstrap.py
```

## Task 对应关系

| Task | 描述 | MiniGrid | MiniWorld |
|------|------|----------|-----------|
| A | 原子状态转移预测 | `build_exam.py` | `build_exam.py` |
| B | 结构化场景感知 | `build_exam.py` | - |
| C | 时序判断推理 | `build_exam.py` | `build_exam.py` |
| D | 轨迹结果分类 | `generate_task_d_v3_144.py` | - |
| E | 复杂推理 | `build_exam_task_e_*.py` | - |

## 使用方法

### 生成 MiniGrid 试卷
```bash
cd 01_generators/minigrid
python build_exam.py --root ../../outputs/raw_data --out-dir ../../outputs/exams
```

### 生成 Task D
```bash
cd 01_generators/minigrid
python generate_task_d_v3_144.py --raw-root ../../outputs/raw_data
```

### 生成 MiniWorld 试卷
```bash
cd 01_generators/miniworld
python build_exam.py --env MiniWorld-FourRooms-v0 --num-groups 10
```
