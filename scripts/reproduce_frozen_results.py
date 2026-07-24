#!/usr/bin/env python3
"""Re-score all frozen Task D-R responses and verify published results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    exam_dir = root / "datasets/minigrid/taskd-r"
    frozen = root / "frozen_outputs/minigrid/taskd-r"
    scorer = root / "scripts/05_scoring/score_exam.py"
    summarizer = root / "scripts/05_scoring/summarize_taskdr.py"

    subprocess.run(
        [sys.executable, str(root / "scripts/validate_release.py"), "--repo-root", str(root)],
        check=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    model_dirs = sorted(path for path in frozen.iterdir() if path.is_dir())
    for model_dir in model_dirs:
        target = output / model_dir.name
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable, str(scorer),
            "--exam_dir", str(exam_dir),
            "--responses", str(model_dir / "responses.jsonl"),
            "--out", str(target / "score_report.json"),
            "--per_row_out", str(target / "score_per_row.jsonl"),
        ], check=True)
        reproduced = read_jsonl(target / "score_per_row.jsonl")
        expected = read_jsonl(model_dir / "score_per_row.jsonl")
        if reproduced != expected:
            raise AssertionError(f"{model_dir.name}: reproduced per-row scores differ")

    metrics = output / "taskdr_metrics.csv"
    subprocess.run([
        sys.executable, str(summarizer),
        "--scores-root", str(output),
        "--output", str(metrics),
    ], check=True)

    expected_metrics = list(csv.DictReader((frozen / "taskdr_metrics.csv").open(encoding="utf-8")))
    actual_metrics = list(csv.DictReader(metrics.open(encoding="utf-8")))
    for row in expected_metrics:
        row["source"] = Path(row["source"]).name
    for row in actual_metrics:
        row["source"] = Path(row["source"]).name
    if actual_metrics != expected_metrics:
        raise AssertionError("reproduced Task D-R metrics differ from frozen metrics")
    print(f"PASS: reproduced {len(model_dirs)} models and metrics in {output}")


if __name__ == "__main__":
    main()
