#!/usr/bin/env python3
"""Canonical SSOT pipeline with unified CSV export as the default results layer."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, desc, cwd=None):
    """Execute command and print output"""
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"{'='*70}")
    # Make sure to run from the correct working directory
    script_dir = Path(__file__).parent
    if cwd is None:
        cwd = script_dir
    # Use sys.executable's directory in PATH so "python" resolves to the correct interpreter.
    # This avoids the bug where "python" resolves to a base miniconda python without pandas.
    python_dir = str(Path(sys.executable).parent)
    base_path = subprocess.os.environ.get("PATH", "")
    full_env = {
        **subprocess.os.environ,
        "PYTHONPATH": str(script_dir.parent),
        "PATH": f"{python_dir}:{base_path}",
    }
    result = subprocess.run(
        cmd, shell=True,
        cwd=str(cwd),
        env=full_env,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Canonical SSOT Pipeline")
    parser.add_argument("--responses", "-r", required=True, 
                        help="Path to responses JSONL file")
    parser.add_argument("--exam-dir", "-e", required=True,
                        help="Path to exam directory")
    parser.add_argument("--model", "-m", default=None,
                        help="Model name (auto-detect from responses if not provided)")
    parser.add_argument("--output-dir", "-o", default="./results",
                        help="Output directory for results")
    args = parser.parse_args()
    
    responses = Path(args.responses)
    exam_dir = Path(args.exam_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect model name
    if args.model:
        model = args.model
    else:
        # Infer from responses filename or content
        model = responses.parent.name.split("_")[-1] if "_" in responses.parent.name else "unknown"
        # Try to read from first record
        with open(responses) as f:
            first = json.loads(f.readline())
            model = first.get('meta', {}).get('model', model)
    print(f"\n>>> Model: {model}")
    
    # Step 1: Score Exam (JAccneu, AccA, AccB)
    print("\n>>> Step 1: Scoring exam...")
    score_report = output_dir / "score_report.json"
    cmd = f"python score_exam.py --exam_dir {exam_dir} --responses {responses} --out {score_report}"
    run_cmd(cmd, "Step 1: Score Exam")
    
    # Step 2: Export unified CSV layer
    print("\n>>> Step 2: Exporting unified outputs...")
    unified_dir = output_dir / "unified_outputs"
    cmd = f"python export_unified_outputs.py --exam_dir {exam_dir} --responses {responses} --out_dir {unified_dir}"
    run_cmd(cmd, "Step 2: Export Unified Outputs")

    print("\n>>> Step 3: Validating unified outputs...")
    cmd = f"python validate_unified_outputs.py --export-dir {unified_dir}"
    run_cmd(cmd, "Step 3: Validate Unified Outputs")

    per_row_csv = unified_dir / "per-row.csv"
    model_level_csv = unified_dir / "model-level.csv"

    # Step 4: Compute IDR
    print("\n>>> Step 4: Computing IDR...")
    idr_csv = output_dir / "idr.csv"
    cmd = f"python compute_idr.py --per-row-csv {per_row_csv} --output-csv {idr_csv}"
    run_cmd(cmd, "Step 4: Compute IDR")
    
    # Step 5: Compute LES
    print("\n>>> Step 5: Computing LES...")
    les_csv = output_dir / "les.csv"
    cmd = f"python compute_les.py --per-row-csv {per_row_csv} --output-csv {les_csv}"
    run_cmd(cmd, "Step 5: Compute LES")
    
    # Step 6: Export NS-IDR Quadrant
    print("\n>>> Step 6: Exporting NS-IDR quadrant...")
    quadrant_csv = output_dir / "ns_idr_quadrant.csv"
    cmd = f"python export_ns_idr_quadrant.py --per-row-csv {per_row_csv} --output-csv {quadrant_csv} --include-les"
    run_cmd(cmd, "Step 6: Export NS-IDR Quadrant")
    
    # Step 7: D7 Audit
    print("\n>>> Step 7: Running D7 audit...")
    d7_report = output_dir / "d7_audit_report.json"
    # Scan exam files (not responses)
    exam_c = exam_dir / "task_c_exam.jsonl"
    cmd = f"python d7_payload_scan.py --input {exam_c} --output {d7_report}"
    if run_cmd(cmd, "Step 7: D7 Audit"):
        # If successful, try to integrate into main pipeline
        pass
    
    # Step 8: Generate Main Table
    print("\n>>> Step 8: Generating main table...")
    main_table = output_dir / "main_table.csv"
    cmd = (
        f"python generate_main_table.py --model-level {model_level_csv} "
        f"--idr {idr_csv} --les {les_csv} --quadrant {quadrant_csv} "
        f"--model {model} --output {main_table}"
    )
    run_cmd(cmd, "Step 8: Generate Main Table")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    for f in output_dir.glob("*"):
        size = f.stat().st_size
        print(f"  - {f.name} ({size} bytes)")
    
    # Print main table preview
    print(f"\n{'='*70}")
    print(f"  MAIN TABLE PREVIEW")
    print(f"{'='*70}")
    if main_table.exists():
        import pandas as pd
        main_df = pd.read_csv(main_table)
        print(main_df.to_string())


if __name__ == "__main__":
    main()
