#!/usr/bin/env python3
"""Generate mock responses for testing GridWM-Judge scoring pipeline."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate mock VLM responses")
    parser.add_argument("--exam-dir", type=str, required=True, help="Exam directory with exam JSONL")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--correct-rate", type=float, default=0.65, help="Correct answer rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    exam_dir = Path(args.exam_dir)
    exam_files = list(exam_dir.glob("*.jsonl"))
    if not exam_files:
        print(f"No exam JSONL found in {exam_dir}")
        return

    # Find all exam files
    exam_patterns = ["task_a_exam.jsonl", "task_b_exam.jsonl", "task_c_exam.jsonl",
                     "task_d_exam.jsonl", "task_e_exam.jsonl", "task_e_exam_v12.jsonl"]
    exam_files_to_process = []
    for pattern in exam_patterns:
        path = exam_dir / pattern
        if path.exists():
            exam_files_to_process.append(path)

    if not exam_files_to_process:
        exam_files_to_process = [f for f in exam_files if f.name.endswith("_exam.jsonl")]
    if not exam_files_to_process:
        exam_files_to_process = exam_files

    print(f"Found exam files: {[f.name for f in exam_files_to_process]}")
    responses = []

    for exam_file in exam_files_to_process:
        print(f"Reading exam from: {exam_file}")
        with open(exam_file) as f:
            for line in f:
                record = json.loads(line)
                uid = record.get("uid") or record.get("exam_id", "unknown")

                # Determine correct answer and task type
                task = record.get("task", "")
                if "answer" in record:
                    correct = str(record["answer"])
                elif "label" in record:
                    correct = "Success" if record.get("label") == 1 else "Fail"
                else:
                    correct = "Success"

                # Randomly decide if model gets it right
                if random.random() < args.correct_rate:
                    pred = correct
                else:
                    # Pick wrong answer based on task type
                    if task == "E":
                        # Task E: pick from label_space (excluding correct answer)
                        label_space = record.get("label_space", [])
                        if label_space:
                            wrong_options = [l for l in label_space if l != correct]
                            if wrong_options:
                                pred = random.choice(wrong_options)
                            else:
                                pred = correct
                        else:
                            pred = correct + "_wrong"
                    elif correct == "Success":
                        pred = "Fail"
                    elif correct == "Fail":
                        pred = "Success"
                    else:
                        # For multiple choice, pick different option
                        pred = correct + "_wrong"

                # Format response based on task type
                if task == "E":
                    # Task E requires JSON format: {"answer": "value"}
                    response_pred = json.dumps({"answer": pred})
                else:
                    response_pred = pred

                responses.append({
                    "uid": uid,
                    "pred": pred,
                    "raw": response_pred,
                    "meta": {"ok": True}
                })

    # Write responses
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for r in responses:
            f.write(json.dumps(r) + '\n')

    print(f"Generated {len(responses)} mock responses -> {out_path}")
    print(f"Expected accuracy: ~{args.correct_rate:.1%}")


if __name__ == "__main__":
    main()
