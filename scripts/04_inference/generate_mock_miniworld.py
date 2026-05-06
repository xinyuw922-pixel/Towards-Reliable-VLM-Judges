#!/usr/bin/env python3
"""Generate mock responses for MiniWorld exams."""

import argparse
import json
import random
import re
from pathlib import Path


def generate_mock_responses(exam_path: Path, out_path: Path, correct_rate: float = 0.65, seed: int = 42):
    """Generate mock responses for a MiniWorld exam."""
    random.seed(seed)

    rows = []
    with open(exam_path) as f:
        for line in f:
            exam = json.loads(line)

            # Construct UID from available fields
            uid = exam.get("uid") or exam.get("exam_id", "unknown")
            if uid == "unknown":
                # Try to construct UID from template_id and variant
                template_id = exam.get("template_id", "")
                variant = exam.get("variant", "")
                family = exam.get("family", "")
                if template_id:
                    uid = f"{family}.{template_id}.{variant}" if variant else f"{family}.{template_id}"
                else:
                    # Fallback: use first 50 chars of prompt
                    uid = exam.get("prompt", "")[:50].replace("\n", " ")

            task = exam.get("task", "")
            # Detect task type from schema or uid prefix
            schema = exam.get("schema_version", "")
            uid_prefix = str(uid).split(".")[0] if uid else ""

            if task.startswith("A") or "A-MW" in schema or uid_prefix in ("A", "A-MW"):
                # Task A: answer is A/B/C/D
                ground_truth = str(exam.get("ground_truth", exam.get("correct_candidate", exam.get("answer", "A")))).upper()
                # Extract just the letter
                match = re.search(r'\b([A-D])\b', ground_truth)
                if match:
                    answer = match.group(1)
                else:
                    answer = ground_truth if ground_truth in ("A", "B", "C", "D") else "A"

                if random.random() < correct_rate:
                    pred = answer
                else:
                    # Pick a wrong answer
                    wrong_answers = [a for a in ("A", "B", "C", "D") if a != answer]
                    pred = random.choice(wrong_answers) if wrong_answers else answer

                rows.append({
                    "uid": uid,
                    "pred": pred,
                    "raw": pred,
                    "meta": {"ok": True, "model": "mock"}
                })
            elif task.startswith("C") or "C-MW" in task or "C-MW" in schema:
                # Task C: answer is "Success" or "Fail"
                answer = str(exam.get("answer", "Success"))
                pred = answer if random.random() < correct_rate else ("Fail" if answer == "Success" else "Success")
                rows.append({
                    "uid": uid,
                    "pred": pred,
                    "raw": pred,
                    "meta": {"ok": True, "model": "mock"}
                })
            elif task.startswith("D") or "D-MW" in task or task == "D-MW-RF" or "D-MW" in schema or "D-MW" in uid:
                # Task D: various formats
                prompt = str(exam.get("prompt", "")).lower()

                # Check if it uses ground_truth (A/B/C/D format) or answer (Success/Fail format)
                if "ground_truth" in exam or "correct_candidate" in exam:
                    # A/B/C/D format (like Task A)
                    ground_truth = str(exam.get("ground_truth", exam.get("correct_candidate", "A"))).upper()
                    match = re.search(r'\b([A-D])\b', ground_truth)
                    if match:
                        answer = match.group(1)
                    else:
                        answer = ground_truth if ground_truth in ("A", "B", "C", "D") else "A"

                    if random.random() < correct_rate:
                        pred = answer
                    else:
                        wrong_answers = [a for a in ("A", "B", "C", "D") if a != answer]
                        pred = random.choice(wrong_answers) if wrong_answers else answer

                    rows.append({
                        "uid": uid,
                        "pred": pred,
                        "raw": pred,
                        "meta": {"ok": True, "model": "mock"}
                    })
                elif "yes or no" in prompt:
                    # Prompt asks for Yes/No
                    answer = str(exam.get("answer", "Success"))
                    # Normalize answer to Yes/No
                    gold_yesno = "Yes" if answer == "Success" else "No"
                    pred_yesno = gold_yesno if random.random() < correct_rate else ("No" if gold_yesno == "Yes" else "Yes")
                    rows.append({
                        "uid": uid,
                        "pred": pred_yesno,
                        "raw": pred_yesno,
                        "meta": {"ok": True, "model": "mock"}
                    })
                else:
                    # Prompt asks for Success/Fail
                    answer = str(exam.get("answer", "Success"))
                    pred = answer if random.random() < correct_rate else ("Fail" if answer == "Success" else "Success")
                    rows.append({
                        "uid": uid,
                        "pred": pred,
                        "raw": pred,
                        "meta": {"ok": True, "model": "mock"}
                    })
            else:
                # Default: Success
                rows.append({
                    "uid": uid,
                    "pred": "Success",
                    "raw": "Success",
                    "meta": {"ok": True, "model": "mock"}
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    print(f"Generated {len(rows)} mock responses -> {out_path}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate mock responses for MiniWorld exams")
    parser.add_argument("--exam", type=str, required=True, help="Path to exam JSONL")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--correct-rate", type=float, default=1.0, help="Correct answer rate (1.0 = perfect)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_mock_responses(Path(args.exam), Path(args.out), args.correct_rate, args.seed)


if __name__ == "__main__":
    main()
