#!/usr/bin/env python3
"""Simple scorer for MiniWorld canonical exams with mock responses."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_verdict(text):
    """Normalize verdict text to Success/Fail."""
    text = str(text).lower().strip()
    if text in ('success', 'yes', 'true', 'succeed'):
        return 'Success'
    if text in ('fail', 'no', 'false', 'failed', 'failure'):
        return 'Fail'
    return text


def detect_task_type(exam):
    """Detect task type from exam record."""
    task = exam.get("task", "")
    schema = exam.get("schema_version", "")
    uid = exam.get("uid", exam.get("exam_id", ""))

    # Method 1: Check explicit task field
    if task in ("A", "A-MW", "B", "B-MW"):
        return "A"  # Both A and B use A/B/C/D answer format
    if task in ("C", "C-MW", "C-MW-formal"):
        return "C"
    if task.startswith("D") or "D-MW" in task or task == "D-MW-RF":
        return "D"

    # Method 2: Fallback - check schema version for MW exams
    if "mw_exam" in schema:
        if uid.startswith("A"):
            return "A"
        if uid.startswith("C"):
            return "C"
        if uid.startswith("D"):
            return "D"

    # Method 3: Fallback - check answer format
    answer = exam.get("answer", "")
    if isinstance(answer, str):
        answer_upper = answer.upper()
        if answer_upper in ("A", "B", "C", "D"):
            return "A"  # A/B/C/D format → Task A
        if answer_upper in ("SUCCESS", "FAIL", "YES", "NO"):
            return "C"  # Success/Fail format → Task C

    # Default: assume Task C
    return "C"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    exams = load_jsonl(args.exam)
    responses = load_jsonl(args.responses)

    resp_map = {}
    for r in responses:
        uid = r.get("uid") or r.get("exam_id", "unknown")
        resp_map[uid] = r

    # Detect task type from first exam record
    task_type = detect_task_type(exams[0]) if exams else "C"
    print(f"Detected task type: {task_type}")

    correct = 0
    total = 0
    parse_fail = 0
    missing = 0

    per_row = []

    for exam in exams:
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

        resp = resp_map.get(uid)

        row = {"uid": uid}

        if resp is None:
            missing += 1
            row["correct"] = False
            row["pred"] = None
            row["gold"] = None
            per_row.append(row)
            continue

        total += 1
        pred = resp.get("pred", resp.get("raw", ""))
        row["pred_raw"] = pred

        if task_type == "A":
            # Task A: answer is A/B/C/D (or ground_truth/correct_candidate)
            gold = str(exam.get("ground_truth", exam.get("correct_candidate", exam.get("answer", "A")))).upper()
            row["gold"] = gold
            pred_norm = str(pred).upper().strip()
            # Extract just A/B/C/D if model returns full text
            match = re.search(r'\b([A-D])\b', pred_norm)
            if match:
                pred_norm = match.group(1)
            if pred_norm in ("A", "B", "C", "D"):
                row["pred"] = pred_norm
                row["correct"] = (pred_norm == gold)
            else:
                row["pred"] = None
                row["correct"] = False
                parse_fail += 1

        elif task_type in ("C", "D"):
            # Check if exam uses ground_truth (A/B format) or answer (Success/Fail format)
            if "ground_truth" in exam or "correct_candidate" in exam:
                # A/B/C/D format (like Task A)
                gold = str(exam.get("ground_truth", exam.get("correct_candidate", "A"))).upper()
                row["gold"] = gold
                pred_norm = str(pred).upper().strip()
                match = re.search(r'\b([A-D])\b', pred_norm)
                if match:
                    pred_norm = match.group(1)
                if pred_norm in ("A", "B", "C", "D"):
                    row["pred"] = pred_norm
                    row["correct"] = (pred_norm == gold)
                else:
                    row["pred"] = None
                    row["correct"] = False
                    parse_fail += 1
            else:
                # Success/Fail or Yes/No format
                gold = str(exam.get("answer", "Success"))
                row["gold"] = gold
                # Check prompt format
                prompt = str(exam.get("prompt", "")).lower()
                if "yes or no" in prompt:
                    # Normalize Yes/No to match gold
                    gold_yesno = "Yes" if gold == "Success" else "No"
                    row["gold"] = gold_yesno  # Use Yes/No for comparison
                pred_norm = normalize_verdict(pred)
                if "yes or no" in prompt:
                    # Convert normalized verdict to Yes/No
                    pred_norm = "Yes" if pred_norm == "Success" else "No"
                row["pred"] = pred_norm
                row["correct"] = (pred_norm == row["gold"])

        else:
            gold = str(exam.get("answer", "Success"))
            row["gold"] = gold
            pred_norm = normalize_verdict(pred)
            row["pred"] = pred_norm
            row["correct"] = (pred_norm == gold)

        if row["correct"]:
            correct += 1

        per_row.append(row)

    acc = correct / total if total > 0 else 0

    report = {
        "task": task_type,
        "total_exams": len(exams),
        "scored": total,
        "missing": missing,
        "correct": correct,
        "accuracy": acc,
        "parse_failures": parse_fail,
    }

    print("=" * 60)
    print(f"Scoring MiniWorld {task_type}")
    print("=" * 60)
    print(f"  Total exams:   {len(exams)}")
    print(f"  Scored:        {total}")
    print(f"  Missing:       {missing}")
    print(f"  Correct:       {correct}")
    print(f"  Accuracy:      {acc:.1%}")
    print("=" * 60)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved report to: {args.out}")

    return report


if __name__ == "__main__":
    main()
