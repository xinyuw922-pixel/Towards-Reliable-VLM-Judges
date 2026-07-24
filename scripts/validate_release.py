#!/usr/bin/env python3
"""Strict validation for the published rebuttal artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

MODELS = {
    "claude-sonnet-4-6", "gemini-2.0-flash", "gemini-2.5-flash-lite",
    "gemini-3-flash-preview", "gpt-4o", "gpt-5.4", "kimi-k2.5",
    "qwen3-235b-a22b", "qwen3-32b", "qwen3-8b",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def uid(row: dict[str, Any]) -> str | None:
    return row.get("uid") or row.get("exam_id")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_hashes(root: Path, manifest: Path) -> None:
    if not manifest.exists():
        return
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        require(path.is_file(), f"hash manifest line {number}: missing {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"hash mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    exam_dir = root / "datasets/minigrid/taskd-r"
    output_root = root / "frozen_outputs/minigrid/taskd-r"
    exam_path = exam_dir / "task_d_exam.jsonl"
    alias_path = exam_dir / "task_d_r_exam.jsonl"

    exam = read_jsonl(exam_path)
    exam_uids = {uid(row) for row in exam}
    require(len(exam) == len(exam_uids) == 144, "Task D-R exam must have 144 unique UIDs")
    require(all(value and value.startswith("DR.") for value in exam_uids), "exam namespace must be DR.*")
    require(exam_path.read_bytes() == alias_path.read_bytes(), "Task D-R exam alias differs")
    require(Counter(row["env_task"] for row in exam) == Counter({
        "doorkey": 24, "keycorridor": 24, "lavagap": 24,
        "memory": 24, "multiroom": 24, "redblue": 24,
    }), "environment distribution mismatch")
    require(Counter(row["variant"] for row in exam) == Counter({
        "full": 48, "nocue": 48, "cf": 48,
    }), "variant distribution mismatch")
    require(Counter(row["answer"] for row in exam) == Counter({
        "Success": 96, "Fail": 48,
    }), "gold distribution mismatch")
    require(len({row["group_id"] for row in exam}) == 12, "expected 12 query groups")
    for row in exam:
        require((exam_dir / row["image"]).is_file(), f"missing image for {row['uid']}")

    model_dirs = {path.name for path in output_root.iterdir() if path.is_dir()}
    require(model_dirs == MODELS, f"model set mismatch: {sorted(model_dirs ^ MODELS)}")
    for model in sorted(MODELS):
        model_dir = output_root / model
        responses = read_jsonl(model_dir / "responses.jsonl")
        scores = read_jsonl(model_dir / "score_per_row.jsonl")
        report = json.loads((model_dir / "score_report.json").read_text(encoding="utf-8"))
        response_uids = {uid(row) for row in responses}
        score_uids = {uid(row) for row in scores}
        require(len(responses) == len(response_uids) == 144, f"{model}: response count/duplicates")
        require(len(scores) == len(score_uids) == 144, f"{model}: score count/duplicates")
        require(response_uids == exam_uids == score_uids, f"{model}: UID set mismatch")
        require(all(value and value.startswith("DR.") for value in response_uids), f"{model}: wrong namespace")
        require(all(row.get("failure") not in {"missing_response", "api_error"} for row in scores),
                f"{model}: missing response or API error in scores")
        by_uid = {row["uid"]: row for row in exam}
        for row in scores:
            gold = by_uid[row["uid"]]
            for field in ("task", "env_task", "group_id", "variant", "temporal", "visual", "framing"):
                require(row.get(field) == gold.get(field), f"{model}/{row['uid']}: {field} mismatch")
        coverage = report["coverage"]
        require(coverage["n_gold"] == coverage["n_response_uids"] == coverage["n_scored"] == 144,
                f"{model}: report coverage count mismatch")
        require(coverage["missing_responses"] == 0, f"{model}: report has missing responses")
        require(coverage["extra_responses_not_in_gold"] == 0, f"{model}: report has extra responses")
        score_acc = sum(row.get("correct") is True for row in scores) / 144
        require(abs(score_acc - report["overall_micro"]["acc"]) < 1e-12,
                f"{model}: report accuracy mismatch")

    validate_hashes(root, output_root / "SHA256SUMS")
    print("PASS: Task D-R exam, images, 10 response bundles, scores, reports, and hashes")


if __name__ == "__main__":
    main()
