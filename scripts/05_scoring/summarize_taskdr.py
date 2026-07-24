#!/usr/bin/env python3
"""Summarize frozen Task D-R per-row scores into paper-facing metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

FIELDS = [
    "representation", "model", "JAccneu_D", "Acc_full_D", "Acc_nocue_D",
    "Acc_cf_D", "SuccD", "IDR_raw_D", "IDR_cond_D", "G_D", "G_active_D",
    "p_active_D", "G_active_pct_D", "Degenerate_Rate_D", "Degenerate_D",
    "NS_frame_D", "JCR_frame_D", "Flip_frame_D", "NS_vis_D", "JCR_vis_D",
    "Flip_vis_D", "Avg_NS_D", "Avg_JCR_D", "LES_frame_D",
    "d_logit_frame_D", "LES_frame_method", "LES_vis_D", "d_logit_vis_D",
    "LES_vis_method", "n_rows_D_raw", "n_groups_D", "source",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (item["group_id"], item["uid"])):
        by_group.setdefault(row["group_id"], row)
    return list(by_group.values())


def prediction_map(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    return {row["group_id"]: row.get("pred_norm") for row in dedupe(rows)}


def pair_ns(left: dict[str, str | None], right: dict[str, str | None]) -> float | None:
    values = []
    for group_id in sorted(set(left) & set(right)):
        pair = (left[group_id], right[group_id])
        if all(value in {"Success", "Fail"} for value in pair):
            values.append(float(pair[0] == pair[1]))
    return mean(values)


def joint_consistency(*maps: dict[str, str | None]) -> float | None:
    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)
    values = []
    for group_id in sorted(common):
        predictions = [mapping[group_id] for mapping in maps]
        if all(value in {"Success", "Fail"} for value in predictions):
            values.append(float(len(set(predictions)) == 1))
    return mean(values)


def classify(p_active: float | None, acc_cf: float | None) -> str:
    if p_active is None:
        return ""
    if p_active < 0.1:
        return "all_fail"
    if p_active > 0.9 and acc_cf is not None and acc_cf < 0.1:
        return "success_biased"
    return "nondegenerate"


def les(
    rows: list[dict[str, Any]], model: str, family: str, baseline: str
) -> tuple[float | None, float | None, str]:
    """Apply the frozen L3 McNemar effect-size rule."""
    if not rows:
        return None, None, "unavailable"
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(row["group_id"], []).append(row)
    discordant_01 = 0
    discordant_10 = 0
    for group_rows in by_group.values():
        baseline_rows = [row for row in group_rows if row.get(family) == baseline]
        probe_rows = [row for row in group_rows if row.get(family) != baseline]
        if not baseline_rows or not probe_rows:
            continue
        left = baseline_rows[0].get("pred_norm")
        right = probe_rows[0].get("pred_norm")
        if left not in {"Success", "Fail"} or right not in {"Success", "Fail"}:
            continue
        if left == "Fail" and right == "Success":
            discordant_01 += 1
        elif left == "Success" and right == "Fail":
            discordant_10 += 1
    if discordant_10 > 0 and discordant_01 > 0:
        beta = math.log(discordant_01 / discordant_10)
    else:
        beta = 0.0
    effect = abs(beta)
    return effect, effect / (math.pi / math.sqrt(3)), "L3_McNemar"


def summarize_model(model_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(model_dir / "score_per_row.jsonl")
    model = model_dir.name

    def select(variant: str, visual: str = "clean", framing: str = "neu"):
        return dedupe([
            row for row in rows
            if row.get("variant") == variant
            and row.get("temporal") == "orig"
            and row.get("visual") == visual
            and row.get("framing") == framing
        ])

    baseline = {variant: select(variant) for variant in ("full", "nocue", "cf")}

    def accuracy(variant: str) -> float | None:
        return mean(float(row.get("correct") is True) for row in baseline[variant])

    full_map = prediction_map(baseline["full"])
    cf_map = prediction_map(baseline["cf"])
    valid_groups = [
        group_id for group_id in sorted(set(full_map) & set(cf_map))
        if full_map[group_id] in {"Success", "Fail"}
        and cf_map[group_id] in {"Success", "Fail"}
    ]
    active = sum(full_map[group_id] == "Success" for group_id in valid_groups)
    discriminated = sum(
        full_map[group_id] == "Success" and cf_map[group_id] == "Fail"
        for group_id in valid_groups
    )
    same = sum(full_map[group_id] == cf_map[group_id] for group_id in valid_groups)
    total = len(valid_groups)
    p_active = active / total if total else None
    idr_raw = discriminated / total if total else None
    idr_cond = idr_raw / p_active if p_active and idr_raw is not None else None

    neutral = prediction_map(select("full"))
    positive = prediction_map(select("full", framing="pos"))
    negative = prediction_map(select("full", framing="neg"))
    style = prediction_map(select("full", visual="style"))
    ns_frame = mean([pair_ns(neutral, positive), pair_ns(neutral, negative)])
    jcr_frame = joint_consistency(neutral, positive, negative)
    ns_visual = pair_ns(neutral, style)
    jcr_visual = joint_consistency(neutral, style)

    full_rows = [row for row in rows if row.get("variant") == "full"]
    frame_rows = [
        row for row in full_rows
        if row.get("temporal") == "orig"
        and row.get("visual") == "clean"
        and row.get("framing") in {"neu", "pos", "neg"}
    ]
    visual_rows = [
        row for row in full_rows
        if row.get("temporal") == "orig"
        and row.get("framing") == "neu"
        and row.get("visual") in {"clean", "style"}
    ]
    les_frame, d_frame, method_frame = les(frame_rows, model, "framing", "neu")
    les_visual, d_visual, method_visual = les(visual_rows, model, "visual", "clean")
    acc_cf = accuracy("cf")

    return {
        "representation": "composite",
        "model": model,
        "JAccneu_D": accuracy("full"),
        "Acc_full_D": accuracy("full"),
        "Acc_nocue_D": accuracy("nocue"),
        "Acc_cf_D": acc_cf,
        "SuccD": mean(float(row.get("pred_norm") == "Success") for row in baseline["full"]),
        "IDR_raw_D": idr_raw,
        "IDR_cond_D": idr_cond,
        "G_D": total,
        "G_active_D": active,
        "p_active_D": p_active,
        "G_active_pct_D": p_active,
        "Degenerate_Rate_D": same / total if total else None,
        "Degenerate_D": classify(p_active, acc_cf),
        "NS_frame_D": ns_frame,
        "JCR_frame_D": jcr_frame,
        "Flip_frame_D": 1 - jcr_frame if jcr_frame is not None else None,
        "NS_vis_D": ns_visual,
        "JCR_vis_D": jcr_visual,
        "Flip_vis_D": 1 - jcr_visual if jcr_visual is not None else None,
        "Avg_NS_D": mean([ns_frame, ns_visual]),
        "Avg_JCR_D": mean([jcr_frame, jcr_visual]),
        "LES_frame_D": les_frame,
        "d_logit_frame_D": d_frame,
        "LES_frame_method": method_frame,
        "LES_vis_D": les_visual,
        "d_logit_vis_D": d_visual,
        "LES_vis_method": method_visual,
        "n_rows_D_raw": len(rows),
        "n_groups_D": len(baseline["full"]),
        "source": f"{model_dir.as_posix()}/score_per_row.jsonl",
    }


def format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dirs = sorted(path for path in args.scores_root.iterdir() if path.is_dir())
    rows = [summarize_model(path) for path in model_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in FIELDS})
    print(f"Wrote {len(rows)} model rows to {args.output}")


if __name__ == "__main__":
    main()
