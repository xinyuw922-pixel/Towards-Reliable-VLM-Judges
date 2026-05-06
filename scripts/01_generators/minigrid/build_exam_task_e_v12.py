#!/usr/bin/env python3
"""
build_exam_task_e_v12.py — Phase 56a: Task E v1.2 Exam Expansion.

v1.2 expands v1.1 (29 rows, 1-2 groups) to use all 6 keycorridor groups
with fixed per-group role sampling. Adds sampling_role field for analysis.

Two main query families:
  - identity_of_first_picked_object
  - status_of_first_picked_object_at_checkpoint

USAGE:
  python scripts/build_exam_task_e_v12.py \
    --root datasets/raw_data \
    --out-dir /tmp/taske_v12_check

  python scripts/build_exam_task_e_v12.py \
    --root datasets/raw_data \
    --out-dir datasets/exams_taske_v12_candidate
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import sys as _sys

_scripts_dir = Path(__file__).resolve().parent
_root = _scripts_dir.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
elif str(_scripts_dir) not in _sys.path:
    _sys.path.insert(0, str(_scripts_dir))

from PIL import Image

# ─── Schema ───────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "gridwm.exam.v1.2"
GENERATOR_VERSION = "task_e_v12_expansion"

# ─── Label Spaces ─────────────────────────────────────────────────────────────

IDENTITY_LABELS = [
    "key_yellow", "key_green", "key_purple", "key_blue",
    "key_red", "key_grey",
    "ball_yellow", "ball_green", "ball_blue", "ball_purple", "ball_red",
]

STATUS_LABELS = [
    "holding",
    "dropped",
    "reacquired_and_holding",
]

# ─── Horizon Buckets ───────────────────────────────────────────────────────────

HORIZON_SHORT = "short"    # 1–10
HORIZON_MEDIUM = "medium"  # 11–25
HORIZON_LONG = "long"     # 26+

HORIZON_BUCKETS = {
    (1, 10): HORIZON_SHORT,
    (11, 25): HORIZON_MEDIUM,
    (26, 9999): HORIZON_LONG,
}

# ─── Query Templates ───────────────────────────────────────────────────────────

IDENTITY_TEMPLATE = (
    "Below is a MiniGrid trajectory storyboard showing the trajectory up to the last shown frame only.\n"
    "Question: Earlier in the shown trajectory, what was the first object the agent picked up?\n"
    "Answer using JSON only: {{\"answer\":\"<one label from {label_list}>\"}}"
)

STATUS_TEMPLATE = (
    "Below is a MiniGrid trajectory storyboard showing the trajectory up to the last shown frame only.\n"
    "Question: By the last shown frame, what is the status of the first object the agent picked up earlier?\n"
    "Answer using JSON only: {{\"answer\":\"<one label from holding, dropped, reacquired_and_holding>\"}}"
)

# ─── Normalization ─────────────────────────────────────────────────────────────

def norm_carrying_label(carrying: Any) -> str:
    if carrying is None:
        return "none"
    if isinstance(carrying, dict):
        return f"{carrying.get('type', 'unknown')}_{carrying.get('color', 'unknown')}"
    return str(carrying)


def normalize_trace(state_seq: List[dict]) -> List[dict]:
    steps = []
    prev_carry = None
    for i, s in enumerate(state_seq):
        if not isinstance(s, dict):
            steps.append({"step": i, "raw": s})
            continue
        agent = s.get("agent", {})
        carrying = agent.get("carrying")
        carry_label = norm_carrying_label(carrying)
        steps.append({
            "step": i,
            "carrying": carrying,
            "carrying_label": carry_label,
            "carry_changed": (carrying != prev_carry),
        })
        prev_carry = carrying
    return steps


# ─── Event Extraction ──────────────────────────────────────────────────────────

def extract_events(trace: List[dict]) -> Dict[str, Any]:
    pickups = []
    drops = []
    reacquires = []
    ever_held = set()
    prev_carry = None

    for st in trace:
        i = st["step"]
        carrying = st["carrying"]
        label = st["carrying_label"]

        if carrying is not None and prev_carry is None:
            if label in ever_held:
                reacquires.append((i, label))
            else:
                pickups.append((i, label))
                ever_held.add(label)
        elif carrying is None and prev_carry is not None:
            drops.append((i, norm_carrying_label(prev_carry)))
            ever_held.add(norm_carrying_label(prev_carry))

        prev_carry = carrying

    return {
        "pickups": pickups,
        "drops": drops,
        "reacquires": reacquires,
        "terminal_label": trace[-1]["carrying_label"] if trace else "none",
        "n_steps": len(trace),
    }


# ─── Status Classification ─────────────────────────────────────────────────────

def classify_status(
    first_label: str,
    pickups: List[Tuple[int, str]],
    drops: List[Tuple[int, str]],
    reacquires: List[Tuple[int, str]],
    query_step: int,
) -> str:
    lp = [(s, l) for s, l in pickups if l == first_label]
    ld = [(s, l) for s, l in drops if l == first_label]
    lr = [(s, l) for s, l in reacquires if l == first_label]

    if not lp:
        return "dropped"

    last_pickup_before_q = max((s for s, l in lp if s <= query_step), default=None)
    last_drop_before_q = max((s for s, l in ld if s <= query_step), default=None)
    last_reacq_before_q = max((s for s, l in lr if s <= query_step), default=None)

    if last_reacq_before_q is not None:
        return "reacquired_and_holding"
    if last_pickup_before_q is not None:
        if last_drop_before_q is None or last_drop_before_q < last_pickup_before_q:
            return "holding"
        if last_drop_before_q > last_pickup_before_q:
            return "dropped"
    return "dropped"


# ─── Horizon Helpers ───────────────────────────────────────────────────────────

def assign_horizon_bucket(gap: int) -> str:
    for (lo, hi), bucket in HORIZON_BUCKETS.items():
        if lo <= gap <= hi:
            return bucket
    return HORIZON_LONG


# ─── Role-Specific Query Step Selection ───────────────────────────────────────

def pick_identity_steps(
    first_pickup_step: int,
    n_steps: int,
) -> Dict[str, Optional[int]]:
    """
    Pick canonical query_step for each identity role.
    Returns {role: query_step_or_None}.
    """
    # Short: earliest step in short bucket
    short_candidates = [
        qs for qs in range(first_pickup_step + 1, min(first_pickup_step + 11, n_steps))
        if qs - first_pickup_step <= 10
    ]
    # Medium: earliest step in medium bucket
    medium_candidates = [
        qs for qs in range(first_pickup_step + 11, min(first_pickup_step + 26, n_steps))
        if 11 <= qs - first_pickup_step <= 25
    ]
    # Long: earliest step in long bucket
    long_candidates = [
        qs for qs in range(first_pickup_step + 26, n_steps)
        if qs - first_pickup_step >= 26
    ]
    return {
        "identity_short": short_candidates[0] if short_candidates else None,
        "identity_medium": medium_candidates[0] if medium_candidates else None,
        "identity_long": long_candidates[0] if long_candidates else None,
    }


def pick_status_steps(
    first_pickup_step: int,
    first_drop_step: Optional[int],
    n_steps: int,
) -> Dict[str, Optional[int]]:
    """
    Pick canonical query_step for each status role.
    """
    results = {}

    # early_holding: 1-3 steps after first pickup
    early_cands = [qs for qs in range(first_pickup_step + 1, min(first_pickup_step + 4, n_steps))]
    results["status_early_holding"] = early_cands[0] if early_cands else None

    # mid_holding: roughly halfway between pickup and drop (or pickup+15 if no drop)
    if first_drop_step is not None and first_drop_step > first_pickup_step + 5:
        mid = (first_pickup_step + first_drop_step) // 2
        results["status_mid_holding"] = mid
    else:
        # No clear drop gap; pick around step +15
        results["status_mid_holding"] = min(first_pickup_step + 15, n_steps - 1)

    # first_post_drop: earliest step strictly after the drop
    if first_drop_step is not None:
        post_drop = first_drop_step + 1
        results["status_first_post_drop"] = post_drop if post_drop < n_steps else None
        # late_post_drop: use n-1 (last frame), but if n-1 == post_drop (rare),
        # use n-2 (second-to-last). n-1 always shows the final state.
        late = n_steps - 1
        if late == post_drop:
            late = n_steps - 2
        results["status_late_post_drop"] = late if late > first_drop_step and late < n_steps else None
    else:
        # Never dropped — no post-drop roles
        results["status_first_post_drop"] = None
        results["status_late_post_drop"] = None

    return results


# ─── Frame Selection (prefix-only) ────────────────────────────────────────────

def select_prefix_frames(
    query_step: int,
    first_pickup_step: int,
    k_min: int = 5,
    k_max: int = 12,
) -> List[int]:
    n = query_step + 1
    selected = set()
    selected.add(0)
    selected.add(query_step)
    if query_step >= 1:
        selected.add(query_step - 1)
    for dx in (-1, 0, 1):
        s = max(0, min(query_step, first_pickup_step + dx))
        selected.add(s)
    sorted_steps = sorted(selected)
    for i in range(len(sorted_steps) - 1):
        gap = sorted_steps[i + 1] - sorted_steps[i]
        if gap > 3:
            mid = (sorted_steps[i] + sorted_steps[i + 1]) // 2
            selected.add(mid)
    result = sorted(selected)
    if len(result) > k_max:
        return result  # caller must check and reject if > k_max
    if len(result) < k_min and n > len(result):
        candidates = []
        if result[0] > 0:
            candidates.append(result[0] - 1)
        if result[-1] < query_step:
            candidates.append(result[-1] + 1)
        for c in candidates:
            if len(result) < k_min and c not in result and 0 <= c <= query_step:
                result.append(c)
        result = sorted(result)
    return result


# ─── Layout ────────────────────────────────────────────────────────────────────

def compute_layout(n_frames: int) -> Dict[str, Any]:
    if n_frames <= 1:
        return {"rows": 1, "cols": 1, "n_cells": 1, "n_empty": 0}
    best_r, best_c = 1, n_frames
    best_score = float("inf")
    for r in range(1, n_frames + 1):
        c = (n_frames + r - 1) // r
        cells = r * c
        n_empty = cells - n_frames
        aspect_ratio = c / r if r > 0 else float("inf")
        score = abs(math.log(aspect_ratio)) * 2 + n_empty if aspect_ratio > 0 else float("inf")
        if score < best_score:
            best_score = score
            best_r, best_c = r, c
    if best_c < best_r:
        best_r, best_c = best_c, best_r
    return {
        "rows": best_r,
        "cols": best_c,
        "n_cells": best_r * best_c,
        "n_empty": best_r * best_c - n_frames,
    }


# ─── Rendering ─────────────────────────────────────────────────────────────────

def load_frame(env_dir: Path, frame_rel: str) -> Image.Image:
    path = env_dir / frame_rel
    if not path.exists():
        raise FileNotFoundError(f"Missing frame: {path}")
    return Image.open(path).convert("RGB")


def render_montage(
    env_dir: Path,
    frames: List[str],
    selected_steps: List[int],
    layout: Dict[str, Any],
    out_path: Path,
) -> Image.Image:
    images = [load_frame(env_dir, frames[i]) for i in selected_steps]
    if not images:
        raise ValueError(f"No frames for montage: {out_path}")
    w, h = images[0].size
    rows, cols = layout.get("rows", 1), layout.get("cols", len(images))
    canvas = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
    for idx, img in enumerate(images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        canvas.paste(img, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return canvas


# ─── Exam Row Builder ─────────────────────────────────────────────────────────

def build_v12_row(
    env: str,
    gid: str,
    query_type: str,
    query_step: int,
    anchor_step: int,
    horizon_gap: int,
    horizon_bucket: str,
    answer: str,
    first_label: Optional[str],
    frames: List[str],
    env_dir: Path,
    out_dir: Path,
    row_idx: int,
    sampling_role: str,
) -> dict:
    qs_bucket_str = horizon_bucket[:3]

    if query_type == "identity_of_first_picked_object":
        label_space = IDENTITY_LABELS
        prompt = IDENTITY_TEMPLATE.format(label_list=", ".join(IDENTITY_LABELS))
        img_filename = f"identity_first_pick_{qs_bucket_str}_q{row_idx:03d}.png"
    else:
        label_space = STATUS_LABELS
        prompt = STATUS_TEMPLATE
        img_filename = f"status_first_pick_{qs_bucket_str}_q{row_idx:03d}.png"

    img_rel = f"images/taskE_v12/{env}/{gid}/{img_filename}"
    img_out = out_dir / img_rel

    selected_steps = select_prefix_frames(query_step, anchor_step, k_min=5, k_max=12)

    assert all(0 <= s <= query_step for s in selected_steps), \
        f"Frame step {selected_steps} exceeds query_step {query_step}"

    layout = compute_layout(len(selected_steps))
    render_montage(env_dir, frames, selected_steps, layout, img_out)

    exam_id = f"E_v12.{env}.{gid}.{sampling_role}.q{row_idx:03d}"
    uid = exam_id

    return {
        "schema_version": SCHEMA_VERSION,
        "task": "E",
        "query_type": query_type,
        "exam_id": exam_id,
        "uid": uid,
        "env_task": env,
        "group_id": gid,
        "image": img_rel,
        "prompt": prompt,
        "label_space": label_space,
        "answer": answer,
        "answer_json": {"answer": answer},
        "query_step": query_step,
        "history_end_step": query_step,
        "anchor_step": anchor_step,
        "horizon_gap": horizon_gap,
        "horizon_bucket": horizon_bucket,
        "selected_steps": selected_steps,
        "sampling_role": sampling_role,
        "support_steps": {
            "first_pickup": anchor_step,
            "query_step": query_step,
        },
        "first_picked_object_gold": first_label,
        "meta": {
            "query_semantics": "checkpoint_prefix_only",
            "generator_version": GENERATOR_VERSION,
            "schema": SCHEMA_VERSION,
        },
    }


# ─── I/O ──────────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(out_dir: Path, rows: List[dict]) -> None:
    by_env = defaultdict(int)
    by_qt = defaultdict(int)
    by_hb = defaultdict(int)
    by_role = defaultdict(int)
    by_env_qt_hb = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    by_env_role = defaultdict(lambda: defaultdict(int))

    for row in rows:
        by_env[row["env_task"]] += 1
        by_qt[row["query_type"]] += 1
        by_hb[row["horizon_bucket"]] += 1
        by_role[row["sampling_role"]] += 1
        by_env_qt_hb[row["env_task"]][row["query_type"]][row["horizon_bucket"]] += 1
        by_env_role[row["env_task"]][row["sampling_role"]] += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "exam_file": "task_e_exam_v12.jsonl",
        "n_rows": len(rows),
        "description": (
            "Task E v1.2 expanded exam. "
            "Prefix-only storyboard (0..query_step). "
            "Two query families: identity_of_first_picked_object, "
            "status_of_first_picked_object_at_checkpoint. "
            "sampling_role field added for role-level analysis."
        ),
        "label_spaces": {
            "identity_of_first_picked_object": IDENTITY_LABELS,
            "status_of_first_picked_object_at_checkpoint": STATUS_LABELS,
        },
        "distribution": {
            "by_env": dict(by_env),
            "by_query_type": dict(by_qt),
            "by_horizon_bucket": dict(by_hb),
            "by_sampling_role": dict(by_role),
            "by_env_x_role": {env: dict(roles) for env, roles in by_env_role.items()},
            "by_env_x_query_x_bucket": {
                env: {qt: dict(buckets) for qt, buckets in v.items()}
                for env, v in by_env_qt_hb.items()
            },
        },
    }

    path = out_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Manifest written: {path}")


# ─── Auto Gate ────────────────────────────────────────────────────────────────

def run_auto_gate(rows: List[dict], out_dir: Path) -> Tuple[bool, List[str]]:
    failures = []

    exam_path = out_dir / "task_e_exam_v12.jsonl"
    if not exam_path.exists():
        failures.append("G1: exam JSONL missing")
    else:
        failures.append("G1: PASS")

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        failures.append("G2: manifest missing")
    else:
        failures.append("G2: PASS")

    for row in rows:
        img_path = out_dir / row["image"]
        if not img_path.exists():
            failures.append(f"G3: missing image {row['image']}")
    if not any(f.startswith("G3") for f in failures):
        failures.append("G3: PASS")

    for row in rows:
        qs = row["query_step"]
        for s in row["selected_steps"]:
            if s > qs:
                failures.append(f"G4: selected_step {s} > query_step {qs} in {row['exam_id']}")
    if not any(f.startswith("G4") for f in failures):
        failures.append("G4: PASS")

    for row in rows:
        if row["answer"] not in row["label_space"]:
            failures.append(f"G5: answer {row['answer']} not in label_space for {row['exam_id']}")
    if not any(f.startswith("G5") for f in failures):
        failures.append("G5: PASS")

    # Check for duplicate group + query_type + query_step
    seen = {}
    for row in rows:
        k = (row["group_id"], row["query_type"], row["query_step"])
        if k in seen:
            failures.append(f"G6: duplicate key {k} (rows: {seen[k]}, {row['exam_id']})")
        seen[k] = row["exam_id"]
    if not any(f.startswith("G6") for f in failures):
        failures.append("G6: PASS")

    pass_all = not any(f.startswith("G") and not f.endswith("PASS") for f in failures)
    return pass_all, failures


# ─── Main Builder ─────────────────────────────────────────────────────────────

def build_v12(root: Path, out_dir: Path) -> Tuple[List[dict], Dict[str, Any]]:
    rows = []
    stats = defaultdict(int)

    env_configs = [
        {"env": "keycorridor", "do_retention_control": False},
        {"env": "doorkey", "do_retention_control": True},
    ]

    for cfg in env_configs:
        env = cfg["env"]
        env_dir = root / env
        triplets_path = env_dir / "triplets.jsonl"
        if not triplets_path.exists():
            print(f"[WARN] {env}: triplets.jsonl not found")
            continue

        triplets = read_jsonl(triplets_path)

        seen_gids = {}
        for rec in triplets:
            gid = rec.get("group_id", "")
            if gid in seen_gids:
                continue
            if rec.get("variant") == "full":
                seen_gids[gid] = rec
            elif gid not in seen_gids:
                seen_gids[gid] = rec

        for gid, rec in sorted(seen_gids.items()):
            state_seq = rec.get("state_seq", [])
            frames = rec.get("frames", [])
            if not state_seq or not frames:
                stats["missing_state_seq"] += 1
                continue

            trace = normalize_trace(state_seq)
            events = extract_events(trace)
            n = events["n_steps"]

            pickups = [(s, l) for s, l in events["pickups"] if l != "none"]
            drops = events["drops"]

            if not pickups:
                stats[f"no_pickup_{env}"] += 1
                continue

            first_pickup_step = pickups[0][0]
            first_label = pickups[0][1]
            first_drop_step = drops[0][0] if drops else None

            if cfg["do_retention_control"]:
                # Doorkey: retained_control
                # Use earliest short holding step
                short_cands = [
                    qs for qs in range(first_pickup_step + 1, min(first_pickup_step + 11, n))
                    if qs - first_pickup_step <= 10
                ]
                qs = short_cands[0] if short_cands else first_pickup_step + 1
                gap = qs - first_pickup_step
                bucket = assign_horizon_bucket(gap)
                selected = select_prefix_frames(qs, first_pickup_step)
                if len(selected) <= 12 and qs < n:
                    try:
                        row = build_v12_row(
                            env=env, gid=gid,
                            query_type="status_of_first_picked_object_at_checkpoint",
                            query_step=qs, anchor_step=first_pickup_step,
                            horizon_gap=gap, horizon_bucket=bucket,
                            answer="holding",
                            first_label=first_label,
                            frames=frames, env_dir=env_dir,
                            out_dir=out_dir, row_idx=len(rows),
                            sampling_role="retained_control",
                        )
                        rows.append(row)
                        stats[f"row_{gid}_retained_control"] += 1
                        stats["row_total"] += 1
                    except Exception as ex:
                        stats[f"render_error"] += 1

                # Doorkey: add medium if available (e.g. s000005 has medium)
                medium_cands = [
                    qs for qs in range(first_pickup_step + 11, n)
                    if 11 <= qs - first_pickup_step <= 25
                ]
                if medium_cands:
                    qs2 = medium_cands[0]
                    gap2 = qs2 - first_pickup_step
                    bucket2 = assign_horizon_bucket(gap2)
                    selected2 = select_prefix_frames(qs2, first_pickup_step)
                    if len(selected2) <= 12 and qs2 < n:
                        try:
                            row2 = build_v12_row(
                                env=env, gid=gid,
                                query_type="status_of_first_picked_object_at_checkpoint",
                                query_step=qs2, anchor_step=first_pickup_step,
                                horizon_gap=gap2, horizon_bucket=bucket2,
                                answer="holding",
                                first_label=first_label,
                                frames=frames, env_dir=env_dir,
                                out_dir=out_dir, row_idx=len(rows),
                                sampling_role="retained_control_medium",
                            )
                            rows.append(row2)
                            stats[f"row_{gid}_retained_control_medium"] += 1
                            stats["row_total"] += 1
                        except Exception:
                            stats["render_error"] += 1

            else:
                # Keycorridor: identity roles
                identity_steps = pick_identity_steps(first_pickup_step, n)

                for role, qs in identity_steps.items():
                    if qs is None:
                        stats[f"no_candidate_{role}_{gid}"] += 1
                        continue
                    gap = qs - first_pickup_step
                    bucket = assign_horizon_bucket(gap)
                    selected = select_prefix_frames(qs, first_pickup_step)
                    if len(selected) > 12:
                        stats[f"kmax_exceeded_{role}_{gid}"] += 1
                        continue
                    if qs >= n:
                        continue
                    try:
                        row = build_v12_row(
                            env=env, gid=gid,
                            query_type="identity_of_first_picked_object",
                            query_step=qs, anchor_step=first_pickup_step,
                            horizon_gap=gap, horizon_bucket=bucket,
                            answer=first_label,
                            first_label=first_label,
                            frames=frames, env_dir=env_dir,
                            out_dir=out_dir, row_idx=len(rows),
                            sampling_role=role,
                        )
                        rows.append(row)
                        stats[f"row_{gid}_{role}"] += 1
                        stats["row_total"] += 1
                    except Exception as ex:
                        stats[f"render_error_{role}_{gid}"] += 1

                # Keycorridor: status roles
                status_steps = pick_status_steps(first_pickup_step, first_drop_step, n)

                # Deduplicate: if two roles collide on the same query_step,
                # keep only the first encountered and log the skip.
                used_status_qs = set()
                for role, qs in status_steps.items():
                    if qs is None:
                        stats[f"no_candidate_{role}_{gid}"] += 1
                        continue
                    if qs in used_status_qs:
                        # Same query_step already used by another status role
                        stats[f"dup_status_qs_{role}_{gid}"] += 1
                        continue
                    gap = qs - first_pickup_step
                    bucket = assign_horizon_bucket(gap)
                    status_answer = classify_status(
                        first_label,
                        pickups, drops, events["reacquires"], qs
                    )
                    # Skip holding status roles if object already dropped
                    if role.startswith("status_") and "holding" in role:
                        if status_answer != "holding":
                            stats[f"status_not_holding_{role}_{gid}"] += 1
                            continue
                    selected = select_prefix_frames(qs, first_pickup_step)
                    if len(selected) > 12:
                        stats[f"kmax_exceeded_{role}_{gid}"] += 1
                        continue
                    if qs >= n:
                        continue
                    try:
                        row = build_v12_row(
                            env=env, gid=gid,
                            query_type="status_of_first_picked_object_at_checkpoint",
                            query_step=qs, anchor_step=first_pickup_step,
                            horizon_gap=gap, horizon_bucket=bucket,
                            answer=status_answer,
                            first_label=first_label,
                            frames=frames, env_dir=env_dir,
                            out_dir=out_dir, row_idx=len(rows),
                            sampling_role=role,
                        )
                        rows.append(row)
                        used_status_qs.add(qs)
                        stats[f"row_{gid}_{role}"] += 1
                        stats["row_total"] += 1
                    except Exception as ex:
                        stats[f"render_error_{role}_{gid}"] += 1

    return rows, dict(stats)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build Task E v1.2 Expanded Exam")
    parser.add_argument(
        "--root", required=True,
        help="Root dir with env subdirs containing triplets.jsonl")
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for candidate exam")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    if not root.exists():
        print(f"[ERROR] Root not found: {root}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Building v1.2 exam (schema={SCHEMA_VERSION})")
    print(f"[INFO] Output: {out_dir}")

    rows, stats = build_v12(root, out_dir)

    exam_file = out_dir / "task_e_exam_v12.jsonl"
    write_jsonl(exam_file, rows)
    print(f"[INFO] Written {len(rows)} rows to {exam_file}")

    write_manifest(out_dir, rows)

    print(f"\n[INFO] === Auto Gate ===")
    passed, failures = run_auto_gate(rows, out_dir)
    for f in failures:
        tag = "PASS" if f.endswith("PASS") else "FAIL"
        print(f"  [{tag}] {f}")

    print(f"\n[INFO] === Row Distribution ===")
    by_cell = defaultdict(lambda: defaultdict(int))
    for row in rows:
        by_cell[row["env_task"]][row["sampling_role"]] += 1
    for env in sorted(by_cell):
        print(f"  {env}:")
        for role, cnt in sorted(by_cell[env].items()):
            print(f"    {role}: {cnt}")

    print(f"\n[INFO] === Stats ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    print(f"\n[RESULT] {'✅ PASS' if passed else '❌ FAIL'} auto gate — {len(rows)} rows")


if __name__ == "__main__":
    main()
