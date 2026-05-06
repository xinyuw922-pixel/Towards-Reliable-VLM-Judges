#!/usr/bin/env python3
"""
Apply the Memory hybrid first-frame rule to Task D / Task D-F.

This creates alternate exam artifacts:
- Task D: prepend a real cue-room first frame to Memory rows only
- Task D-F: same treatment for Memory rows only

Non-Memory rows remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import minigrid
from PIL import Image, ImageDraw


EXAM_SCHEMA_VERSION = "gridwm.exam.v1"
OVERLAY_PAD = 6
OVERLAY_BOX_PADDING = 2


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_frame(raw_root: Path, env: str, rel_path: str) -> Image.Image:
    p = raw_root / env / rel_path
    if not p.exists():
        raise FileNotFoundError(f"Frame not found: {p}")
    return Image.open(p).convert("RGB")


def annotate_frame(img: Image.Image, label: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    try:
        from PIL import ImageFont

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                max(10, min(w, h) // 12),
            )
        except OSError:
            font = ImageFont.load_default()
    except ImportError:
        font = None

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    bx0 = OVERLAY_PAD
    by0 = OVERLAY_PAD
    bx1 = bx0 + text_w + 2 * OVERLAY_BOX_PADDING
    by1 = by0 + text_h + 2 * OVERLAY_BOX_PADDING
    draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0, 160))
    draw.text(
        (bx0 + OVERLAY_BOX_PADDING, by0 + OVERLAY_BOX_PADDING),
        label,
        fill=(255, 255, 255, 255),
        font=font,
    )
    return out


def choose_cols(n_frames: int, min_cols: int = 4, max_cols: int = 8) -> int:
    raw = round(math.sqrt(n_frames))
    return max(min_cols, min(max_cols, raw))


def make_montage(frames: List[Image.Image]) -> Tuple[Image.Image, Dict[str, int | str]]:
    if not frames:
        raise ValueError("empty frame list")
    w, h = frames[0].size
    n_frames = len(frames)
    cols = choose_cols(n_frames)
    rows = math.ceil(n_frames / cols)
    montage = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
    for i, frame in enumerate(frames):
        col = i % cols
        row = i // cols
        montage.paste(frame, (col * w, row * h))
    return montage, {
        "cols": cols,
        "rows": rows,
        "n_frames": n_frames,
        "layout_mode": "adaptive",
        "min_cols": 4,
        "max_cols": 8,
    }


def pick_cue_and_ends(objs: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    xs = [o["pos"][0] for o in objs]
    min_x = min(xs)
    max_x = max(xs)
    cue = sorted([o for o in objs if o["pos"][0] == min_x], key=lambda o: (o["pos"][1], o["type"], o["color"]))[0]
    ends = sorted([o for o in objs if o["pos"][0] == max_x], key=lambda o: (o["pos"][1], o["type"], o["color"]))
    return cue, ends[0], ends[1]


def find_room_bbox_around_cue(u, cue: Dict[str, Any]) -> Tuple[int, int, int, int]:
    cx, cy = int(cue["pos"][0]), int(cue["pos"][1])

    def is_wall(x: int, y: int) -> bool:
        cell = u.grid.get(x, y)
        return cell is not None and getattr(cell, "type", None) == "wall"

    left = cx
    while left > 0 and not is_wall(left, cy):
        left -= 1
    right = cx
    while right < u.width - 1 and not is_wall(right, cy):
        right += 1
    top = cy
    while top > 0 and not is_wall(cx, top):
        top -= 1
    bottom = cy
    while bottom < u.height - 1 and not is_wall(cx, bottom):
        bottom += 1
    return left, top, right + 1, bottom + 1


def build_cue_room_frame(env_id: str, seed: int, cue: Dict[str, Any], tile_size: int = 32) -> Tuple[Image.Image, List[int]]:
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        env.reset(seed=int(seed))
        full = Image.fromarray(env.unwrapped.get_frame(tile_size=tile_size, agent_pov=False, highlight=False))
        left_tile, top_tile, right_tile, bottom_tile = find_room_bbox_around_cue(env.unwrapped, cue)
        crop = full.crop((
            left_tile * tile_size,
            top_tile * tile_size,
            right_tile * tile_size,
            bottom_tile * tile_size,
        ))
        cue_frame = crop.resize((tile_size * 7, tile_size * 7), Image.Resampling.NEAREST)
        return cue_frame, [left_tile, top_tile, right_tile - 1, bottom_tile - 1]
    finally:
        env.close()


def load_group_records(raw_root: Path, env: str) -> Dict[str, Dict[str, dict]]:
    groups: Dict[str, Dict[str, dict]] = {}
    for rec in read_jsonl(raw_root / env / "triplets.jsonl"):
        groups.setdefault(rec["group_id"], {})[rec["variant"]] = rec
    return groups


def build_memory_hybrid_montage(raw_root: Path, env: str, rec: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
    cue, end_a, end_b = pick_cue_and_ends(rec["state_seq"][0]["objects"])
    cue_frame, cue_bbox = build_cue_room_frame(rec["env_id"], int(rec["seed"]), cue)
    frames = [annotate_frame(cue_frame, "cue")]
    for step_idx, rel in enumerate(rec["frames"]):
        frames.append(annotate_frame(load_frame(raw_root, env, rel), f"t={step_idx}"))
    montage, layout = make_montage(frames)
    meta = {
        "cue": cue,
        "end_a": end_a,
        "end_b": end_b,
        "cue_room_bbox": cue_bbox,
        "n_frames_original": len(rec["frames"]),
        "n_frames_hybrid": len(frames),
    }
    return montage, {"layout": layout, "meta": meta}


def apply_to_taskd(taskd_exam: Path, raw_root: Path, out_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    rows = read_jsonl(taskd_exam)
    groups = load_group_records(raw_root, "memory")
    out_rows: List[dict] = []
    changed: List[Dict[str, Any]] = []

    img_root = out_dir / "images" / "taskD_memory_hybrid"
    for row in rows:
        env = row["uid"].split(".")[1]
        if env != "memory":
            out_rows.append(row)
            continue
        gid = row["source_group_id"]
        variant = row["source_variant"]
        rec = groups[gid][variant]
        montage, extra = build_memory_hybrid_montage(raw_root, "memory", rec)
        rel = f"memory/{gid}/{variant}_orig_clean_hybrid.png"
        full_path = img_root / rel
        full_path.parent.mkdir(parents=True, exist_ok=True)
        montage.save(full_path, "PNG")
        new_row = dict(row)
        new_row["image"] = f"images/taskD_memory_hybrid/{rel}"
        new_row["n_frames"] = int(extra["meta"]["n_frames_hybrid"])
        new_row["layout"] = extra["layout"]
        new_row["hybrid_memory_first_frame"] = True
        new_row["memory_cue_room_bbox"] = extra["meta"]["cue_room_bbox"]
        out_rows.append(new_row)
        changed.append({
            "uid": row["uid"],
            "group_id": gid,
            "variant": variant,
            "image": new_row["image"],
            "cue_room_bbox": extra["meta"]["cue_room_bbox"],
        })

    out_exam = out_dir / "task_d_exam_memory_hybrid.jsonl"
    write_jsonl(out_exam, out_rows)
    return out_exam, {
        "source_exam": str(taskd_exam),
        "output_exam": str(out_exam),
        "changed_memory_rows": len(changed),
        "changes": changed,
    }


def apply_to_taskdf(taskdf_exam: Path, raw_root: Path, out_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    rows = read_jsonl(taskdf_exam)
    groups = load_group_records(raw_root, "memory")
    out_rows: List[dict] = []
    changed: List[Dict[str, Any]] = []
    img_root = out_dir / "images" / "taskDF_memory_hybrid"
    image_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in rows:
        env = row["uid"].split(".")[1]
        if env != "memory":
            out_rows.append(row)
            continue
        gid = row["source_group_id"]
        variant = row["source_variant"]
        key = (gid, variant)
        if key not in image_cache:
            rec = groups[gid][variant]
            montage, extra = build_memory_hybrid_montage(raw_root, "memory", rec)
            rel = f"{gid}/{variant}_orig_clean_hybrid.png"
            full_path = img_root / rel
            full_path.parent.mkdir(parents=True, exist_ok=True)
            montage.save(full_path, "PNG")
            image_cache[key] = {
                "image": f"images/taskDF_memory_hybrid/{rel}",
                "layout": extra["layout"],
                "cue_room_bbox": extra["meta"]["cue_room_bbox"],
            }
        new_row = dict(row)
        new_row["image"] = image_cache[key]["image"]
        new_row["n_frames"] = 1 + int(new_row["n_frames"])
        new_row["layout"] = image_cache[key]["layout"]
        new_row["hybrid_memory_first_frame"] = True
        new_row["memory_cue_room_bbox"] = image_cache[key]["cue_room_bbox"]
        out_rows.append(new_row)
        changed.append({
            "uid": row["uid"],
            "group_id": gid,
            "variant": variant,
            "framing": row.get("framing"),
            "image": new_row["image"],
            "cue_room_bbox": image_cache[key]["cue_room_bbox"],
        })

    out_exam = out_dir / "task_d_exam_memory_hybrid.jsonl"
    write_jsonl(out_exam, out_rows)
    return out_exam, {
        "source_exam": str(taskdf_exam),
        "output_exam": str(out_exam),
        "changed_memory_rows": len(changed),
        "changes": changed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply Memory hybrid first frame to Task D / D-F")
    ap.add_argument("--raw-root", type=Path, default=Path("outputs/raw_data"))
    ap.add_argument("--taskd-exam", type=Path, default=Path("outputs/exams_taskd/task_d_exam.jsonl"))
    ap.add_argument("--taskdf-exam", type=Path, default=Path("outputs/exams_taskd/taskdf_exam.jsonl"))
    ap.add_argument("--taskd-out-dir", type=Path, default=Path("outputs/exams_taskd"))
    ap.add_argument("--taskdf-out-dir", type=Path, default=Path("outputs/exams_taskdf"))
    args = ap.parse_args()

    out_taskd_exam, taskd_manifest = apply_to_taskd(args.taskd_exam, args.raw_root, args.taskd_out_dir)
    out_taskdf_exam, taskdf_manifest = apply_to_taskdf(args.taskdf_exam, args.raw_root, args.taskdf_out_dir)

    write_json(args.taskd_out_dir / "memory_hybrid_manifest.json", taskd_manifest)
    write_json(args.taskdf_out_dir / "memory_hybrid_manifest.json", taskdf_manifest)

    print(json.dumps({
        "taskd_exam": str(out_taskd_exam),
        "taskdf_exam": str(out_taskdf_exam),
        "taskd_changed_memory_rows": taskd_manifest["changed_memory_rows"],
        "taskdf_changed_memory_rows": taskdf_manifest["changed_memory_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
