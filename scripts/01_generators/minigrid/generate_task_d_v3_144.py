#!/usr/bin/env python3
"""
Generate Task D exam from MiniGrid triplet data.

Design:
- Keep Task D semantics: full-frame continuous montage, variants = full/nocue/cf
- Select N groups per env (6 env x N groups x 3 variants = rows)
- Default source: scripts/outputs/raw_data (sampled data)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from PIL import Image, ImageDraw


EXAM_SCHEMA_VERSION = "gridwm.exam.v1"
DEFAULT_LAYOUT_MODE: Literal["fixed-5", "adaptive"] = "adaptive"
DEFAULT_MIN_COLS = 4
DEFAULT_MAX_COLS = 8
TASK_D_PROMPT = (
    "Below is the full time-ordered trajectory storyboard, arranged from left to right "
    "and top to bottom. Judge whether the agent ultimately completed the task. "
    "Answer with ONLY: Success or Fail."
)
DEFAULT_ENVS = ["doorkey", "keycorridor", "lavagap", "memory", "multiroom", "redblue"]
DEFAULT_VARIANTS = ["full", "nocue", "cf"]
OVERLAY_PAD = 6
OVERLAY_BOX_PADDING = 2


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_frame(raw_root: Path, env: str, rel_path: str) -> Image.Image:
    # rel_path like "doorkey_s000001/full/step_000.png"
    frame_path = raw_root / env / rel_path
    if frame_path.exists():
        return Image.open(frame_path).convert("RGB")

    # Try alternative path
    frame_path_alt = raw_root / rel_path
    if frame_path_alt.exists():
        return Image.open(frame_path_alt).convert("RGB")

    raise FileNotFoundError(f"Frame not found: {frame_path}")


def safe_load_frames(
    raw_root: Path, env: str, frames_rel: List[str]
) -> List[Image.Image]:
    """Load frames, skipping any that don't exist."""
    frames = []
    for step_idx, rel in enumerate(frames_rel):
        try:
            frames.append(annotate_frame(load_frame(raw_root, env, rel), step_idx))
        except FileNotFoundError:
            pass  # Skip missing frames
    return frames


def annotate_frame(img: Image.Image, step_idx: int) -> Image.Image:
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    label = f"t={step_idx}"

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
    return img


def choose_cols(
    n_frames: int,
    mode: str = DEFAULT_LAYOUT_MODE,
    min_cols: int = DEFAULT_MIN_COLS,
    max_cols: int = DEFAULT_MAX_COLS,
) -> int:
    if mode == "fixed-5":
        return 5
    raw = round(math.sqrt(n_frames))
    return max(min_cols, min(max_cols, raw))


def make_montage(
    frames: List[Image.Image],
    layout_mode: str = DEFAULT_LAYOUT_MODE,
    min_cols: int = DEFAULT_MIN_COLS,
    max_cols: int = DEFAULT_MAX_COLS,
) -> Tuple[Image.Image, Dict[str, int | str]]:
    if not frames:
        raise ValueError("make_montage received empty frames")

    w, h = frames[0].size
    n_frames = len(frames)
    cols = choose_cols(n_frames, layout_mode, min_cols, max_cols)
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
        "layout_mode": layout_mode,
        "min_cols": min_cols,
        "max_cols": max_cols,
    }


def load_group_variants(triplets_path: Path) -> Dict[str, Dict[str, dict]]:
    groups: Dict[str, Dict[str, dict]] = {}
    for rec in read_jsonl(triplets_path):
        groups.setdefault(rec["group_id"], {})[rec["variant"]] = rec
    return groups


def build_task_d_rows(
    raw_root: Path,
    out_dir: Path,
    *,
    envs: List[str],
    groups_per_env: int,
    layout_mode: str,
    min_cols: int,
    max_cols: int,
) -> Tuple[List[dict], Dict[str, Any]]:
    rows: List[dict] = []
    stats: Dict[str, Any] = {
        "envs": {},
        "selected_groups": {},
        "skipped": {},
        "made": 0,
        "groups_per_env": groups_per_env,
    }
    img_root = out_dir / "images" / "taskD"

    for env in envs:
        triplets_path = raw_root / env / "triplets.jsonl"
        groups = load_group_variants(triplets_path)
        eligible = [
            gid for gid in sorted(groups)
            if all(variant in groups[gid] for variant in DEFAULT_VARIANTS)
        ]
        selected = eligible[:groups_per_env]
        stats["selected_groups"][env] = selected
        stats["envs"][env] = 0
        stats["skipped"][env] = max(0, len(eligible) - len(selected))

        for gid in selected:
            for variant in DEFAULT_VARIANTS:
                rec = groups[gid][variant]
                frames_rel = rec.get("frames", [])
                success = bool(rec.get("success", False))

                frames = safe_load_frames(raw_root, env, frames_rel)
                if not frames:
                    print(f"⚠️  Skipping {env}/{gid}/{variant}: no valid frames")
                    continue
                montage, layout = make_montage(
                    frames,
                    layout_mode=layout_mode,
                    min_cols=min_cols,
                    max_cols=max_cols,
                )

                img_rel_path = f"{env}/{gid}/{variant}_orig_clean.png"
                img_full_path = img_root / img_rel_path
                img_full_path.parent.mkdir(parents=True, exist_ok=True)
                montage.save(img_full_path, "PNG")

                uid = f"D.{env}.{gid}.{variant}.orig.clean"
                row = {
                    "exam_id": uid,
                    "uid": uid,
                    "schema_version": EXAM_SCHEMA_VERSION,
                    "task": "D",
                    "env_task": env,
                    "env_id": env,
                    "group_id": gid,
                    "variant": variant,
                    "temporal": "orig",
                    "visual": "clean",
                    "image": f"images/taskD/{img_rel_path}",
                    "prompt": TASK_D_PROMPT,
                    "answer": "Success" if success else "Fail",
                    "label": int(success),
                    "n_frames": layout["n_frames"],
                    "layout": layout,
                    "source_group_id": gid,
                    "source_variant": variant,
                    "source_env": env,
                }
                rows.append(row)
                stats["envs"][env] += 1
                stats["made"] += 1

    return rows, stats


def verify_rows(rows: List[dict], out_dir: Path) -> Dict[str, Any]:
    errors: List[str] = []
    uids = [row["uid"] for row in rows]
    if len(uids) != len(set(uids)):
        errors.append("Duplicate UIDs detected")
    for row in rows:
        img = out_dir / row["image"]
        if not img.exists():
            errors.append(f"Missing image for {row['uid']}: {img}")

    env_counts = Counter(row["uid"].split(".")[1] for row in rows)
    variant_counts = Counter(row["variant"] for row in rows)
    answer_counts = Counter(row["answer"] for row in rows)
    return {
        "total_rows": len(rows),
        "uid_unique": len(uids) == len(set(uids)),
        "all_images_exist": not errors,
        "errors": errors,
        "env_counts": dict(env_counts),
        "variant_counts": dict(variant_counts),
        "answer_counts": dict(answer_counts),
    }


def write_qc_report(out_dir: Path, stats: Dict[str, Any], report: Dict[str, Any]) -> None:
    lines = [
        "# Task D v3 144-row Candidate — QC Report",
        "",
        f"- Output: `{out_dir}`",
        f"- Total rows: {report['total_rows']}",
        f"- UID unique: {report['uid_unique']}",
        f"- All images exist: {report['all_images_exist']}",
        "",
        "## Env Counts",
        "",
        "| Env | Rows | Selected groups |",
        "|---|---:|---|",
    ]
    for env in DEFAULT_ENVS:
        lines.append(
            f"| {env} | {report['env_counts'].get(env, 0)} | "
            f"{', '.join(stats['selected_groups'].get(env, []))} |"
        )

    lines.extend(
        [
            "",
            "## Variant Counts",
            "",
            "| Variant | Rows |",
            "|---|---:|",
        ]
    )
    for variant, count in sorted(report["variant_counts"].items()):
        lines.append(f"| {variant} | {count} |")

    lines.extend(
        [
            "",
            "## Answer Counts",
            "",
            "| Answer | Rows |",
            "|---|---:|",
        ]
    )
    for answer, count in sorted(report["answer_counts"].items()):
        lines.append(f"| {answer} | {count} |")

    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        for err in report["errors"]:
            lines.append(f"- {err}")

    (out_dir / "qc_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task D exam")
    parser.add_argument(
        "--raw-root",
        type=str,
        default=str(Path("scripts/outputs/raw_data")),
        help="Raw data root with triplets.jsonl per env (default: scripts/outputs/raw_data)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("outputs/exams_taskd")),
        help="Output directory (default: outputs/exams_taskd)",
    )
    parser.add_argument(
        "--groups-per-env",
        type=int,
        default=8,
        help="Number of groups selected per environment",
    )
    parser.add_argument(
        "--layout",
        choices=["fixed-5", "adaptive"],
        default=DEFAULT_LAYOUT_MODE,
        help="Montage layout mode",
    )
    parser.add_argument("--min-cols", type=int, default=DEFAULT_MIN_COLS)
    parser.add_argument("--max-cols", type=int, default=DEFAULT_MAX_COLS)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    rows, stats = build_task_d_rows(
        raw_root=raw_root,
        out_dir=out_dir,
        envs=DEFAULT_ENVS,
        groups_per_env=args.groups_per_env,
        layout_mode=args.layout,
        min_cols=args.min_cols,
        max_cols=args.max_cols,
    )
    report = verify_rows(rows, out_dir)
    write_jsonl(out_dir / "task_d_exam.jsonl", rows)
    write_qc_report(out_dir, stats, report)

    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        pass

    manifest = {
        "schema_version": EXAM_SCHEMA_VERSION,
        "version": "v3_144_candidate",
        "generator": "scripts/generate_task_d_v3_144.py",
        "git_commit": git_hash,
        "source_raw_root": str(raw_root),
        "groups_per_env": args.groups_per_env,
        "variants": DEFAULT_VARIANTS,
        "prompt_contract": "task_d_minimal_outcome_judgment",
        "layout": {
            "mode": args.layout,
            "min_cols": args.min_cols,
            "max_cols": args.max_cols,
        },
        "outputs": {
            "task_d_exam": len(rows),
            "images": sum(1 for _ in (out_dir / "images" / "taskD").rglob("*.png")),
        },
        "stats": stats,
        "qc": report,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({
        "out_dir": str(out_dir),
        "rows": len(rows),
        "env_counts": report["env_counts"],
        "variant_counts": report["variant_counts"],
        "answer_counts": report["answer_counts"],
        "uid_unique": report["uid_unique"],
        "all_images_exist": report["all_images_exist"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
