#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image

import sys as _sys
_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent.parent.parent
# Add repo root to path
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from config import REPO_ROOT, MINIGRID_ROOT

ROOT = REPO_ROOT
DEFAULT_CANONICAL_ROOT = MINIGRID_ROOT

TASK_C_EXAM = "task_c_exam.jsonl"
TASK_E_EXAM = "task_e_exam_v13.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def slugify_uid(uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", uid)


def compute_layout(n_frames: int) -> dict[str, int]:
    if n_frames <= 1:
        return {"rows": 1, "cols": 1}

    best_r, best_c = 1, n_frames
    best_score = float("inf")
    for r in range(1, n_frames + 1):
        c = (n_frames + r - 1) // r
        cells = r * c
        n_empty = cells - n_frames
        aspect_ratio = c / r if r > 0 else float("inf")
        aspect_penalty = abs(math.log(aspect_ratio)) if aspect_ratio > 0 else float("inf")
        score = aspect_penalty * 2 + n_empty
        if score < best_score:
            best_score = score
            best_r, best_c = r, c

    if best_c < best_r:
        best_r, best_c = best_c, best_r
    return {"rows": best_r, "cols": best_c}


def selected_frame_count(row: dict[str, Any], task: str) -> int:
    if task == "C":
        if row.get("n_frames_selected") is not None:
            return int(row["n_frames_selected"])
    selected_steps = row.get("selected_steps") or []
    return len(selected_steps)


def split_layout(row: dict[str, Any], task: str) -> dict[str, int]:
    if task == "C":
        layout = row.get("layout") or {}
        rows = int(layout.get("rows", 0))
        cols = int(layout.get("cols", 0))
        if rows > 0 and cols > 0:
            return {"rows": rows, "cols": cols}
    return compute_layout(selected_frame_count(row, task))


def crop_storyboard_tiles(
    row: dict[str, Any],
    task: str,
    exam_root: Path,
    assets_root: Path,
) -> list[str]:
    rel = str(row["image"])
    src = exam_root / rel
    if not src.exists():
        raise FileNotFoundError(f"Missing storyboard image for {row['uid']}: {src}")

    n_frames = selected_frame_count(row, task)
    layout = split_layout(row, task)
    rows = layout["rows"]
    cols = layout["cols"]
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid layout for {row['uid']}: {layout}")

    uid_dir = assets_root / task.lower() / slugify_uid(str(row["uid"]))
    uid_dir.mkdir(parents=True, exist_ok=True)

    out_images: list[str] = []
    with Image.open(src) as im0:
        im = im0.convert("RGB")
        width, height = im.size
        cell_w = width // cols
        cell_h = height // rows
        if cell_w <= 0 or cell_h <= 0:
            raise ValueError(f"Degenerate tile size for {row['uid']}: {(width, height)} / {layout}")

        for idx in range(n_frames):
            x = (idx % cols) * cell_w
            y = (idx // cols) * cell_h
            crop = im.crop((x, y, x + cell_w, y + cell_h))
            out_path = uid_dir / f"frame_{idx:02d}.png"
            crop.save(out_path)
            out_images.append(str(out_path))

    return out_images


def normalize_task_e_prompt(prompt: str) -> str:
    """
    Normalize TaskE answer JSON in prompts by fixing double-brace Python literal artifacts.

    Exact replacements:
      '{{"answer":'  -> '{"answer":'
      '>"}}'          -> '>"}'   (only the trailing '>"}}' that closes the double-brace)

    The canonical exam has status prompts ending with:
      ...{"answer":"<one label from holding, dropped, reacquired_and_holding>"}}
    which is the double-brace artifact. Identity prompts end with:
      ...{"answer":"<one label from key_yellow, key_green, ..."}}
    but those don't have a closing > before the }} so only status prompts are affected.
    """
    p = prompt
    p = p.replace('{{"answer":', '{"answer":')
    p = p.replace('>"}}', '>"}')
    return p


def adapt_prompt(prompt: str, task: str) -> str:
    lines = prompt.splitlines()
    if task == "C":
        first_line = (
            "Below are separate images from one MiniGrid trajectory, shown in chronological order from first to last."
        )
    else:
        first_line = (
            "Below are separate images from a MiniGrid trajectory prefix, shown in chronological order from first to last. "
            "The sequence ends at the last shown frame only."
        )

    if lines and lines[0].strip().lower().startswith("below is"):
        lines[0] = first_line
    else:
        lines = [first_line, ""] + lines
    return "\n".join(lines).strip()


def make_exam_stub(row: dict[str, Any], task: str) -> dict[str, Any]:
    return {
        "uid": row["uid"],
        "exam_id": row.get("exam_id", row["uid"]),
        "task": task,
        "env_task": row.get("env_task"),
        "group_id": row.get("group_id"),
        "schema_version": row.get("schema_version", "gridwm.exam.v1"),
    }


def build_requests(
    rows: list[dict[str, Any]],
    task: str,
    exam_root: Path,
    assets_root: Path,
    api_content_order: str,
    api_image_detail: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    composite: list[dict[str, Any]] = []
    split: list[dict[str, Any]] = []
    split_frame_counts: list[int] = []

    for row in rows:
        uid = str(row["uid"])
        prompt = str(row["prompt"]).strip()
        # Normalize TaskE double-brace artifacts from canonical exam
        if task == "E":
            prompt = normalize_task_e_prompt(prompt)
        composite_req: dict[str, Any] = {
            "uid": uid,
            "exam_id": row.get("exam_id", uid),
            "schema_version": row.get("schema_version", "gridwm.exam.v1"),
            "images": [str(exam_root / str(row["image"]))],
            "prompt": prompt,
            "exam": make_exam_stub(row, task),
        }
        if api_content_order:
            composite_req["api_content_order"] = api_content_order
        if api_image_detail:
            composite_req["api_image_detail"] = api_image_detail
        composite.append(composite_req)

        split_images = crop_storyboard_tiles(row, task, exam_root, assets_root)
        split_req: dict[str, Any] = {
            "uid": uid,
            "exam_id": row.get("exam_id", uid),
            "schema_version": row.get("schema_version", "gridwm.exam.v1"),
            "images": split_images,
            "prompt": adapt_prompt(prompt, task),
            "exam": make_exam_stub(row, task),
        }
        if api_content_order:
            split_req["api_content_order"] = api_content_order
        if api_image_detail:
            split_req["api_image_detail"] = api_image_detail
        split.append(split_req)
        split_frame_counts.append(len(split_images))

    telemetry = {
        "n_rows": len(rows),
        "min_images_per_split_request": min(split_frame_counts) if split_frame_counts else 0,
        "max_images_per_split_request": max(split_frame_counts) if split_frame_counts else 0,
    }
    return composite, split, telemetry


def choose_rows(rows: list[dict[str, Any]], num: int, ordered: bool, seed: int) -> list[dict[str, Any]]:
    if num <= 0 or num >= len(rows):
        return list(rows)
    if ordered:
        return rows[:num]

    by_env: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        env = str(row.get("env_task") or "unknown")
        by_env.setdefault(env, []).append(row)

    rng = random.Random(seed)
    env_names = sorted(by_env)
    for env in env_names:
        rng.shuffle(by_env[env])

    chosen: list[dict[str, Any]] = []
    while len(chosen) < num:
        progressed = False
        for env in env_names:
            bucket = by_env[env]
            if bucket and len(chosen) < num:
                chosen.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    return chosen


def export_task_bundle(
    task: str,
    exam_root: Path,
    exam_file: str,
    out_root: Path,
    num: int,
    ordered: bool,
    seed: int,
    api_content_order: str,
    api_image_detail: str,
) -> dict[str, Any]:
    exam_path = exam_root / exam_file
    rows = load_jsonl(exam_path)
    chosen = choose_rows(rows, num=num, ordered=ordered, seed=seed)

    task_dir = out_root / task.lower()
    composite_dir = task_dir / "composite"
    split_dir = task_dir / "split"
    assets_dir = task_dir / "split_assets"

    write_jsonl(composite_dir / exam_file, chosen)
    write_jsonl(split_dir / exam_file, chosen)

    composite_requests, split_requests, telemetry = build_requests(
        rows=chosen,
        task=task,
        exam_root=exam_root,
        assets_root=assets_dir,
        api_content_order=api_content_order,
        api_image_detail=api_image_detail,
    )

    write_jsonl(composite_dir / f"requests_task{task}_composite.jsonl", composite_requests)
    write_jsonl(split_dir / f"requests_task{task}_split.jsonl", split_requests)

    manifest = {
        "task": task,
        "exam_root": str(exam_root),
        "exam_file": exam_file,
        "n_total": len(rows),
        "n_selected": len(chosen),
        "uids": [row["uid"] for row in chosen],
        "ordered": ordered,
        "seed": seed,
        "api_content_order": api_content_order or None,
        "api_image_detail": api_image_detail or None,
        "composite_exam": str((composite_dir / exam_file).relative_to(out_root)),
        "split_exam": str((split_dir / exam_file).relative_to(out_root)),
        "composite_requests": str((composite_dir / f"requests_task{task}_composite.jsonl").relative_to(out_root)),
        "split_requests": str((split_dir / f"requests_task{task}_split.jsonl").relative_to(out_root)),
        "split_assets_root": str(assets_dir.relative_to(out_root)),
        "telemetry": telemetry,
    }
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build paired composite-vs-split smoke requests for Task C and Task E.")
    ap.add_argument("--canonical_root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--tasks",
        nargs="+",
        choices=["C", "E"],
        default=["C", "E"],
        help="Which tasks to export. Default: both C and E.",
    )
    ap.add_argument("--num_c", type=int, default=24, help="Number of Task C rows to export. Use 0 for full export.")
    ap.add_argument("--num_e", type=int, default=24, help="Number of Task E rows to export. Use 0 for full export.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ordered", action="store_true", help="Preserve exam order instead of random sampling.")
    ap.add_argument("--api_content_order", default="text_first")
    ap.add_argument("--api_image_detail", default="auto")
    args = ap.parse_args()

    canonical_root = args.canonical_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, dict[str, Any]] = {}
    tasks = set(args.tasks)
    if "C" in tasks:
        manifests["taskC"] = export_task_bundle(
            task="C",
            exam_root=canonical_root / "taskc",
            exam_file=TASK_C_EXAM,
            out_root=out_dir,
            num=args.num_c,
            ordered=args.ordered,
            seed=args.seed,
            api_content_order=args.api_content_order,
            api_image_detail=args.api_image_detail,
        )
    if "E" in tasks:
        manifests["taskE"] = export_task_bundle(
            task="E",
            exam_root=canonical_root / "taske",
            exam_file=TASK_E_EXAM,
            out_root=out_dir,
            num=args.num_e,
            ordered=args.ordered,
            seed=args.seed,
            api_content_order=args.api_content_order,
            api_image_detail=args.api_image_detail,
        )

    manifest = {
        "canonical_root": str(canonical_root),
        "out_dir": str(out_dir),
        "tasks_selected": sorted(tasks),
        "ordered": args.ordered,
        "seed": args.seed,
        "api_content_order": args.api_content_order or None,
        "api_image_detail": args.api_image_detail or None,
        "tasks": manifests,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
