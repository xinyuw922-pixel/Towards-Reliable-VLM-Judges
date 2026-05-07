#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


FONT_PATHS = [
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]

PAGE_W = 2000
PAGE_H = 2828
MARGIN = 90

FAMILY_NAMES = {
    "doorkey": "DoorKey",
    "multiroom": "MultiRoom",
    "redblue": "RedBlue",
}

DIFFICULTY_NAMES = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}

NOTE_DESCRIPTIONS = {
    "migration_doorkey_canonical": "Rules unchanged: pick up key, open yellow door, reach red goal.",
    "migration_doorkey_medium": "Rules unchanged: goal is farther, with light visual distractions.",
    "migration_doorkey_hard": "Rules unchanged: longer event span, farther post-door path, more distractions.",
    "migration_multiroom_canonical": "Rules unchanged: pass through two sequential doors, reach goal.",
    "migration_multiroom_medium": "Rules unchanged: longer corridors, larger interval between two door opens.",
    "migration_multiroom_hard": "Rules unchanged: longer chain, farther goal, additional distractions.",
    "migration_redblue_canonical": "Rules unchanged: must open red door first, then blue door.",
    "migration_redblue_medium": "Rules unchanged: larger interval between red and blue doors.",
    "migration_redblue_hard": "Rules unchanged: same sequence constraint, longer duration, more distractions.",
}

INTERVENTION_NAMES = {
    "remove_key": "Remove key",
    "lock_gate2_forever": "Second door permanently locked",
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def create_blank_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    return page, ImageDraw.Draw(page)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        buf = ""
        for ch in para:
            test = buf + ch
            if draw.textlength(test, font=font) <= max_width:
                buf = test
            else:
                if buf:
                    lines.append(buf)
                buf = ch
        if buf:
            lines.append(buf)
    return lines


def add_wrapped_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    line_gap: int,
    *,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> int:
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_gap
    return y


def fit_contain(img: Image.Image, target_w: int, target_h: int, *, bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    scale = min(target_w / img.width, target_h / img.height, 1.0)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def load_summary(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    difficulty_rank = {"easy": 0, "medium": 1, "hard": 2}
    rows.sort(key=lambda r: (r["family"], difficulty_rank.get(r["layout_metadata"].get("difficulty", ""), 9)))
    return rows


def get_family_name(family: str) -> str:
    return FAMILY_NAMES.get(family, family)


def get_difficulty_name(difficulty: str) -> str:
    return DIFFICULTY_NAMES.get(difficulty, difficulty)


def get_note_description(row: dict[str, Any]) -> str:
    return NOTE_DESCRIPTIONS.get(row["template_id"], row.get("note", ""))


def build_cover(rows: list[dict[str, Any]], mode: str, summary_path: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(72)
    sub_font = load_font(42)
    body_font = load_font(34)

    draw.text((MARGIN, 120), "MiniWorld Task D Migration Environment Exam Pack", font=title_font, fill=(0, 0, 0))
    mode_label = "Blind Review Version" if mode == "blind" else "Answer Key Version"
    draw.text((MARGIN, 230), mode_label, font=sub_font, fill=(0, 0, 0))

    y = 340
    lines = [
        f"Data index: {summary_path}",
        f"Total items: {len(rows)}",
        "",
        "This exam pack is organized by three migration environments from MiniGrid to MiniWorld:",
        "DoorKey, MultiRoom, RedBlue; each environment includes Easy / Medium / Hard templates.",
        "",
        "Blind review version:",
        "Each page shows only one full storyboard trajectory with question text, without answers.",
        "",
        "Answer key version:",
        "Each page shows full storyboard, cf storyboard, event coverage visualization and key metadata.",
    ]
    y = add_wrapped_text(draw, MARGIN, y, "\n".join(lines), body_font, PAGE_W - 2 * MARGIN, 48)

    y += 30
    for family in ("doorkey", "multiroom", "redblue"):
        family_rows = [r for r in rows if r["family"] == family]
        templates = ", ".join(
            f"{get_difficulty_name(str(r['layout_metadata'].get('difficulty', '')))} = {r['template_id']}"
            for r in family_rows
        )
        y = add_wrapped_text(draw, MARGIN, y, f"{get_family_name(family)}: {templates}", body_font, PAGE_W - 2 * MARGIN, 44)
        y += 12

    return page


def build_prompt_block(row: dict[str, Any], *, blind: bool) -> str:
    family = row["family"]
    difficulty = row["layout_metadata"].get("difficulty", "unknown")
    note = get_note_description(row)
    seq_rel = row["manifest_relpath"]
    prompt = [
        f"Environment family: {get_family_name(family)}",
        f"Difficulty: {get_difficulty_name(str(difficulty))}",
        f"Template name: {row['template_id']}",
        f"Description: {note}",
        "",
        "Question:",
        "Determine whether this storyboard trajectory successfully completes the task.",
        "Answer requirement: Output only one English word: SUCCESS or FAIL.",
    ]
    if not blind:
        prompt.extend(
            [
                "",
                f"full success = {row['full_success']}",
                f"cf success = {row['cf_success']}",
                f"cf intervention = {INTERVENTION_NAMES.get(row['cf_intervention'], row['cf_intervention'])}",
                f"full event steps = {row['full_event_steps']}",
                f"cf event steps = {row['cf_event_steps']}",
                f"manifest path = {seq_rel}",
            ]
        )
    return "\n".join(prompt)


def build_row_page(row: dict[str, Any], *, root: Path, mode: str) -> Image.Image:
    blind = mode == "blind"
    page, draw = create_blank_page()
    title_font = load_font(52)
    body_font = load_font(32)
    small_font = load_font(26)

    family = row["family"]
    difficulty = row["layout_metadata"].get("difficulty", "unknown")
    title = f"{get_family_name(family)} / {get_difficulty_name(str(difficulty))} / {row['template_id']}"
    draw.text((MARGIN, 70), title, font=title_font, fill=(0, 0, 0))

    prompt_x = MARGIN
    prompt_y = 155
    prompt_w = 620
    prompt_h = PAGE_H - 2 * MARGIN - 40
    draw.rounded_rectangle((prompt_x, prompt_y, prompt_x + prompt_w, prompt_y + prompt_h), radius=22, outline=(0, 0, 0), width=3)
    text = build_prompt_block(row, blind=blind)
    add_wrapped_text(draw, prompt_x + 24, prompt_y + 24, text, body_font, prompt_w - 48, 42)

    right_x = prompt_x + prompt_w + 40
    right_w = PAGE_W - right_x - MARGIN

    full_img = Image.open(root / row["full_storyboard_relpath"]).convert("RGB")
    full_target_h = 1040 if blind else 760
    full_fit = fit_contain(full_img, right_w, full_target_h)
    page.paste(full_fit, (right_x, 150))
    draw.text((right_x, 118), "Full Trajectory", font=body_font, fill=(0, 0, 0))

    if blind:
        meta = f"Family={get_family_name(family)} | Difficulty={get_difficulty_name(str(difficulty))} | Answer hidden"
        draw.text((right_x, 1220), meta, font=small_font, fill=(70, 70, 70))
        return page

    cf_img = Image.open(root / row["cf_storyboard_relpath"]).convert("RGB")
    cf_fit = fit_contain(cf_img, right_w // 2 - 20, 620)
    event_cov_path = row.get("event_coverage_relpath")
    event_fit = None
    if event_cov_path:
        event_img = Image.open(root / event_cov_path).convert("RGB")
        event_fit = fit_contain(event_img, right_w // 2 - 20, 620)
    page.paste(cf_fit, (right_x, 980))
    draw.text((right_x, 948), "CF Trajectory", font=body_font, fill=(0, 0, 0))
    if event_fit is not None:
        x2 = right_x + right_w // 2 + 20
        page.paste(event_fit, (x2, 980))
        draw.text((x2, 948), "Event Coverage Map", font=body_font, fill=(0, 0, 0))

    meta_lines = [
        f"Pair valid = {row['pair_ok']}",
        f"CF intervention = {INTERVENTION_NAMES.get(row['cf_intervention'], row['cf_intervention'])}",
        f"Full event steps = {row['full_event_steps']}",
        f"CF event steps = {row['cf_event_steps']}",
    ]
    add_wrapped_text(draw, right_x, 1660, "\n".join(meta_lines), small_font, right_w, 36, fill=(70, 70, 70))
    return page


def save_pdf(pages: list[Image.Image], out_pdf: Path) -> None:
    rgb_pages = [p.convert("RGB") for p in pages]
    rgb_pages[0].save(out_pdf, save_all=True, append_images=rgb_pages[1:], resolution=150.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render MiniWorld migration family showcase PDFs")
    ap.add_argument(
        "--summary_json",
        default="tmp_miniworld/taskD-MW-migration-showcase-families-paired/migration_showcase_summary.json",
    )
    ap.add_argument(
        "--mode",
        choices=["blind", "answer-key"],
        default="answer-key",
    )
    ap.add_argument("--out_pdf", default=None)
    args = ap.parse_args()

    summary_path = Path(args.summary_json)
    root = summary_path.parent
    rows = load_summary(summary_path)

    pages = [build_cover(rows, args.mode, summary_path)]
    for row in rows:
        pages.append(build_row_page(row, root=root, mode=args.mode))

    if args.out_pdf is None:
        suffix = "blind" if args.mode == "blind" else "answer_key"
        out_pdf = root / f"taskD_MW_migration_showcase_{suffix}.pdf"
    else:
        out_pdf = Path(args.out_pdf)
    save_pdf(pages, out_pdf)
    print(f"Saved PDF to {out_pdf}")


if __name__ == "__main__":
    main()
