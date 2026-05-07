#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SUMMARY = ROOT / "tmp_miniworld" / "miniworld-tier1-paper-pack" / "miniworld_tier1_results_summary.json"
DEFAULT_OUT = ROOT / "tmp_miniworld" / "miniworld-tier1-paper-pack" / "miniworld_tier1_review.pdf"
DEFAULT_FIG_DIR = ROOT / "tmp_miniworld" / "miniworld-tier1-paper-pack" / "figures"
FONT_PATHS = [
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
]


def pick_font() -> FontProperties | None:
    for path in FONT_PATHS:
        if path.exists():
            return FontProperties(fname=str(path))
    return None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def acc(block: dict) -> float:
    n = int(block["n"])
    correct = int(block["correct"])
    return (correct / n) if n else 0.0


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def add_cover(pdf: PdfPages) -> None:
    font = pick_font()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.88, "MiniWorld Tier-1 Paper Pack Review", fontsize=26, fontweight="bold", fontproperties=font)
    ax.text(0.06, 0.82, "Dual-Model Results Review Pack", fontsize=18, fontproperties=font)
    ax.text(
        0.06,
        0.70,
        "Contents:\n"
        "1. Dual-model overview table\n"
        "2. A / D / D-framing / C / C-framing breakdown tables\n"
        "3. 6 dual-model comparison figures\n"
        "4. Paper results summary",
        fontsize=16,
        fontproperties=font,
        linespacing=1.8,
    )
    ax.text(
        0.06,
        0.28,
        "Models: GPT-5.4 vs Gemini-2.5-Flash\n"
        "Purpose: Paper table draft and chart review",
        fontsize=14,
        fontproperties=font,
        color="#444444",
        linespacing=1.7,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, columns: list[str], rows: list[list[str]], note: str | None = None) -> None:
    font = pick_font()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.84])
    ax.axis("off")
    ax.text(0.0, 1.05, title, fontsize=20, fontweight="bold", transform=ax.transAxes, fontproperties=font)

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.22 if note else 0.08, 1.0, 0.68 if note else 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        if font is not None:
            cell.get_text().set_fontproperties(font)
        if r == 0:
            cell.set_facecolor("#EAF1FB")
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#F8FAFD")
        else:
            cell.set_facecolor("#FFFFFF")

    if note:
        ax.text(
            0.0,
            0.04,
            note,
            fontsize=12,
            fontproperties=font,
            color="#444444",
            transform=ax.transAxes,
            va="bottom",
            linespacing=1.5,
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_figure_page(pdf: PdfPages, title: str, image_path: Path, note: str | None = None) -> None:
    font = pick_font()
    img = mpimg.imread(image_path)
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.78])
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.04, 0.94, title, fontsize=20, fontweight="bold", fontproperties=font)
    if note:
        fig.text(0.04, 0.055, note, fontsize=12, color="#444444", fontproperties=font)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    font = pick_font()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.05, 0.06, 0.9, 0.88])
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=20, fontweight="bold", va="top", transform=ax.transAxes, fontproperties=font)
    y = 0.90
    for para in paragraphs:
        wrapped = "\n".join(textwrap.wrap(para, width=86))
        ax.text(0.0, y, wrapped, fontsize=13, va="top", linespacing=1.7, transform=ax.transAxes, fontproperties=font)
        y -= 0.18
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a review PDF for the MiniWorld Tier-1 paper pack.")
    ap.add_argument("--summary_json", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--fig_dir", default=str(DEFAULT_FIG_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    summary = load_json(Path(args.summary_json))
    fig_dir = Path(args.fig_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    line_map = {line["line_id"]: line for line in summary["lines"]}
    a = line_map["A-MW-v2-crossenv"]
    d = line_map["D-MW-reference-family-exam50"]
    d_framing = line_map["D-MW-framing-full"]
    c = line_map["C-MW-formal"]
    c_framing = line_map["C-MW-framing-full"]

    def run(line: dict, label: str) -> dict:
        return next(x for x in line["model_runs"] if x["model_label"] == label)

    a_gpt, a_gem = run(a, "GPT-5.4"), run(a, "Gemini-2.5-Flash")
    d_gpt, d_gem = run(d, "GPT-5.4"), run(d, "Gemini-2.5-Flash")
    df_gpt, df_gem = run(d_framing, "GPT-5.4"), run(d_framing, "Gemini-2.5-Flash")
    c_gpt, c_gem = run(c, "GPT-5.4"), run(c, "Gemini-2.5-Flash")
    cf_gpt, cf_gem = run(c_framing, "GPT-5.4"), run(c_framing, "Gemini-2.5-Flash")

    with PdfPages(out) as pdf:
        add_cover(pdf)
        add_table_page(
            pdf,
            "Table 1. MiniWorld Tier-1 Dual-Model Overview",
            ["Line", "Task type", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["A-MW-v2-crossenv", "successor prediction", str(a_gpt["overall"]["n"]), pct(a_gpt["overall"]["accuracy"]), pct(a_gem["overall"]["accuracy"])],
                ["D-MW-reference-family-exam50", "trajectory outcome judgment", str(d_gpt["overall"]["n"]), pct(d_gpt["overall"]["accuracy"]), pct(d_gem["overall"]["accuracy"])],
                ["D-MW-framing-full", "prompt framing probe", str(df_gpt["overall"]["n"]), pct(df_gpt["overall"]["accuracy"]), pct(df_gem["overall"]["accuracy"])],
                ["C-MW-formal", "sparse keyframe judgment", str(c_gpt["overall"]["n"]), pct(c_gpt["overall"]["accuracy"]), pct(c_gem["overall"]["accuracy"])],
                ["C-MW-framing-full", "prompt framing probe", str(cf_gpt["overall"]["n"]), pct(cf_gpt["overall"]["accuracy"]), pct(cf_gem["overall"]["accuracy"])],
            ],
            "This page is good for overall headlines: A and D are main results; D-framing and C-framing are used to check if wording sensitivity replicates across tasks.",
        )
        add_table_page(
            pdf,
            "Table 2. A-MW-v2-crossenv by Environment",
            ["Environment", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["fourrooms", str(a_gpt["breakdowns"]["by_env"]["fourrooms"]["n"]), pct(a_gpt["breakdowns"]["by_env"]["fourrooms"]["accuracy"]), pct(a_gem["breakdowns"]["by_env"]["fourrooms"]["accuracy"])],
                ["pickupobjects", str(a_gpt["breakdowns"]["by_env"]["pickupobjects"]["n"]), pct(a_gpt["breakdowns"]["by_env"]["pickupobjects"]["accuracy"]), pct(a_gem["breakdowns"]["by_env"]["pickupobjects"]["accuracy"])],
                ["putnext", str(a_gpt["breakdowns"]["by_env"]["putnext"]["n"]), pct(a_gpt["breakdowns"]["by_env"]["putnext"]["accuracy"]), pct(a_gem["breakdowns"]["by_env"]["putnext"]["accuracy"])],
            ],
            "Gemini is significantly weaker on FourRooms, but on par with GPT-5.4 on PickupObjects and PutNext.",
        )
        add_table_page(
            pdf,
            "Table 3. D-MW by Family",
            ["Family", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["doorkey", str(d_gpt["breakdowns"]["by_family"]["doorkey"]["n"]), pct(acc(d_gpt["breakdowns"]["by_family"]["doorkey"])), pct(acc(d_gem["breakdowns"]["by_family"]["doorkey"]))],
                ["multiroom", str(d_gpt["breakdowns"]["by_family"]["multiroom"]["n"]), pct(acc(d_gpt["breakdowns"]["by_family"]["multiroom"])), pct(acc(d_gem["breakdowns"]["by_family"]["multiroom"]))],
                ["redblue", str(d_gpt["breakdowns"]["by_family"]["redblue"]["n"]), pct(acc(d_gpt["breakdowns"]["by_family"]["redblue"])), pct(acc(d_gem["breakdowns"]["by_family"]["redblue"]))],
            ],
            "This page best shows family-level structural differences: GPT-5.4 is stronger on DoorKey, while Gemini is actually stronger on RedBlue.",
        )
        add_table_page(
            pdf,
            "Table 4. D-MW Framing Sensitivity",
            ["Framing", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["pos", str(df_gpt["breakdowns"]["by_framing"]["pos"]["n"]), pct(acc(df_gpt["breakdowns"]["by_framing"]["pos"])), pct(acc(df_gem["breakdowns"]["by_framing"]["pos"]))],
                ["neu", str(df_gpt["breakdowns"]["by_framing"]["neu"]["n"]), pct(acc(df_gpt["breakdowns"]["by_framing"]["neu"])), pct(acc(df_gem["breakdowns"]["by_framing"]["neu"]))],
                ["neg", str(df_gpt["breakdowns"]["by_framing"]["neg"]["n"]), pct(acc(df_gpt["breakdowns"]["by_framing"]["neg"])), pct(acc(df_gem["breakdowns"]["by_framing"]["neg"]))],
            ],
            "If the same direction framing effect appears on the D track, it shows that wording sensitivity is not unique to Task C.",
        )
        add_table_page(
            pdf,
            "Table 5. C-MW-formal by Variant",
            ["Variant", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["full", str(c_gpt["breakdowns"]["by_variant"]["full"]["n"]), pct(acc(c_gpt["breakdowns"]["by_variant"]["full"])), pct(acc(c_gem["breakdowns"]["by_variant"]["full"]))],
                ["nocue", str(c_gpt["breakdowns"]["by_variant"]["nocue"]["n"]), pct(acc(c_gpt["breakdowns"]["by_variant"]["nocue"])), pct(acc(c_gem["breakdowns"]["by_variant"]["nocue"]))],
                ["cf", str(c_gpt["breakdowns"]["by_variant"]["cf"]["n"]), pct(acc(c_gpt["breakdowns"]["by_variant"]["cf"])), pct(acc(c_gem["breakdowns"]["by_variant"]["cf"]))],
            ],
            "GPT-5.4 clearly biases toward failure judgments, while Gemini is more balanced on full/nocue.",
        )
        add_table_page(
            pdf,
            "Table 6. C-MW Framing Sensitivity",
            ["Framing", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["pos", str(cf_gpt["breakdowns"]["by_framing"]["pos"]["n"]), pct(acc(cf_gpt["breakdowns"]["by_framing"]["pos"])), pct(acc(cf_gem["breakdowns"]["by_framing"]["pos"]))],
                ["neu", str(cf_gpt["breakdowns"]["by_framing"]["neu"]["n"]), pct(acc(cf_gpt["breakdowns"]["by_framing"]["neu"])), pct(acc(cf_gem["breakdowns"]["by_framing"]["neu"]))],
                ["neg", str(cf_gpt["breakdowns"]["by_framing"]["neg"]["n"]), pct(acc(cf_gpt["breakdowns"]["by_framing"]["neg"])), pct(acc(cf_gem["breakdowns"]["by_framing"]["neg"]))],
            ],
            "Both models preserve the same-direction framing effect: higher for positive, lower for negative.",
        )

        add_figure_page(pdf, "Figure 1. MiniWorld Tier-1 Dual-Model Overview", fig_dir / "figure_mw_tier1_dual_model_overview.png")
        add_figure_page(pdf, "Figure 2. A-MW-v2-crossenv", fig_dir / "figure_a_mw_crossenv_accuracy.png")
        add_figure_page(pdf, "Figure 3. D-MW Family Comparison", fig_dir / "figure_d_mw_family_accuracy.png")
        add_figure_page(pdf, "Figure 4. D-MW Framing Comparison", fig_dir / "figure_d_mw_framing_accuracy.png")
        add_figure_page(pdf, "Figure 5. C-MW-formal Variant Comparison", fig_dir / "figure_c_mw_variant_accuracy.png")
        add_figure_page(pdf, "Figure 6. C-MW Framing Comparison", fig_dir / "figure_c_mw_framing_accuracy.png")

        add_text_page(
            pdf,
            "Paper-Facing Result Summary",
            [
                "The dual-model version is more convincing than the previous single-model chart because it no longer just shows one number, but shows which patterns replicate across models and which have clear model dependency.",
                "The most important information from A-MW-v2-crossenv is not the 60.0% single point, but the significant divergence between GPT-5.4 and Gemini-2.5-Flash on FourRooms, while they realign on PickupObjects and PutNext. This transforms the A track from 'a regular bar chart' into a more explanatory cross-env comparison.",
                "D-MW-reference-family-exam50 remains the strongest MiniWorld headline. Both models score similarly overall, but with different family structures: GPT-5.4 is stronger on DoorKey, Gemini stronger on RedBlue. This difference makes the family-level chart more suitable for the paper.",
                "The new D-MW full framing results can answer a stronger question: whether prompt wording effects are unique to sparse keyframe tasks like C. If the same directional change appears on D, this conclusion becomes more convincing.",
                "C-MW-formal and C-MW full framing together form the analysis evidence. C-formal shows that sparse-keyframe judgment error patterns are not uniform, while full framing further shows that prompt wording directional effects replicate across both models and the full item bank.",
            ],
        )

    print(f"Rendered review PDF -> {out}")


if __name__ == "__main__":
    main()
