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
DEFAULT_OUT = ROOT / "tmp_miniworld" / "miniworld-tier1-paper-pack" / "miniworld_tier1_review_zh.pdf"
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
    ax.text(0.06, 0.82, "双模型结果审阅包", fontsize=18, fontproperties=font)
    ax.text(
        0.06,
        0.70,
        "内容包含：\n"
        "1. 双模型总览表\n"
        "2. A / D / D-framing / C / C-framing 分表\n"
        "3. 6 张双模型对比图\n"
        "4. 一页论文结果总结",
        fontsize=16,
        fontproperties=font,
        linespacing=1.8,
    )
    ax.text(
        0.06,
        0.28,
        "模型：GPT-5.4 vs Gemini-2.5-Flash\n"
        "用途：论文表格草稿与图表观感审阅",
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
            "Table 1. MiniWorld Tier-1 双模型总表",
            ["Line", "Task type", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["A-MW-v2-crossenv", "successor prediction", str(a_gpt["overall"]["n"]), pct(a_gpt["overall"]["accuracy"]), pct(a_gem["overall"]["accuracy"])],
                ["D-MW-reference-family-exam50", "trajectory outcome judgment", str(d_gpt["overall"]["n"]), pct(d_gpt["overall"]["accuracy"]), pct(d_gem["overall"]["accuracy"])],
                ["D-MW-framing-full", "prompt framing probe", str(df_gpt["overall"]["n"]), pct(df_gpt["overall"]["accuracy"]), pct(df_gem["overall"]["accuracy"])],
                ["C-MW-formal", "sparse keyframe judgment", str(c_gpt["overall"]["n"]), pct(c_gpt["overall"]["accuracy"]), pct(c_gem["overall"]["accuracy"])],
                ["C-MW-framing-full", "prompt framing probe", str(cf_gpt["overall"]["n"]), pct(cf_gpt["overall"]["accuracy"]), pct(cf_gem["overall"]["accuracy"])],
            ],
            "这页适合看整体 headline：A 和 D 是主结果；D-framing 与 C-framing 用来判断 wording sensitivity 是否跨任务复现。",
        )
        add_table_page(
            pdf,
            "Table 2. A-MW-v2-crossenv 按环境",
            ["Environment", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["fourrooms", str(a_gpt["breakdowns"]["by_env"]["fourrooms"]["n"]), pct(a_gpt["breakdowns"]["by_env"]["fourrooms"]["accuracy"]), pct(a_gem["breakdowns"]["by_env"]["fourrooms"]["accuracy"])],
                ["pickupobjects", str(a_gpt["breakdowns"]["by_env"]["pickupobjects"]["n"]), pct(a_gpt["breakdowns"]["by_env"]["pickupobjects"]["accuracy"]), pct(a_gem["breakdowns"]["by_env"]["pickupobjects"]["accuracy"])],
                ["putnext", str(a_gpt["breakdowns"]["by_env"]["putnext"]["n"]), pct(a_gpt["breakdowns"]["by_env"]["putnext"]["accuracy"]), pct(a_gem["breakdowns"]["by_env"]["putnext"]["accuracy"])],
            ],
            "Gemini 在 FourRooms 上明显更弱，但在 PickupObjects / PutNext 上和 GPT-5.4 持平。",
        )
        add_table_page(
            pdf,
            "Table 3. D-MW 按 family",
            ["Family", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["doorkey", str(d_gpt["breakdowns"]["by_family"]["doorkey"]["n"]), pct(acc(d_gpt["breakdowns"]["by_family"]["doorkey"])), pct(acc(d_gem["breakdowns"]["by_family"]["doorkey"]))],
                ["multiroom", str(d_gpt["breakdowns"]["by_family"]["multiroom"]["n"]), pct(acc(d_gpt["breakdowns"]["by_family"]["multiroom"])), pct(acc(d_gem["breakdowns"]["by_family"]["multiroom"]))],
                ["redblue", str(d_gpt["breakdowns"]["by_family"]["redblue"]["n"]), pct(acc(d_gpt["breakdowns"]["by_family"]["redblue"])), pct(acc(d_gem["breakdowns"]["by_family"]["redblue"]))],
            ],
            "这页最能看出 family-level 结构差异：GPT-5.4 更强于 DoorKey，而 Gemini 在 RedBlue 上反而更强。",
        )
        add_table_page(
            pdf,
            "Table 4. D-MW framing sensitivity",
            ["Framing", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["pos", str(df_gpt["breakdowns"]["by_framing"]["pos"]["n"]), pct(acc(df_gpt["breakdowns"]["by_framing"]["pos"])), pct(acc(df_gem["breakdowns"]["by_framing"]["pos"]))],
                ["neu", str(df_gpt["breakdowns"]["by_framing"]["neu"]["n"]), pct(acc(df_gpt["breakdowns"]["by_framing"]["neu"])), pct(acc(df_gem["breakdowns"]["by_framing"]["neu"]))],
                ["neg", str(df_gpt["breakdowns"]["by_framing"]["neg"]["n"]), pct(acc(df_gpt["breakdowns"]["by_framing"]["neg"])), pct(acc(df_gem["breakdowns"]["by_framing"]["neg"]))],
            ],
            "如果 D 线上也出现同方向 framing effect，就说明 wording sensitivity 不是 Task C 的特例。",
        )
        add_table_page(
            pdf,
            "Table 5. C-MW-formal 按 variant",
            ["Variant", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["full", str(c_gpt["breakdowns"]["by_variant"]["full"]["n"]), pct(acc(c_gpt["breakdowns"]["by_variant"]["full"])), pct(acc(c_gem["breakdowns"]["by_variant"]["full"]))],
                ["nocue", str(c_gpt["breakdowns"]["by_variant"]["nocue"]["n"]), pct(acc(c_gpt["breakdowns"]["by_variant"]["nocue"])), pct(acc(c_gem["breakdowns"]["by_variant"]["nocue"]))],
                ["cf", str(c_gpt["breakdowns"]["by_variant"]["cf"]["n"]), pct(acc(c_gpt["breakdowns"]["by_variant"]["cf"])), pct(acc(c_gem["breakdowns"]["by_variant"]["cf"]))],
            ],
            "GPT-5.4 明显偏向失败判断，而 Gemini 在 full / nocue 上更平衡。",
        )
        add_table_page(
            pdf,
            "Table 6. C-MW framing sensitivity",
            ["Framing", "n", "GPT-5.4", "Gemini-2.5-Flash"],
            [
                ["pos", str(cf_gpt["breakdowns"]["by_framing"]["pos"]["n"]), pct(acc(cf_gpt["breakdowns"]["by_framing"]["pos"])), pct(acc(cf_gem["breakdowns"]["by_framing"]["pos"]))],
                ["neu", str(cf_gpt["breakdowns"]["by_framing"]["neu"]["n"]), pct(acc(cf_gpt["breakdowns"]["by_framing"]["neu"])), pct(acc(cf_gem["breakdowns"]["by_framing"]["neu"]))],
                ["neg", str(cf_gpt["breakdowns"]["by_framing"]["neg"]["n"]), pct(acc(cf_gpt["breakdowns"]["by_framing"]["neg"])), pct(acc(cf_gem["breakdowns"]["by_framing"]["neg"]))],
            ],
            "两模型都保留了同方向 framing effect：positive 更高，negative 更低。",
        )

        add_figure_page(pdf, "Figure 1. MiniWorld Tier-1 双模型总览", fig_dir / "figure_mw_tier1_dual_model_overview.png")
        add_figure_page(pdf, "Figure 2. A-MW-v2-crossenv", fig_dir / "figure_a_mw_crossenv_accuracy.png")
        add_figure_page(pdf, "Figure 3. D-MW family 对比", fig_dir / "figure_d_mw_family_accuracy.png")
        add_figure_page(pdf, "Figure 4. D-MW framing 对比", fig_dir / "figure_d_mw_framing_accuracy.png")
        add_figure_page(pdf, "Figure 5. C-MW-formal variant 对比", fig_dir / "figure_c_mw_variant_accuracy.png")
        add_figure_page(pdf, "Figure 6. C-MW framing 对比", fig_dir / "figure_c_mw_framing_accuracy.png")

        add_text_page(
            pdf,
            "Paper-Facing Result Summary",
            [
                "双模型版本比之前的单模型图更有说服力，因为它不再只展示一个数字，而是展示哪些模式能跨模型复现、哪些模式具有明显的模型依赖性。",
                "A-MW-v2-crossenv 现在最重要的信息不是 60.0% 这个单点结果，而是 GPT-5.4 与 Gemini-2.5-Flash 在 FourRooms 上出现了显著分歧，但在 PickupObjects 与 PutNext 上又重新对齐。这让 A 线从“一个普通条形图”变成了更有解释力的 cross-env 对照。",
                "D-MW-reference-family-exam50 仍然是最强的 MiniWorld headline 线。两模型总分接近，但 family 结构不同：GPT-5.4 更强于 DoorKey，Gemini 更强于 RedBlue。这种差异使 family-level 图更值得进论文。",
                "新增的 D-MW full framing 结果可以回答一个更强的问题：prompt wording 的影响是不是只出现在 C 这种稀疏关键帧任务里。如果 D 上也出现同方向变化，这个结论就更有说服力。",
                "C-MW-formal 与 C-MW full framing 则共同构成分析证据。C-formal 说明 sparse-keyframe judgment 的误差模式并不统一，而 full framing 进一步表明 prompt wording 的方向性影响在两模型和全量题库上都能复现。",
            ],
        )

    print(f"Rendered review PDF -> {out}")


if __name__ == "__main__":
    main()
