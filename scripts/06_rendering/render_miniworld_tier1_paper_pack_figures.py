#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def acc(block: dict) -> float:
    n = int(block["n"])
    correct = int(block["correct"])
    return (correct / n) if n else 0.0


def _style_ax(ax, title: str, ylabel: str = "Accuracy") -> None:
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_bars(ax, bars, vals) -> None:
    for rect, v in zip(bars, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.02,
            f"{v:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _run(line: dict, label: str) -> dict:
    return next(x for x in line["model_runs"] if x["model_label"] == label)


def _grouped_bars(ax, categories, left_vals, right_vals, left_label, right_label, title):
    x = np.arange(len(categories))
    width = 0.34
    bars1 = ax.bar(x - width / 2, left_vals, width, label=left_label, color="#4C78A8")
    bars2 = ax.bar(x + width / 2, right_vals, width, label=right_label, color="#E45756")
    _style_ax(ax, title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(frameon=False, fontsize=9)
    _annotate_bars(ax, bars1, left_vals)
    _annotate_bars(ax, bars2, right_vals)


def render_main_dual(summary: dict, out_dir: Path) -> None:
    lines = summary["lines"]
    categories = ["A crossenv", "D exam50", "D framing", "C formal", "C framing"]
    gpt_vals = [_run(line, "GPT-5.4")["overall"]["accuracy"] for line in lines]
    gem_vals = [_run(line, "Gemini-2.5-Flash")["overall"]["accuracy"] for line in lines]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _grouped_bars(ax, categories, gpt_vals, gem_vals, "GPT-5.4", "Gemini-2.5-Flash", "MiniWorld Tier-1 Dual-Model Overview")
    fig.tight_layout()
    out = out_dir / "figure_mw_tier1_dual_model_overview.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_a_crossenv(summary: dict, out_dir: Path) -> None:
    line = next(x for x in summary["lines"] if x["line_id"] == "A-MW-v2-crossenv")
    gpt = _run(line, "GPT-5.4")
    gem = _run(line, "Gemini-2.5-Flash")
    labels = ["FourRooms", "PickupObjects", "PutNext"]
    gpt_vals = [
        gpt["breakdowns"]["by_env"]["fourrooms"]["accuracy"],
        gpt["breakdowns"]["by_env"]["pickupobjects"]["accuracy"],
        gpt["breakdowns"]["by_env"]["putnext"]["accuracy"],
    ]
    gem_vals = [
        gem["breakdowns"]["by_env"]["fourrooms"]["accuracy"],
        gem["breakdowns"]["by_env"]["pickupobjects"]["accuracy"],
        gem["breakdowns"]["by_env"]["putnext"]["accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    _grouped_bars(ax, labels, gpt_vals, gem_vals, "GPT-5.4", "Gemini-2.5-Flash", "A-MW-v2 Cross-Environment Accuracy")
    fig.tight_layout()
    out = out_dir / "figure_a_mw_crossenv_accuracy.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_d_family(summary: dict, out_dir: Path) -> None:
    line = next(x for x in summary["lines"] if x["line_id"] == "D-MW-reference-family-exam50")
    gpt = _run(line, "GPT-5.4")
    gem = _run(line, "Gemini-2.5-Flash")
    labels = ["DoorKey", "MultiRoom", "RedBlue"]
    gpt_vals = [
        acc(gpt["breakdowns"]["by_family"]["doorkey"]),
        acc(gpt["breakdowns"]["by_family"]["multiroom"]),
        acc(gpt["breakdowns"]["by_family"]["redblue"]),
    ]
    gem_vals = [
        acc(gem["breakdowns"]["by_family"]["doorkey"]),
        acc(gem["breakdowns"]["by_family"]["multiroom"]),
        acc(gem["breakdowns"]["by_family"]["redblue"]),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    _grouped_bars(ax, labels, gpt_vals, gem_vals, "GPT-5.4", "Gemini-2.5-Flash", "D-MW Family Accuracy")
    fig.tight_layout()
    out = out_dir / "figure_d_mw_family_accuracy.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_c_variant(summary: dict, out_dir: Path) -> None:
    line = next(x for x in summary["lines"] if x["line_id"] == "C-MW-formal")
    gpt = _run(line, "GPT-5.4")
    gem = _run(line, "Gemini-2.5-Flash")
    labels = ["full", "nocue", "cf"]
    gpt_vals = [
        acc(gpt["breakdowns"]["by_variant"]["full"]),
        acc(gpt["breakdowns"]["by_variant"]["nocue"]),
        acc(gpt["breakdowns"]["by_variant"]["cf"]),
    ]
    gem_vals = [
        acc(gem["breakdowns"]["by_variant"]["full"]),
        acc(gem["breakdowns"]["by_variant"]["nocue"]),
        acc(gem["breakdowns"]["by_variant"]["cf"]),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    _grouped_bars(ax, labels, gpt_vals, gem_vals, "GPT-5.4", "Gemini-2.5-Flash", "C-MW Variant Accuracy")
    fig.tight_layout()
    out = out_dir / "figure_c_mw_variant_accuracy.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_c_framing(summary: dict, out_dir: Path) -> None:
    line = next(x for x in summary["lines"] if x["line_id"] == "C-MW-framing-full")
    gpt = _run(line, "GPT-5.4")
    gem = _run(line, "Gemini-2.5-Flash")
    labels = ["pos", "neu", "neg"]
    gpt_vals = [
        acc(gpt["breakdowns"]["by_framing"]["pos"]),
        acc(gpt["breakdowns"]["by_framing"]["neu"]),
        acc(gpt["breakdowns"]["by_framing"]["neg"]),
    ]
    gem_vals = [
        acc(gem["breakdowns"]["by_framing"]["pos"]),
        acc(gem["breakdowns"]["by_framing"]["neu"]),
        acc(gem["breakdowns"]["by_framing"]["neg"]),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    _grouped_bars(ax, labels, gpt_vals, gem_vals, "GPT-5.4", "Gemini-2.5-Flash", "C-MW Framing Sensitivity")
    fig.tight_layout()
    out = out_dir / "figure_c_mw_framing_accuracy.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_d_framing(summary: dict, out_dir: Path) -> None:
    line = next(x for x in summary["lines"] if x["line_id"] == "D-MW-framing-full")
    gpt = _run(line, "GPT-5.4")
    gem = _run(line, "Gemini-2.5-Flash")
    labels = ["pos", "neu", "neg"]
    gpt_vals = [
        acc(gpt["breakdowns"]["by_framing"]["pos"]),
        acc(gpt["breakdowns"]["by_framing"]["neu"]),
        acc(gpt["breakdowns"]["by_framing"]["neg"]),
    ]
    gem_vals = [
        acc(gem["breakdowns"]["by_framing"]["pos"]),
        acc(gem["breakdowns"]["by_framing"]["neu"]),
        acc(gem["breakdowns"]["by_framing"]["neg"]),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    _grouped_bars(ax, labels, gpt_vals, gem_vals, "GPT-5.4", "Gemini-2.5-Flash", "D-MW Framing Sensitivity")
    fig.tight_layout()
    out = out_dir / "figure_d_mw_framing_accuracy.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render MiniWorld Tier-1 paper-pack figures.")
    ap.add_argument(
        "--summary_json",
        default="tmp_miniworld/miniworld-tier1-paper-pack/miniworld_tier1_results_summary.json",
    )
    ap.add_argument(
        "--out_dir",
        default="tmp_miniworld/miniworld-tier1-paper-pack/figures",
    )
    args = ap.parse_args()

    summary = load_summary(Path(args.summary_json))
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    render_main_dual(summary, out_dir)
    render_a_crossenv(summary, out_dir)
    render_d_family(summary, out_dir)
    render_d_framing(summary, out_dir)
    render_c_variant(summary, out_dir)
    render_c_framing(summary, out_dir)

    print(f"Rendered figures -> {out_dir}")


if __name__ == "__main__":
    main()
