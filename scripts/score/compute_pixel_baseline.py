#!/usr/bin/env python3
"""
Pixel Similarity Baseline Calculator

Reference: Stats Spec §6.5
Compute pixel-level heuristic baseline for Task A: SSIM and L2
Quality gate: acc_ssim ≤ 0.35, acc_l2 ≤ 0.35
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import sys

# Try to import scikit-image
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available, SSIM baseline will not work", file=sys.stderr)

from PIL import Image


@dataclass
class PixelBaselineResult:
    """Pixel baseline result"""
    acc_ssim: float
    acc_l2: float
    mean_ssim_gold: float
    mean_ssim_neg: float
    ssim_separability: float
    gate_passed: bool  # acc_ssim ≤ 0.35


def load_task_a_exam(exam_path: str) -> List[Dict]:
    """Load Task A exam"""
    items = []
    with open(exam_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_image(path: str):
    """Load image, return PIL Image object"""
    img = Image.open(path)
    return img


def compute_pixel_baseline(exam_items: List[Dict], image_root: str = None) -> Dict[str, PixelBaselineResult]:
    """
    Compute pixel-level heuristic baseline.

    Note: Supports two exam formats:
    1. New format (build_exam.py): image field points to composite image, extract candidates from it
    2. Old format (direct build): current_frame, candidates, gold_index fields

    Args:
        exam_items: Task A exam items
        image_root: Image root directory

    Returns:
        Baseline results grouped by environment
    """
    if not SKIMAGE_AVAILABLE:
        print("Error: scikit-image required for SSIM calculation")
        return {}

    results = {}

    # Group by task
    by_task = defaultdict(list)
    for item in exam_items:
        task = item.get('uid', '').split('.')[1] if '.' in item.get('uid', '') else 'unknown'
        by_task[task].append(item)

    for task, items in by_task.items():
        correct_ssim = 0
        correct_l2 = 0

        all_gold_ssim = []
        all_neg_ssim = []

        valid_count = 0

        for item in items:
            # Determine image root directory
            img_root = Path(image_root) if image_root else Path.cwd()

            # New format: build_exam.py outputs a 2x2 composite image
            # Need to extract candidates directly from the image
            if 'image' in item:
                # Infer from composite image path
                img_path = Path(item['image'])
                if not img_root.exists():
                    img_root = img_path.parent
                full_path = img_root / img_path

                if not full_path.exists():
                    continue

                # Load composite image
                try:
                    composite = load_image(str(full_path))
                except Exception:
                    continue

                # Parse composite image layout
                # build_exam.py creates 3x2 layout: current frame in left column (vertically centered), candidates in right 2x2 grid
                # composite is now a PIL Image, get size properly
                w, h = composite.size
                tile_w = w // 3  # width per column (original frame width)
                tile_h = h // 2  # height per row (original frame height)

                # Extract current frame (left column, vertically centered)
                # comp.paste(curr_im, (0, h//2)) original code places current frame at y=h//2
                current = np.array(composite.crop((0, h//2, tile_w, h//2+tile_h)))

                # Extract 4 candidates (right half 2x2 grid)
                # Layout: col=1,2 (x=tile_w or 2*tile_w), row=0,1 (y=0 or tile_h)
                # Order: A=(tile_w, 0), B=(2*tile_w, 0), C=(tile_w, tile_h), D=(2*tile_w, tile_h)
                candidates = []
                for row in range(2):
                    for col in range(1, 3):  # Columns 1 and 2
                        x_start = col * tile_w
                        y_start = row * tile_h
                        cand_img = np.array(composite.crop((x_start, y_start, x_start+tile_w, y_start+tile_h)))
                        candidates.append(cand_img)

                # Get correct answer index
                answer = item.get('answer', '')
                label = item.get('label', 0)
                # answer is letter (A/B/C/D), label is numeric index (0/1/2/3)
                if isinstance(answer, str) and answer:
                    correct_idx = ord(answer.upper()) - ord('A')
                else:
                    correct_idx = label if label is not None else 0

            else:
                # Old format: has current_frame and candidates directly
                current_path = img_root / item.get('current_frame', '')
                if not current_path.exists():
                    continue

                try:
                    current = np.array(load_image(str(current_path)))
                except Exception:
                    continue

                candidates_info = item.get('candidates', [])
                correct_idx = item.get('gold_index', 0)

                candidates = []
                for cand_info in candidates_info:
                    cand_path = img_root / cand_info.get('image', '')
                    if not cand_path.exists():
                        continue
                    try:
                        candidates.append(np.array(load_image(str(cand_path))))
                    except Exception:
                        continue

            if len(candidates) != 4:
                continue

            # Compute SSIM and L2
            ssim_scores = []
            l2_scores = []

            for idx, cand in enumerate(candidates):
                # SSIM
                s = ssim(current, cand, win_size=7, channel_axis=2, data_range=255)
                ssim_scores.append(s)

                # L2 (negative, higher is better)
                l2 = -np.mean((current.astype(float) - cand.astype(float)) ** 2)
                l2_scores.append(l2)

                if idx == correct_idx:
                    all_gold_ssim.append(s)
                else:
                    all_neg_ssim.append(s)

            if len(ssim_scores) == 4:
                # Predict: choose the one with highest score
                pred_ssim = np.argmax(ssim_scores)
                pred_l2 = np.argmax(l2_scores)

                if pred_ssim == correct_idx:
                    correct_ssim += 1
                if pred_l2 == correct_idx:
                    correct_l2 += 1

                valid_count += 1

        if valid_count > 0:
            acc_ssim = correct_ssim / valid_count
            acc_l2 = correct_l2 / valid_count

            mean_gold = np.mean(all_gold_ssim) if all_gold_ssim else 0
            mean_neg = np.mean(all_neg_ssim) if all_neg_ssim else 0

            # Separability
            if len(all_gold_ssim) > 1 and len(all_neg_ssim) > 1:
                pooled_std = np.sqrt((np.var(all_gold_ssim) + np.var(all_neg_ssim)) / 2)
                if pooled_std > 0:
                    ssim_sep = (mean_gold - mean_neg) / pooled_std
                else:
                    ssim_sep = 0
            else:
                ssim_sep = 0

            results[task] = PixelBaselineResult(
                acc_ssim=acc_ssim,
                acc_l2=acc_l2,
                mean_ssim_gold=mean_gold,
                mean_ssim_neg=mean_neg,
                ssim_separability=ssim_sep,
                gate_passed=acc_ssim <= 0.35
            )

    return results


def print_baseline_results(results: Dict[str, PixelBaselineResult]):
    """Print baseline results"""
    print("\n" + "=" * 70)
    print("Pixel Similarity Baseline Results")
    print("=" * 70)
    print(f"{'Task':<15} {'Acc_SSIM':>10} {'Acc_L2':>10} {'SSIM_sep':>10} {'Gate':>10}")
    print("-" * 70)
    
    for task, r in results.items():
        gate = "PASS" if r.gate_passed else "FAIL"
        print(f"{task:<15} {r.acc_ssim:>10.3f} {r.acc_l2:>10.3f} "
              f"{r.ssim_separability:>10.3f} {gate:>10}")
    
    # Summary
    total = len(results)
    passed = sum(1 for r in results.values() if r.gate_passed)
    print("-" * 70)
    print(f"Overall: {passed}/{total} tasks passed gate (acc_ssim ≤ 0.35)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pixel Similarity Baseline Calculator")
    parser.add_argument("--exam", "-e", required=True, help="Task A exam JSONL")
    parser.add_argument("--image-root", "-i", help="Image root directory")
    parser.add_argument("--output-csv", "-o", help="Output CSV path")
    args = parser.parse_args()
    
    # Load exam
    exam_items = load_task_a_exam(args.exam)
    print(f"Loaded {len(exam_items)} Task A items")
    
    # Compute
    results = compute_pixel_baseline(exam_items, args.image_root)
    
    # Print
    print_baseline_results(results)
    
    # Output CSV
    if args.output_csv:
        import pandas as pd
        rows = []
        for task, r in results.items():
            rows.append({
                'task': task,
                'acc_ssim': r.acc_ssim,
                'acc_l2': r.acc_l2,
                'mean_ssim_gold': r.mean_ssim_gold,
                'mean_ssim_neg': r.mean_ssim_neg,
                'ssim_separability': r.ssim_separability,
                'gate_passed': r.gate_passed
            })
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
