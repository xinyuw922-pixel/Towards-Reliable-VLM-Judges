#!/usr/bin/env python3
"""
Action Ablation Scorer

Reference: Stats Spec §2.3 C2
Compute action ablation accuracy drop + McNemar significance test
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.stats import chi2
from dataclasses import dataclass


@dataclass
class AblationResult:
    """Action ablation result"""
    acc_original: float
    acc_shuffle: float
    acc_mask: float
    delta_shuffle: float    # acc_original - acc_shuffle
    delta_mask: float      # acc_original - acc_mask
    p_shuffle: float       # McNemar p-value
    p_mask: float          # McNemar p-value
    action_ignorant: bool  # Whether marked as Action-Ignorant


def load_responses(response_path: str) -> pd.DataFrame:
    """Load inference results.

    Handles two input formats:
    - Raw inference output: has 'response' column (raw model text)
    - Gold-enriched output (from score_exam.py --dump_gold): has 'raw' column
      instead of 'response'; we alias it so downstream code works unchanged.
    """
    records = []
    with open(response_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
    df = pd.DataFrame(records)
    # Compatibility:
    # - Raw inference output (from run_inference.py): has 'raw' but no 'response';
    #   model name is nested in meta['model'], not at top level.
    # - Gold-enriched output (from score_exam.py --dump_gold): has neither
    #   'raw' nor 'response', only 'pred_norm'.
    if "response" not in df.columns and "raw" in df.columns:
        df["response"] = df["raw"]
    if "model" not in df.columns and "meta" in df.columns:
        df["model"] = df["meta"].apply(lambda m: m.get("model") if isinstance(m, dict) else None)
    return df


def parse_task_a_response(response: str, choices: List[str] = None) -> int:
    """Parse Task A model response"""
    response = response.lower().strip()
    
    # Try to extract A/B/C/D
    for choice in ['a', 'b', 'c', 'd']:
        if choice in response:
            return ord(choice) - ord('a')
    
    # Try to extract first letter
    if response:
        first = response[0]
        if first in 'abcd':
            return ord(first) - ord('a')
    
    return -1  # Unable to parse


def mcnemar_test(original_results: List[int], ablated_results: List[int]) -> Tuple[int, int, float]:
    """
    McNemar test.
    
    Returns:
        (discordant_01, discordant_10, p_value)
    """
    n00 = n01 = n10 = n11 = 0
    
    for orig, abl in zip(original_results, ablated_results):
        if orig == 1 and abl == 1:
            n11 += 1
        elif orig == 1 and abl == 0:
            n10 += 1
        elif orig == 0 and abl == 1:
            n01 += 1
        else:
            n00 += 1
    
    # McNemar statistic
    if n01 + n10 > 0:
        # Use continuity correction
        chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
    else:
        p_value = 1.0
    
    return n01, n10, p_value


def compute_ablation_score(
    original_responses: pd.DataFrame,
    shuffle_responses: pd.DataFrame = None,
    mask_responses: pd.DataFrame = None
) -> Dict[str, AblationResult]:
    """
    Compute action ablation results.
    
    Args:
        original_responses: Original Task A responses
        shuffle_responses: Shuffle version responses
        mask_responses: Mask version responses
    
    Returns:
        Ablation results grouped by model
    """
    results = {}
    
    # Group by model
    for model_name, model_df in original_responses.groupby('model'):
        # Parse responses and compute accuracy
        model_df = model_df.copy()
        model_df['pred'] = model_df['response'].apply(parse_task_a_response)
        model_df['gold'] = model_df.get('gold', -1)
        model_df['correct'] = (model_df['pred'] == model_df['gold']).astype(int)
        
        acc_original = model_df['correct'].mean()
        
        result = AblationResult(
            acc_original=acc_original,
            acc_shuffle=0.0,
            acc_mask=0.0,
            delta_shuffle=0.0,
            delta_mask=0.0,
            p_shuffle=1.0,
            p_mask=1.0,
            action_ignorant=False
        )
        
        # Shuffle
        if shuffle_responses is not None:
            shuffle_df = shuffle_responses[shuffle_responses['model'] == model_name].copy()
            shuffle_df['pred'] = shuffle_df['response'].apply(parse_task_a_response)
            shuffle_df['gold'] = shuffle_df.get('gold', -1)
            shuffle_df['correct'] = (shuffle_df['pred'] == shuffle_df['gold']).astype(int)
            result.acc_shuffle = shuffle_df['correct'].mean()
            result.delta_shuffle = result.acc_original - result.acc_shuffle
            
            # McNemar test
            if len(model_df) == len(shuffle_df):
                _, _, result.p_shuffle = mcnemar_test(
                    model_df['correct'].tolist(),
                    shuffle_df['correct'].tolist()
                )
        
        # Mask
        if mask_responses is not None:
            mask_df = mask_responses[mask_responses['model'] == model_name].copy()
            mask_df['pred'] = mask_df['response'].apply(parse_task_a_response)
            mask_df['gold'] = mask_df.get('gold', -1)
            mask_df['correct'] = (mask_df['pred'] == mask_df['gold']).astype(int)
            result.acc_mask = mask_df['correct'].mean()
            result.delta_mask = result.acc_original - result.acc_mask
            
            # McNemar test
            if len(model_df) == len(mask_df):
                _, _, result.p_mask = mcnemar_test(
                    model_df['correct'].tolist(),
                    mask_df['correct'].tolist()
                )
        
        # Determine Action-Ignorant
        # Criteria: p < 0.05 and Δ > 0.05
        if (result.p_shuffle < 0.05 or result.p_mask < 0.05) and \
           (result.delta_shuffle > 0.05 or result.delta_mask > 0.05):
            result.action_ignorant = False  # Model used action
        else:
            result.action_ignorant = True   # Mark as Action-Ignorant
        
        results[model_name] = result
    
    return results


def print_ablation_results(results: Dict[str, AblationResult]):
    """Print ablation results"""
    print(f"\n{'='*80}")
    print("Action Ablation Results")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Acc_orig':>10} {'Acc_shuf':>10} {'Acc_mask':>10} "
          f"{'Δ_shuf':>10} {'Δ_mask':>10} {'p_shuf':>10} {'p_mask':>10} {'Action-Ign?':>12}")
    print("-" * 80)
    
    for model, r in results.items():
        ai = "YES" if r.action_ignorant else "NO"
        print(f"{model:<20} {r.acc_original:>10.3f} {r.acc_shuffle:>10.3f} "
              f"{r.acc_mask:>10.3f} {r.delta_shuffle:>10.3f} {r.delta_mask:>10.3f} "
              f"{r.p_shuffle:>10.4f} {r.p_mask:>10.4f} {ai:>12}")


def main():
    parser = argparse.ArgumentParser(description="Action Ablation Scorer")
    parser.add_argument("--original", "-o", required=True, help="Original Task A responses")
    parser.add_argument("--shuffle", "-s", help="Shuffle version responses")
    parser.add_argument("--mask", "-m", help="Mask version responses")
    parser.add_argument("--output-csv", help="Output CSV path")
    args = parser.parse_args()
    
    # Load responses
    original_df = load_responses(args.original)
    shuffle_df = load_responses(args.shuffle) if args.shuffle else None
    mask_df = load_responses(args.mask) if args.mask else None
    
    # Compute results
    results = compute_ablation_score(original_df, shuffle_df, mask_df)
    
    # Print
    print_ablation_results(results)
    
    # Output CSV
    if args.output_csv:
        rows = []
        for model, r in results.items():
            rows.append({
                'model': model,
                'acc_original': r.acc_original,
                'acc_shuffle': r.acc_shuffle,
                'acc_mask': r.acc_mask,
                'delta_shuffle': r.delta_shuffle,
                'delta_mask': r.delta_mask,
                'p_shuffle': r.p_shuffle,
                'p_mask': r.p_mask,
                'action_ignorant': r.action_ignorant
            })
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
