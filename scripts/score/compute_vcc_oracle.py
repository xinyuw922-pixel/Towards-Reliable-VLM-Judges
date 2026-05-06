#!/usr/bin/env python3
"""
VCC_oracle Calculator

Computes pixel vs oracle-text consistency
Reference: Stats Spec §5
VCC_oracle = P(y_pixel = y_text)
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys as _sys
_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent.parent  # repo root is 2 levels up from scripts/<module>/
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from scripts.schema.exam_schema import parse_verdict as _parse_verdict_ssot


def load_responses(path: str) -> List[Dict]:
    """Load responses"""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def parse_verdict(response: str) -> int:
    """Parse verdict (consistent with score_exam.py)."""
    ans, mode = _parse_verdict_ssot(response)
    if ans == "Success":
        return 1
    if ans == "Fail":
        return 0
    return -1


def compute_vcc(pixel_responses: List[Dict], text_responses: List[Dict], strict_filter: bool = True) -> Dict:
    """
    Compute VCC_oracle.

    Args:
        pixel_responses: Pixel version responses
        text_responses: Oracle-text version responses
        strict_filter: If True, only match samples with temporal=orig and visual=clean

    Returns:
        VCC result
    """
    def is_orig_clean(uid: str) -> bool:
        """Check if UID is orig + clean"""
        parts = uid.split('.')
        # Format: C.<env>.<group_id>.<variant>.<temporal>.<visual>[.<framing>]
        temporal = parts[4] if len(parts) > 4 else ''
        visual = parts[5] if len(parts) > 5 else ''
        return temporal in ('orig', '') and visual in ('clean', '')
    
    # Build index
    pixel_index = {}
    for r in pixel_responses:
        uid = r.get('exam_id', '')
        # Remove oracle marker
        base_uid = uid.replace('.oracle', '')
        
        # Strict filter: only keep orig + clean
        if strict_filter and not is_orig_clean(base_uid):
            continue
            
        pixel_index[base_uid] = r

    text_index = {}
    for r in text_responses:
        uid = r.get('exam_id', '')
        base_uid = uid.replace('.oracle', '')
        
        # Strict filter: only keep orig + clean
        if strict_filter and not is_orig_clean(base_uid):
            continue
            
        text_index[base_uid] = r

    # Match
    matches = 0
    total = 0
    by_variant = {}

    for uid in pixel_index:
        if uid in text_index:
            # Prefer pred field, then raw, then response
            pixel_resp = pixel_index[uid].get('pred', pixel_index[uid].get('raw', pixel_index[uid].get('response', '')))
            text_resp = text_index[uid].get('pred', text_index[uid].get('raw', text_index[uid].get('response', '')))

            pixel_verdict = parse_verdict(pixel_resp)
            text_verdict = parse_verdict(text_resp)

            if pixel_verdict >= 0 and text_verdict >= 0:
                # Extract variant
                parts = uid.split('.')
                variant = parts[3] if len(parts) > 3 else 'unknown'

                if pixel_verdict == text_verdict:
                    matches += 1
                    by_variant.setdefault(variant, {'match': 0, 'total': 0})
                    by_variant[variant]['match'] += 1
                else:
                    by_variant.setdefault(variant, {'match': 0, 'total': 0})

                by_variant[variant]['total'] += 1
                total += 1

    vcc_overall = matches / total if total > 0 else 0

    result = {
        'vcc_overall': vcc_overall,
        'n_matched': total,
        'strict_filter': strict_filter,
        'by_variant': {}
    }

    for variant, counts in by_variant.items():
        result['by_variant'][variant] = counts['match'] / counts['total'] if counts['total'] > 0 else 0

    return result


def print_vcc_results(result: Dict):
    """Print VCC results"""
    print("\n" + "=" * 50)
    print("VCC_oracle Results")
    print("=" * 50)
    print(f"Overall VCC:  {result['vcc_overall']:.3f}")
    print(f"Matched:      {result['n_matched']}")
    print("\nBy variant:")
    for variant, vcc in result['by_variant'].items():
        print(f"  {variant}: {vcc:.3f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="VCC_oracle Calculator")
    parser.add_argument("--pixel", "-p", required=True, help="Pixel version responses")
    parser.add_argument("--oracle", "-o", required=True, help="Oracle text version responses")
    parser.add_argument("--output-csv", help="Output CSV")
    parser.add_argument("--no-strict-filter", action="store_true", help="Disable strict orig+clean filter")
    args = parser.parse_args()
    
    # Load
    pixel_responses = load_responses(args.pixel)
    oracle_responses = load_responses(args.oracle)
    
    print(f"Loaded {len(pixel_responses)} pixel responses")
    print(f"Loaded {len(oracle_responses)} oracle responses")
    
    # Compute (default strict filter)
    strict_filter = not args.no_strict_filter
    if strict_filter:
        print("Using strict filter: temporal=orig AND visual=clean only")
    
    result = compute_vcc(pixel_responses, oracle_responses, strict_filter=strict_filter)
    
    # Print
    print_vcc_results(result)
    
    # CSV
    if args.output_csv:
        rows = [{'variant': 'overall', 'vcc': result['vcc_overall'], 'n': result['n_matched']}]
        for variant, vcc in result['by_variant'].items():
            rows.append({'variant': variant, 'vcc': vcc})
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
