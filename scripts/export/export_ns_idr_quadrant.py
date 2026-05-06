#!/usr/bin/env python3
"""
NS-IDR Quadrant Data Exporter

Export data required for NS-IDR quadrant plot.
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# Import existing calculation modules
from compute_idr import load_verdicts, load_verdicts_from_per_row, compute_idr, compute_ns
from compute_les import load_verdicts_for_les, load_verdicts_for_les_from_per_row, compute_les


def export_quadrant_csv(
    idr_results: Dict,
    ns_results: Dict,
    les_results: Dict = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Export NS-IDR quadrant plot data.
    
    Output CSV contains:
    - model, task
    - NS_T, NS_vis, NS_frame
    - IDR_raw, IDR_cond
    - G_active_pct
    - Degenerate
    - LES_T, LES_vis, LES_frame
    """
    rows = []
    
    for model, model_idr in idr_results.items():
        model_ns = ns_results.get(model, {})
        model_les = les_results.get(model, {}) if les_results else {}
        
        for task, idr in model_idr.items():
            ns = model_ns.get(task, {})
            les = model_les.get(task, {})
            
            row = {
                'model': model,
                'task': task,
                'IDR_raw': idr.idr_raw,
                'IDR_cond': idr.idr_cond if not np.isnan(idr.idr_cond) else None,
                'G_active_pct': idr.active_rate,
                'Degenerate': idr.degenerate,
            }
            
            # NS
            row['NS_T'] = ns.get('NS_rev', ns.get('NS_temp'))
            row['NS_vis'] = ns.get('NS_noisy', ns.get('NS_visual'))
            row['NS_frame'] = ns.get('NS_frame')
            
            # LES
            if les:
                for family, les_r in les.items():
                    row[f'LES_{family}'] = les_r.les_beta
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Exported quadrant data to {output_path}")
    
    return df


def classify_quadrant(row: pd.Series) -> str:
    """
    Classify quadrant based on NS and IDR.
    
    NS-IDR quadrants:
    - Ideal: NS high, IDR high
    - Degraded: NS high, IDR low (constant output)
    - Fragile: NS low, IDR high (prompt sensitive)
    - Random: NS low, IDR low (random)
    """
    ns = row.get('NS_T', 0.5)
    idr = row.get('IDR_cond', 0.5)
    degenerate = row.get('Degenerate')
    
    if degenerate:
        return 'degenerate'
    
    if ns >= 0.7 and idr >= 0.5:
        return 'ideal'
    elif ns >= 0.7 and idr < 0.5:
        return 'degraded'
    elif ns < 0.7 and idr >= 0.5:
        return 'fragile'
    else:
        return 'random'


def main():
    parser = argparse.ArgumentParser(description="NS-IDR Quadrant Data Exporter")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--responses", "-r", help="Responses JSONL")
    source_group.add_argument("--per-row-csv", help="Unified per-row CSV")
    parser.add_argument("--output-csv", "-o", required=True, help="Output CSV path")
    parser.add_argument("--include-les", action="store_true", help="Include LES results")
    args = parser.parse_args()

    print("Loading verdicts...")
    if args.per_row_csv:
        verdicts = load_verdicts_from_per_row(args.per_row_csv)
    else:
        verdicts = load_verdicts(args.responses)
    
    print("Computing IDR...")
    idr_results = compute_idr(verdicts)
    
    print("Computing NS...")
    ns_results = compute_ns(verdicts)
    
    les_results = None
    if args.include_les:
        print("Computing LES...")
        if args.per_row_csv:
            les_results = compute_les(load_verdicts_for_les_from_per_row(args.per_row_csv))
        else:
            les_results = compute_les(load_verdicts_for_les(args.responses))
    
    print("Exporting quadrant data...")
    df = export_quadrant_csv(idr_results, ns_results, les_results, args.output_csv)
    
    # Add quadrant classification
    df['quadrant'] = df.apply(classify_quadrant, axis=1)
    
    print(f"\nQuadrant distribution:")
    print(df['quadrant'].value_counts())
    
    print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
