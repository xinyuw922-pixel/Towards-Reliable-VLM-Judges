#!/usr/bin/env python3
"""
LES (Leakage Effect Size) Calculator

Reference: Stats Spec §2
Computes C1 core metrics: LES (Leakage Effect Size)

Three-level fallback:
- L1: GEE Binomial + Exchangeable correlation (preferred)
- L2: Cochran-Mantel-Haenszel test
- L3: McNemar exact test (used when M_probe=2)
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from scipy import stats
import warnings
import sys as _sys

_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from scripts.schema.exam_schema import parse_verdict as _parse_verdict_ssot

# Try to import statsmodels
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available, using fallback methods only")


# Chinn 2000 standardized effect size constant
SIGMA_LOGIT = np.pi / np.sqrt(3)


@dataclass
class LESResult:
    """LES calculation result"""
    les_beta: float          # max|β| (log-odds shift)
    d_logit: float          # Standardized effect size
    odds_ratio: float        # exp(β)
    beta: float              # Raw β coefficient
    se: float                # Standard error
    p_value: float           # Raw p-value
    p_fdr: float            # FDR-corrected p-value
    method: str             # "L1_GEE" / "L2_CMH" / "L3_McNemar"
    converged: bool         # Whether the model converged


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def load_verdicts_for_les(responses_path: str) -> pd.DataFrame:
    """
    Load data format required for LES.

    Returns:
        DataFrame with columns: model, task, group_id, family, phi_id, y_pred
    """
    records = []

    with open(responses_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            exam_id = record.get('exam_id', record.get('uid', ''))
            parts = exam_id.split('.')

            # Parse exam_id: C.<task>.<group_id>.<variant>.<temporal>.<visual>.<framing>
            if len(parts) >= 6:
                task = parts[1]
                group_id = parts[2]
                variant = parts[3]
                temporal = parts[4]
                visual = parts[5] if len(parts) > 5 else ''

                # Determine family
                # framing has highest priority (if pos/neu/neg exists)
                if len(parts) > 6 and parts[6] in ['pos', 'neu', 'neg']:
                    family = 'framing'
                    phi_id = parts[6]
                elif temporal not in ['orig', '']:
                    family = 'temporal'
                    phi_id = temporal
                elif visual not in ['clean', '']:
                    family = 'visual'
                    phi_id = visual
                else:
                    family = 'baseline'
                    phi_id = 'baseline'

                # Only process non-baseline probes
                if family != 'baseline':
                    # Parse model response - prefer pred field, fall back to raw
                    response = record.get('pred', record.get('raw', record.get('response', '')))
                    pred = parse_verdict(response)

                    if pred >= 0:  # Valid parse
                        # Model name may be in meta.model or top-level model field
                        model_name = record.get('model', record.get('meta', {}).get('model', 'unknown'))
                        records.append({
                            'model': model_name,
                            'task': task,
                            'group_id': group_id,
                            'family': family,
                            'phi_id': phi_id,
                            'variant': variant,
                            'y_pred': pred
                        })

    return pd.DataFrame(records)


def load_verdicts_for_les_from_per_row(per_row_csv: str) -> pd.DataFrame:
    """
    Load LES input records from unified `per-row.csv`.

    This mirrors the legacy `load_verdicts_for_les()` record contract so the
    statistical code can stay unchanged while new callers use unified exports.
    """
    df = pd.read_csv(per_row_csv)
    if df.empty:
        return pd.DataFrame(columns=["model", "task", "group_id", "family", "phi_id", "variant", "y_pred"])

    if "task" in df.columns:
        # LES is primarily for legacy Task C (temporal/visual/framing probe richness).
        # Task D has no temporal/visual/framing probes — it would yield empty results, which is
        # correct behavior. Include both so the script handles any future extension gracefully.
        df = df[df["task"].isin(["C", "D"])].copy()

    records = []
    for _, row in df.iterrows():
        temporal = _clean_text(row.get("temporal"))
        visual = _clean_text(row.get("visual"))
        framing = _clean_text(row.get("framing"))
        pred_norm = _clean_text(row.get("pred_norm"))
        response = _clean_text(row.get("response_text_pred")) or _clean_text(row.get("response_text_raw"))

        if framing in ["pos", "neu", "neg"]:
            family = "framing"
            phi_id = framing
        elif temporal not in ["", "orig"]:
            family = "temporal"
            phi_id = temporal
        elif visual not in ["", "clean"]:
            family = "visual"
            phi_id = visual
        else:
            family = "baseline"
            phi_id = "baseline"

        if family == "baseline":
            continue

        if pred_norm == "Success":
            pred = 1
        elif pred_norm == "Fail":
            pred = 0
        else:
            pred = parse_verdict(response)

        if pred >= 0:
            records.append({
                "model": _clean_text(row.get("model_id")) or "unknown",
                "task": _clean_text(row.get("env_task")) or "unknown",
                "group_id": _clean_text(row.get("group_id")),
                "family": family,
                "phi_id": phi_id,
                "variant": _clean_text(row.get("variant")),
                "y_pred": pred,
            })

    return pd.DataFrame(records)


def parse_verdict(response: str) -> int:
    """Parse verdict from model response (consistent with score_exam.py)."""
    ans, mode = _parse_verdict_ssot(response)
    if ans == "Success":
        return 1
    if ans == "Fail":
        return 0
    else:
        return -1


def compute_les_gee(df: pd.DataFrame, family: str, baseline_phi_id: str) -> LESResult:
    """
    L1: GEE Binomial method.

    Fits mixed-effects logistic regression:
    logit(P(y = Success)) = β_0 + β_F · φ_id + (1 | g)
    """
    if not STATSMODELS_AVAILABLE:
        return compute_les_cmh(df, family, baseline_phi_id)

    family_df = df[df['family'] == family].copy()

    if len(family_df) < 10:
        return compute_les_cmh(family_df, family, baseline_phi_id)

    # Build encoding
    all_phi_ids = sorted(family_df['phi_id'].unique())
    cat_order = [baseline_phi_id] + [x for x in all_phi_ids if x != baseline_phi_id]

    try:
        family_df['phi_id'] = pd.Categorical(family_df['phi_id'], categories=cat_order)

        # GEE model
        model = smf.gee(
            "y_pred ~ C(phi_id)",
            groups="group_id",
            data=family_df,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable()
        ).fit(disp=False)

        # Extract coefficients
        beta_coeffs = {}
        se_coeffs = {}
        p_values = {}

        for name in model.params.index:
            if 'phi_id' in name and 'Intercept' not in name:
                # Simplify coefficient names
                phi_name = name.replace('C(phi_id)[T.', '').replace(']', '')
                beta_coeffs[phi_name] = model.params[name]
                se_coeffs[phi_name] = model.bse[name]
                p_values[phi_name] = model.pvalues[name]

        if not beta_coeffs:
            return compute_les_cmh(family_df, family, baseline_phi_id)

        # FDR correction
        _, p_fdr, _, _ = multipletests(list(p_values.values()), method='fdr_bh')
        p_fdr_dict = dict(zip(p_values.keys(), p_fdr))

        # Take max |β|
        max_beta_phi = max(beta_coeffs.items(), key=lambda x: abs(x[1]))
        max_phi = max_beta_phi[0]
        les_beta = abs(max_beta_phi[1])

        result = LESResult(
            les_beta=les_beta,
            d_logit=les_beta / SIGMA_LOGIT,
            odds_ratio=np.exp(max_beta_phi[1]),
            beta=max_beta_phi[1],
            se=se_coeffs[max_phi],
            p_value=p_values[max_phi],
            p_fdr=p_fdr_dict[max_phi],
            method="L1_GEE",
            converged=True
        )

        return result

    except Exception as e:
        # Fall back to L2 CMH when GEE fails to converge
        return compute_les_cmh(family_df, family, baseline_phi_id)


def compute_les_cmh(df: pd.DataFrame, family: str, baseline_phi_id: str) -> LESResult:
    """
    L2: Cochran-Mantel-Haenszel test.

    Suitable for M_probe >= 2 with multiple variant strata.
    Tests the effect of probe on success after controlling for variant.
    """
    variants = df['variant'].unique()
    phi_ids = df['phi_id'].unique()

    if len(variants) < 2 or len(phi_ids) < 2:
        # Not enough strata or probe variants, use L3 McNemar
        return compute_les_mcnemar(df, family, baseline_phi_id)

    # Collect 2x2 tables for each variant
    contingency_tables = []

    for variant in variants:
        variant_df = df[df['variant'] == variant]

        # Build 2x2 table: [baseline_success, baseline_failure], [probe_success, probe_failure]
        baseline_data = variant_df[variant_df['phi_id'] == baseline_phi_id]
        probe_data = variant_df[variant_df['phi_id'] != baseline_phi_id]

        if len(baseline_data) == 0 or len(probe_data) == 0:
            continue

        baseline_success = (baseline_data['y_pred'] == 1).sum()
        baseline_failure = (baseline_data['y_pred'] == 0).sum()
        probe_success = (probe_data['y_pred'] == 1).sum()
        probe_failure = (probe_data['y_pred'] == 0).sum()

        if baseline_success + baseline_failure > 0 and probe_success + probe_failure > 0:
            contingency_tables.append([
                [baseline_success, baseline_failure],
                [probe_success, probe_failure]
            ])

    if len(contingency_tables) < 2:
        return compute_les_mcnemar(df, family, baseline_phi_id)

    # Compute CMH statistic
    try:
        # Use scipy's chi2_contingency for CMH test
        # Compute expected values and variance for each variant
        total_n = sum(sum(table[0]) + table[1][0] + table[1][1] for table in contingency_tables)

        # Compute aggregated 2x2 table
        sum_baseline_success = sum(table[0][0] for table in contingency_tables)
        sum_baseline_failure = sum(table[0][1] for table in contingency_tables)
        sum_probe_success = sum(table[1][0] for table in contingency_tables)
        sum_probe_failure = sum(table[1][1] for table in contingency_tables)

        # CMH statistic
        if total_n > 0:
            # Compute expected value
            E_probe_success = (sum_probe_success * (sum_baseline_success + sum_baseline_failure)) / total_n

            # Compute variance (Cochran-Mantel-Haenszel correct formula)
            # Var = Σ [n1i * n2i * (n1i + n2i) - (n1i*n2i - n11i*n22i)²/ni] / (ni - 1)²
            # where n1i = a+b, n2i = c+d, n11i = a, n22i = d, ni = a+b+c+d
            var_sum = 0
            for table in contingency_tables:
                a = table[0][0]  # baseline_success
                b = table[0][1]  # baseline_failure  
                c = table[1][0]  # probe_success
                d = table[1][1]  # probe_failure
                
                n1i = a + b  # n1 (baseline total)
                n2i = c + d  # n2 (probe total)
                ni = a + b + c + d  # total in stratum
                
                if ni > 1:
                    # Correct CMH variance formula (Cochran 1954)
                    term = (n1i * n2i * (n1i + n2i) - (a * d - b * c) ** 2 / ni) / (ni - 1) ** 2
                    var_sum += term

            # Simplified CMH statistic
            observed_probe_success = sum_probe_success
            chi2_stat = 0
            if var_sum > 0:
                chi2_stat = (observed_probe_success - E_probe_success) ** 2 / var_sum

            # Compute p-value
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

            # Compute odds ratio (Mantel-Haenszel)
            if sum_baseline_failure * sum_probe_failure > 0:
                or_mh = (sum_baseline_success * sum_probe_failure) / (sum_baseline_failure * sum_probe_success)
                beta = np.log(or_mh) if or_mh > 0 else 0
            else:
                beta = 0
                or_mh = 1.0

            les_beta = abs(beta)

            return LESResult(
                les_beta=les_beta,
                d_logit=les_beta / SIGMA_LOGIT,
                odds_ratio=or_mh,
                beta=beta,
                se=0.0,
                p_value=p_value,
                p_fdr=p_value,
                method="L2_CMH",
                converged=True
            )
    except Exception as e:
        pass

    # Fall back to L3 McNemar when CMH fails
    return compute_les_mcnemar(df, family, baseline_phi_id)


def compute_les_mcnemar(df: pd.DataFrame, family: str, baseline_phi_id: str) -> LESResult:
    """
    L3: McNemar exact test.
    
    Suitable for M_probe = 2 (temporal family: orig vs rev)
    """
    if len(df) < 2:
        return LESResult(
            les_beta=0.0, d_logit=0.0, odds_ratio=1.0,
            beta=0.0, se=0.0, p_value=1.0, p_fdr=1.0,
            method="L3_McNemar", converged=True
        )
    
    # Build paired table
    groups = df['group_id'].unique()
    discordant_01 = 0  # baseline=0, probe=1
    discordant_10 = 0  # baseline=1, probe=0
    
    for g in groups:
        g_data = df[df['group_id'] == g]
        baseline = g_data[g_data['phi_id'] == baseline_phi_id]['y_pred'].values
        probe = g_data[g_data['phi_id'] != baseline_phi_id]['y_pred'].values
        
        if len(baseline) > 0 and len(probe) > 0:
            b, p = baseline[0], probe[0]
            if b == 0 and p == 1:
                discordant_01 += 1
            elif b == 1 and p == 0:
                discordant_10 += 1
    
    # McNemar test
    if discordant_01 + discordant_10 > 0:
        # Use binomial approximation
        n = discordant_01 + discordant_10
        # H0: p = 0.5, i.e., discordant distribution is symmetric
        p_value = stats.binom_test(discordant_01, n, 0.5) if hasattr(stats, 'binom_test') else \
                  stats.binomtest(discordant_01, n, 0.5).pvalue
        
        # Compute odds ratio
        if discordant_10 > 0:
            odds_ratio = discordant_01 / discordant_10
            beta = np.log(odds_ratio) if odds_ratio > 0 else 0
        else:
            odds_ratio = float('inf') if discordant_01 > 0 else 1.0
            beta = np.log(odds_ratio) if odds_ratio != float('inf') and odds_ratio > 0 else 0
    else:
        p_value = 1.0
        odds_ratio = 1.0
        beta = 0.0
    
    # Simplified: take max|beta| = |log(OR)|
    les_beta = abs(beta)
    
    return LESResult(
        les_beta=les_beta,
        d_logit=les_beta / SIGMA_LOGIT,
        odds_ratio=odds_ratio,
        beta=beta,
        se=0.0,  # McNemar does not directly provide SE
        p_value=p_value,
        p_fdr=p_value,  # Assume FDR = p_value (single test)
        method="L3_McNemar",
        converged=True
    )


def compute_les(verdicts_df: pd.DataFrame, model: str = None) -> Dict[str, Dict[str, LESResult]]:
    """
    Main function to compute LES.

    Returns results grouped by model/task/family.

    framing family uses neu as baseline
    temporal family uses orig as baseline
    visual family uses clean as baseline
    
    Note: SSOT §2 requires FDR correction applied across all (model × family) tests
    """
    for model_name, model_df in verdicts_df.groupby('model'):
        model_results = {}

        for task, task_df in model_df.groupby('task'):
            task_results = {}

            for family, family_df in task_df.groupby('family'):
                # Determine baseline phi_id
                if family == 'temporal':
                    baseline = 'orig'
                elif family == 'visual':
                    baseline = 'clean'
                elif family == 'framing':
                    baseline = 'neu'
                else:
                    continue

                # Try L1 (GEE), fall back to L3 (McNemar) on failure
                result = compute_les_gee(family_df, family, baseline)
                task_results[family] = result
                
                # Collect p-values for FDR correction
                all_p_values.append(result.p_value)
                all_test_keys.append((model_name, task, family))

            model_results[task] = task_results

        results[model_name] = model_results

    # Global FDR correction (Benjamini-Hochberg)
    if len(all_p_values) > 1 and STATSMODELS_AVAILABLE:
        try:
            reject, p_fdr, _, _ = multipletests(all_p_values, method='fdr_bh')
            # Update all results
            for i, (model_name, task, family) in enumerate(all_test_keys):
                if model_name in results and task in results[model_name]:
                    results[model_name][task][family].p_fdr = p_fdr[i]
        except Exception:
            pass  # Keep original p-value when FDR fails

    return results


def print_les_results(results: Dict[str, Dict[str, LESResult]]):
    """Print LES results"""
    for model, model_results in results.items():
        print(f"\n{'='*70}")
        print(f"Model: {model}")
        print(f"{'='*70}")
        print(f"{'Task':<12} {'Family':<10} {'LES(β)':>10} {'d_logit':>10} {'OR':>10} {'p_raw':>10} {'p_FDR':>10} {'Method':<12}")
        print("-" * 70)
        
        for task, task_results in model_results.items():
            for family, r in task_results.items():
                print(f"{task:<12} {family:<10} {r.les_beta:>10.3f} {r.d_logit:>10.3f} "
                      f"{r.odds_ratio:>10.2f} {r.p_value:>10.4f} {r.p_fdr:>10.4f} {r.method:<12}")


def export_les_csv(results: Dict[str, Dict[str, LESResult]], output_path: str):
    """Export LES results to CSV"""
    rows = []
    
    for model, model_results in results.items():
        for task, task_results in model_results.items():
            for family, r in task_results.items():
                rows.append({
                    'model': model,
                    'task': task,
                    'family': family,
                    'LES_beta': r.les_beta,
                    'd_logit': r.d_logit,
                    'odds_ratio': r.odds_ratio,
                    'beta': r.beta,
                    'se': r.se,
                    'p_value': r.p_value,
                    'p_fdr': r.p_fdr,
                    'method': r.method,
                    'converged': r.converged
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Exported LES results to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="LES Calculator")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--responses", "-r", help="Responses JSONL path")
    source_group.add_argument("--per-row-csv", help="Unified per-row CSV path")
    parser.add_argument("--model", "-m", help="Filter by model name")
    parser.add_argument("--output-csv", "-o", help="Output CSV path")
    args = parser.parse_args()

    if args.per_row_csv:
        verdicts_df = load_verdicts_for_les_from_per_row(args.per_row_csv)
    else:
        verdicts_df = load_verdicts_for_les(args.responses)

    if verdicts_df.empty:
        print("⚠️  No LES-eligible rows found (all may be Task D with no temporal/visual probes, or all api_error/missing). LES output skipped.")
        print_les_results({})
        if args.output_csv:
            export_les_csv({}, args.output_csv)
        return

    results = compute_les(verdicts_df, args.model)
    
    print_les_results(results)
    
    if args.output_csv:
        export_les_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
