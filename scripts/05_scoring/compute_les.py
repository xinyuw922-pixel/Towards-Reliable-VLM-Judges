#!/usr/bin/env python3
"""
LES (Leakage Effect Size) Calculator

参考 Stats Spec §2
计算 C1 核心指标：LES (Leakage Effect Size)

三级 fallback：
- L1: GEE Binomial + Exchangeable correlation (首选)
- L2: Cochran-Mantel-Haenszel test
- L3: McNemar exact test (M_probe=2 时使用)
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

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from scripts.exam_schema import parse_verdict as _parse_verdict_ssot

# 尝试导入 statsmodels
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available, using fallback methods only")


# Chinn 2000 标准化效应量常数
SIGMA_LOGIT = np.pi / np.sqrt(3)


@dataclass
class LESResult:
    """LES 计算结果"""
    les_beta: float          # max|β| (log-odds shift)
    d_logit: float          # 标准化效应量
    odds_ratio: float        # exp(β)
    beta: float              # 原始 β 系数
    se: float                # 标准误
    p_value: float           # 原始 p 值
    p_fdr: float            # FDR 校正后 p 值
    method: str             # "L1_GEE" / "L2_CMH" / "L3_McNemar"
    converged: bool         # 是否收敛


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _parse_les_uid(exam_id: str) -> dict:
    """
    Parse exam_id to extract LES-relevant fields.

    Supports two formats:
      MiniGrid: C.<env>.<group_id>.<variant>.<temporal>.<visual>[.<framing>]
      MiniWorld: C-MW.<family>.<template>.<seed>.<variant>.<temporal>.<visual>[.<framing>]
    """
    parts = exam_id.split('.')

    # MiniWorld: C-MW.<family>.<template>.<seed>.<variant>.<temporal>.<visual>[.<framing>]
    if exam_id.startswith('C-MW.') and len(parts) >= 6:
        task = 'C'
        family_or_group = f"{parts[1]}.{parts[2]}.{parts[3]}"  # family.template.seed
        variant = parts[4]
        temporal = parts[5] if len(parts) > 5 else ''
        visual = parts[6] if len(parts) > 6 else ''
        framing = parts[7] if len(parts) > 7 else ''
        return {'task': task, 'group_id': family_or_group, 'variant': variant,
                'temporal': temporal, 'visual': visual, 'framing': framing}

    # MiniGrid: C.<env>.<group_id>.<variant>.<temporal>.<visual>[.<framing>]
    if len(parts) >= 6:
        return {
            'task': parts[1],
            'group_id': parts[2],
            'variant': parts[3],
            'temporal': parts[4],
            'visual': parts[5] if len(parts) > 5 else '',
            'framing': parts[6] if len(parts) > 6 else '',
        }

    return {'task': '', 'group_id': '', 'variant': '', 'temporal': '', 'visual': '', 'framing': ''}


def load_verdicts_for_les(responses_path: str) -> pd.DataFrame:
    """
    加载 LES 所需的数据格式

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
            parsed = _parse_les_uid(exam_id)

            task = parsed['task']
            group_id = parsed['group_id']
            variant = parsed['variant']
            temporal = parsed['temporal']
            visual = parsed['visual']
            framing = parsed['framing']

            if not task:
                continue

            # Determine family
            if framing in ('pos', 'neu', 'neg'):
                family = 'framing'
                phi_id = framing
            elif temporal not in ('orig', ''):
                family = 'temporal'
                phi_id = temporal
            elif visual not in ('clean', ''):
                family = 'visual'
                phi_id = visual
            else:
                family = 'baseline'
                phi_id = 'baseline'

            # Only process non-baseline probes
            if family != 'baseline':
                response = record.get('pred', record.get('raw', record.get('response', '')))
                pred = parse_verdict(response)

                if pred >= 0:
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

    The per-row CSV columns are: uid, task, model, env_task, group_id, template_id,
    seed, variant, framing, temporal, visual, gold, pred, correct, ok, error
    """
    df = pd.read_csv(per_row_csv)
    if df.empty:
        return pd.DataFrame(columns=["model", "task", "group_id", "family", "phi_id", "variant", "y_pred"])

    if "task" not in df.columns:
        return pd.DataFrame(columns=["model", "task", "group_id", "family", "phi_id", "variant", "y_pred"])

    df = df[df["task"].isin(["C", "D"])].copy()

    records = []
    for _, row in df.iterrows():
        temporal = _clean_text(row.get("temporal"))
        visual = _clean_text(row.get("visual"))
        framing = _clean_text(row.get("framing"))
        pred_norm = _clean_text(row.get("pred"))
        response = _clean_text(row.get("raw"))

        if framing in ("pos", "neu", "neg"):
            family = "framing"
            phi_id = framing
        elif temporal not in ("", "orig"):
            family = "temporal"
            phi_id = temporal
        elif visual not in ("", "clean"):
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
                "model": _clean_text(row.get("model")) or "unknown",
                "task": _clean_text(row.get("task")) or "unknown",
                "group_id": _clean_text(row.get("group_id")),
                "family": family,
                "phi_id": phi_id,
                "variant": _clean_text(row.get("variant")),
                "y_pred": pred,
            })

    return pd.DataFrame(records)


def parse_verdict(response: str) -> int:
    """从模型回复中解析 verdict（与 score_exam.py 保持一致）。"""
    ans, mode = _parse_verdict_ssot(response)
    if ans == "Success":
        return 1
    if ans == "Fail":
        return 0
    else:
        return -1


def compute_les_gee(df: pd.DataFrame, family: str, baseline_phi_id: str) -> LESResult:
    """
    L1: GEE Binomial 方法

    拟合混合效应逻辑回归：
    logit(P(y = Success)) = β_0 + β_F · φ_id + (1 | g)
    """
    if not STATSMODELS_AVAILABLE:
        return compute_les_cmh(df, family, baseline_phi_id)

    family_df = df[df['family'] == family].copy()

    if len(family_df) < 10:
        return compute_les_cmh(family_df, family, baseline_phi_id)

    # 构建编码
    all_phi_ids = sorted(family_df['phi_id'].unique())
    cat_order = [baseline_phi_id] + [x for x in all_phi_ids if x != baseline_phi_id]

    try:
        family_df['phi_id'] = pd.Categorical(family_df['phi_id'], categories=cat_order)

        # GEE 模型
        model = smf.gee(
            "y_pred ~ C(phi_id)",
            groups="group_id",
            data=family_df,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable()
        ).fit(disp=False)

        # 提取系数
        beta_coeffs = {}
        se_coeffs = {}
        p_values = {}

        for name in model.params.index:
            if 'phi_id' in name and 'Intercept' not in name:
                # 简化系数名称
                phi_name = name.replace('C(phi_id)[T.', '').replace(']', '')
                beta_coeffs[phi_name] = model.params[name]
                se_coeffs[phi_name] = model.bse[name]
                p_values[phi_name] = model.pvalues[name]

        if not beta_coeffs:
            return compute_les_cmh(family_df, family, baseline_phi_id)

        # FDR 校正
        _, p_fdr, _, _ = multipletests(list(p_values.values()), method='fdr_bh')
        p_fdr_dict = dict(zip(p_values.keys(), p_fdr))

        # 取最大 |β|
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
        # GEE 不收敛时 fallback 到 L2 CMH
        return compute_les_cmh(family_df, family, baseline_phi_id)


def compute_les_cmh(df: pd.DataFrame, family: str, baseline_phi_id: str) -> LESResult:
    """
    L2: Cochran-Mantel-Haenszel test

    适用于 M_probe >= 2 且有多个 variant strata 的情况。
    控制 variant 后检验 probe 对 success 的影响。
    """
    variants = df['variant'].unique()
    phi_ids = df['phi_id'].unique()

    if len(variants) < 2 or len(phi_ids) < 2:
        # 没有足够的 strata 或 probe 变体，使用 L3 McNemar
        return compute_les_mcnemar(df, family, baseline_phi_id)

    # 收集各 variant 的 2x2 表
    contingency_tables = []

    for variant in variants:
        variant_df = df[df['variant'] == variant]

        # 构建 2x2 表: [baseline_success, baseline_failure], [probe_success, probe_failure]
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

    # 计算 CMH 统计量
    try:
        # 使用 scipy 的 chi2_contingency 进行 CMH 检验
        # 对每个 variant 计算期望值和方差
        total_n = sum(sum(table[0]) + table[1][0] + table[1][1] for table in contingency_tables)

        # 计算汇总的 2x2 表
        sum_baseline_success = sum(table[0][0] for table in contingency_tables)
        sum_baseline_failure = sum(table[0][1] for table in contingency_tables)
        sum_probe_success = sum(table[1][0] for table in contingency_tables)
        sum_probe_failure = sum(table[1][1] for table in contingency_tables)

        # CMH 统计量
        if total_n > 0:
            # 计算期望值
            E_probe_success = (sum_probe_success * (sum_baseline_success + sum_baseline_failure)) / total_n

            # 计算方差 (Cochran-Mantel-Haenszel 正确公式)
            # Var = Σ [n1i * n2i * (n1i + n2i) - (n1i*n2i - n11i*n22i)²/ni] / (ni - 1)²
            # 其中 n1i = a+b, n2i = c+d, n11i = a, n22i = d, ni = a+b+c+d
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
                    # 正确的 CMH 方差公式 (Cochran 1954)
                    term = (n1i * n2i * (n1i + n2i) - (a * d - b * c) ** 2 / ni) / (ni - 1) ** 2
                    var_sum += term

            # 简化的 CMH 统计量
            observed_probe_success = sum_probe_success
            chi2_stat = 0
            if var_sum > 0:
                chi2_stat = (observed_probe_success - E_probe_success) ** 2 / var_sum

            # 计算 p 值
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

            # 计算 odds ratio (Mantel-Haenszel)
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

    # CMH 失败，fallback 到 L3 McNemar
    return compute_les_mcnemar(df, family, baseline_phi_id)


def compute_les_mcnemar(df: pd.DataFrame, family: str, baseline_phi_id: str) -> LESResult:
    """
    L3: McNemar exact test
    
    适用于 M_probe = 2 的情况（temporal family: orig vs rev）
    """
    if len(df) < 2:
        return LESResult(
            les_beta=0.0, d_logit=0.0, odds_ratio=1.0,
            beta=0.0, se=0.0, p_value=1.0, p_fdr=1.0,
            method="L3_McNemar", converged=True
        )
    
    # 构建配对表
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
    
    # McNemar 检验
    if discordant_01 + discordant_10 > 0:
        # 使用二项分布近似
        n = discordant_01 + discordant_10
        # H0: p = 0.5，即 discordant 分布对称
        p_value = stats.binom_test(discordant_01, n, 0.5) if hasattr(stats, 'binom_test') else \
                  stats.binomtest(discordant_01, n, 0.5).pvalue
        
        # 计算 odds ratio
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
    
    # 简化：取 max|beta| = |log(OR)|
    les_beta = abs(beta)
    
    return LESResult(
        les_beta=les_beta,
        d_logit=les_beta / SIGMA_LOGIT,
        odds_ratio=odds_ratio,
        beta=beta,
        se=0.0,  # McNemar 不直接提供 SE
        p_value=p_value,
        p_fdr=p_value,  # 假设 FDR = p_value (单次检验)
        method="L3_McNemar",
        converged=True
    )


def compute_les(verdicts_df: pd.DataFrame, model: str = None) -> Dict[str, Dict[str, LESResult]]:
    """
    计算 LES 的主函数

    按模型/任务/family 返回结果

    framing family 使用 neu 作为 baseline
    temporal family 使用 orig 作为 baseline
    visual family 使用 clean 作为 baseline

    注意：SSOT §2 要求 FDR 校正在所有 (model × family) 测试之间应用
    """
    all_p_values: list = []
    all_test_keys: list = []
    results: Dict[str, Dict[str, LESResult]] = {}

    for model_name, model_df in verdicts_df.groupby('model'):
        model_results = {}

        for task, task_df in model_df.groupby('task'):
            task_results = {}

            for family, family_df in task_df.groupby('family'):
                # 确定 baseline phi_id
                if family == 'temporal':
                    baseline = 'orig'
                elif family == 'visual':
                    baseline = 'clean'
                elif family == 'framing':
                    baseline = 'neu'
                else:
                    continue

                # 尝试 L1 (GEE)，失败则 fallback 到 L3 (McNemar)
                result = compute_les_gee(family_df, family, baseline)
                task_results[family] = result
                
                # 收集 p 值用于 FDR 校正
                all_p_values.append(result.p_value)
                all_test_keys.append((model_name, task, family))

            model_results[task] = task_results

        results[model_name] = model_results

    # 全局 FDR 校正 (Benjamini-Hochberg)
    if len(all_p_values) > 1 and STATSMODELS_AVAILABLE:
        try:
            reject, p_fdr, _, _ = multipletests(all_p_values, method='fdr_bh')
            # 更新所有结果
            for i, (model_name, task, family) in enumerate(all_test_keys):
                if model_name in results and task in results[model_name]:
                    results[model_name][task][family].p_fdr = p_fdr[i]
        except Exception:
            pass  # FDR 失败时保持原始 p 值

    return results


def print_les_results(results: Dict[str, Dict[str, LESResult]]):
    """打印 LES 结果"""
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
    """导出 LES 结果为 CSV"""
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
