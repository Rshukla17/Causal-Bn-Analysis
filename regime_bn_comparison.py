"""
================================================================================
REGIME-AWARE BAYESIAN NETWORK COMPARISON v3
For 207-obs sample, 19 variables, two-regime split at March 2020
================================================================================
Pre-COVID:  Dec 2004 - Feb 2020 (n~167)
Post-COVID: Mar 2020 - Sep 2025 (n~40)

OUTPUT:
    results/regime_dag_pre_v3.csv
    results/regime_dag_post_v3.csv
    results/regime_similarity_v3.csv
    results/regime_invariant_effects_v3.csv
    results/regime_arcs_pre_only_v3.csv
    results/regime_arcs_post_only_v3.csv
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from causal_bn_analysis import (
        run_granger, run_pcmci_plus, dynotears, var_lingam,
        consensus_arcs, get_var_cols, quantify_effect
    )
except ImportError:
    print("[ERROR] causal_bn_analysis.py must be in the same folder")
    sys.exit(1)


RESULTS_DIR = "./results"
BREAK_DATE = '2020-03-01'


def split_two_regimes(df, break_date=BREAK_DATE):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Month'])
    pre = df[df['Date'] < break_date].reset_index(drop=True)
    post = df[df['Date'] >= break_date].reset_index(drop=True)
    return {'pre': pre, 'post': post}


def regime_consensus(df_regime, var_cols, regime_name):
    n = len(df_regime)
    print(f"\n   Regime '{regime_name}': n = {n}")
    if n < 25:
        print(f"      [SKIP] Too few observations")
        return pd.DataFrame()

    X = df_regime[var_cols].values
    print(f"      Granger...")
    granger_df = run_granger(X, var_cols, max_lag=2, alpha=0.05)
    print(f"      PCMCI+...")
    pcmci_df = run_pcmci_plus(X, var_cols, max_lag=2, alpha=0.05)
    print(f"      DYNOTEARS...")
    W_dy, A_dy = dynotears(X, p_lags=1, lambda_w=0.05, lambda_a=0.05)
    print(f"      VAR-LiNGAM...")
    causal_order, B_inst, B_lags, _ = var_lingam(X, max_lag=1)
    print(f"      Consensus...")
    cons = consensus_arcs(granger_df, pcmci_df, W_dy, A_dy,
                           B_inst, B_lags, var_cols)
    cons['regime'] = regime_name
    print(f"      4/4 arcs: {(cons['n_methods'] == 4).sum()}, "
          f"3/4 arcs: {(cons['n_methods'] == 3).sum()}")
    return cons


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def regime_arc_set(dag, min_methods):
    if dag.empty:
        return set()
    strong = dag[dag['n_methods'] >= min_methods]
    return set(zip(strong['cause'], strong['effect']))


def main():
    print("=" * 76)
    print("REGIME-AWARE BAYESIAN NETWORK COMPARISON v3")
    print("Two-regime split at March 2020")
    print("=" * 76)

    path = f"{RESULTS_DIR}/merged_dataset.csv"
    if not os.path.exists(path):
        print(f"[ERROR] Run causal_bn_analysis.py first")
        return
    df = pd.read_csv(path)
    df['Month'] = pd.to_datetime(df['Month'])
    var_cols = [c for c in get_var_cols() if c in df.columns]
    print(f"\nLoaded n={len(df)} obs, {len(var_cols)} variables")

    print(f"\n[1/4] Splitting at {BREAK_DATE}...")
    regimes = split_two_regimes(df)
    for name, sub in regimes.items():
        print(f"   {name}: n = {len(sub)}, "
              f"{sub['Date'].min().date()} to {sub['Date'].max().date()}")

    print("\n[2/4] Running BN consensus on each regime...")
    regime_dags = {}
    for name, sub in regimes.items():
        cons = regime_consensus(sub, var_cols, name)
        regime_dags[name] = cons
        if not cons.empty:
            cons.to_csv(f"{RESULTS_DIR}/regime_dag_{name}_v3.csv", index=False)

    print("\n[3/4] Cross-regime structural comparison...")

    pre_3 = regime_arc_set(regime_dags['pre'], 3)
    pre_4 = regime_arc_set(regime_dags['pre'], 4)
    post_3 = regime_arc_set(regime_dags['post'], 3)
    post_4 = regime_arc_set(regime_dags['post'], 4)

    sim = pd.DataFrame([{
        'comparison': 'pre vs post',
        'arcs_in_pre_3plus': len(pre_3),
        'arcs_in_post_3plus': len(post_3),
        'shared_3plus': len(pre_3 & post_3),
        'jaccard_3plus': round(jaccard(pre_3, post_3), 3),
        'arcs_in_pre_4only': len(pre_4),
        'arcs_in_post_4only': len(post_4),
        'shared_4only': len(pre_4 & post_4),
        'jaccard_4only': round(jaccard(pre_4, post_4), 3),
        'hamming_distance_3plus': len(pre_3 ^ post_3),
    }])
    print("\n   Structural similarity:")
    print(sim.to_string(index=False))
    sim.to_csv(f"{RESULTS_DIR}/regime_similarity_v3.csv", index=False)

    pre_only = pre_3 - post_3
    post_only = post_3 - pre_3
    invariant = pre_3 & post_3

    print(f"\n   Arcs UNIQUE to pre-COVID ({len(pre_only)}):")
    for c, e in sorted(pre_only):
        print(f"      {c:<18} -> {e}")

    print(f"\n   Arcs UNIQUE to post-COVID ({len(post_only)}):")
    for c, e in sorted(post_only):
        print(f"      {c:<18} -> {e}")

    print(f"\n   Arcs INVARIANT (in both regimes) ({len(invariant)}):")
    for c, e in sorted(invariant):
        print(f"      {c:<18} -> {e}")

    pd.DataFrame([{'cause': c, 'effect': e, 'unique_to': 'pre'}
                   for c, e in pre_only]).to_csv(
        f"{RESULTS_DIR}/regime_arcs_pre_only_v3.csv", index=False)
    pd.DataFrame([{'cause': c, 'effect': e, 'unique_to': 'post'}
                   for c, e in post_only]).to_csv(
        f"{RESULTS_DIR}/regime_arcs_post_only_v3.csv", index=False)
    pd.DataFrame([{'cause': c, 'effect': e}
                   for c, e in invariant]).to_csv(
        f"{RESULTS_DIR}/regime_invariant_arcs_v3.csv", index=False)

    print(f"\n[4/4] Effect size comparison for INVARIANT arcs...")
    if invariant:
        comparison = []
        X_pre = regimes['pre'][var_cols].values
        X_post = regimes['post'][var_cols].values
        for c, e in sorted(invariant):
            try:
                pre_eff = quantify_effect(X_pre, var_cols, c, e)
                post_eff = quantify_effect(X_post, var_cols, c, e)
                comparison.append({
                    'cause': c, 'effect': e,
                    'pre_std_effect': pre_eff['std_effect'],
                    'pre_t': pre_eff['t_stat'],
                    'pre_R2': pre_eff['R2'],
                    'post_std_effect': post_eff['std_effect'],
                    'post_t': post_eff['t_stat'],
                    'post_R2': post_eff['R2'],
                    'change_in_std_effect': round(post_eff['std_effect'] - pre_eff['std_effect'], 3),
                })
            except Exception:
                continue
        comp_df = pd.DataFrame(comparison)
        comp_df = comp_df.sort_values('change_in_std_effect', key=abs, ascending=False)
        print("\n   Top invariant arcs ranked by change in effect size:")
        print(comp_df.head(15).to_string(index=False))
        comp_df.to_csv(f"{RESULTS_DIR}/regime_invariant_effects_v3.csv", index=False)

    print(f"\n{'=' * 76}")
    print(f"Outputs saved to {RESULTS_DIR}/")
    print(f"{'=' * 76}")


if __name__ == "__main__":
    main()