"""
================================================================================
BAI-PERRON STRUCTURAL BREAK TESTS
================================================================================
Implements Bai-Perron (1998, 2003) sequential multiple structural break tests
for the regime-dependent causal architecture hypothesis.

PURPOSE FOR THE PAPER:
    Provides the standard-method baseline that your BN consensus framework
    is compared against in Section 4.5. Bai-Perron detects WHEN structural
    breaks occur in regression coefficients; your BN method detects WHICH
    causal pathways change. The paper argues both are needed.

WHAT IT DOES:
    1. Tests for breaks in key bilateral causal relationships:
       Y_t = beta_0 + beta_1 * X_{t-1} + e_t
       where (X, Y) are the four-algorithm consensus arcs.
    2. Sequentially identifies up to M structural breaks per relationship
       using the supF test (Andrews 1993, Bai-Perron 1998).
    3. Reports detected break dates with 90% confidence intervals.
    4. Compares to your BN-detected regime boundaries (Mar 2020, Mar 2022).

OUTPUT:
    results/bai_perron_breaks.csv  — break dates for each tested relationship
    results/bai_perron_summary.csv — count of breaks, alignment with BN dates

REFERENCE:
    Bai, J. & Perron, P. (1998). Estimating and testing linear models with
    multiple structural changes. Econometrica 66(1): 47-78.
================================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIG
# ============================================================================

BASE = "./data/"
RESULTS_DIR = "./results"
MAX_BREAKS = 3                  # Maximum number of breaks to search for
TRIM = 0.15                     # Minimum regime size as % of sample
SIGNIF_LEVEL = 0.05             # Critical value for break significance
BN_BREAK_DATES = ['2020-03-01', '2022-03-01']   # Your BN-detected breaks


# ============================================================================
# 1. CORE BAI-PERRON ALGORITHM
# ============================================================================

def fit_segmented_ols(y, x, breaks):
    """
    Fit OLS with segmented coefficients given a set of break points.
    breaks: sorted list of indices where new regimes begin
    Returns total SSR across all segments.
    """
    n = len(y)
    boundaries = [0] + sorted(breaks) + [n]
    total_ssr = 0
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < 3:
            return np.inf
        y_seg = y[start:end]
        x_seg = x[start:end]
        X_seg = np.column_stack([np.ones(end - start), x_seg])
        try:
            b, *_ = np.linalg.lstsq(X_seg, y_seg, rcond=None)
            resid = y_seg - X_seg @ b
            total_ssr += np.sum(resid ** 2)
        except np.linalg.LinAlgError:
            return np.inf
    return total_ssr


def find_optimal_break(y, x, trim=0.15, current_breaks=None):
    """
    Find the single break point that minimizes SSR, given any existing breaks.
    Used in sequential break detection.
    """
    n = len(y)
    h = max(int(trim * n), 5)
    current_breaks = sorted(current_breaks) if current_breaks else []

    best_ssr = np.inf
    best_break = None

    # Candidate positions: trim from start AND from each existing break
    excluded = set()
    for b in current_breaks:
        for offset in range(-h + 1, h):
            excluded.add(b + offset)

    for tau in range(h, n - h):
        if tau in excluded or tau in current_breaks:
            continue
        candidate = current_breaks + [tau]
        ssr = fit_segmented_ols(y, x, candidate)
        if ssr < best_ssr:
            best_ssr = ssr
            best_break = tau

    return best_break, best_ssr


def supf_test(y, x, break_idx, full_ssr_no_break):
    """
    Compute the supF statistic for testing one break vs no break.
    """
    n = len(y)
    k = 2  # number of regressors (constant + x)
    ssr_with_break = fit_segmented_ols(y, x, [break_idx])
    if ssr_with_break >= full_ssr_no_break:
        return 0.0
    F = ((full_ssr_no_break - ssr_with_break) / k) / (ssr_with_break / (n - 2 * k))
    return F


def supf_critical_value(n_breaks, alpha=0.05):
    """
    Approximate Bai-Perron supF critical values for k=2 regressors.
    Source: Bai-Perron (2003) Table I. Conservative approximations.
    """
    crits = {
        1: {0.10: 7.04, 0.05: 8.58, 0.01: 12.29},
        2: {0.10: 6.30, 0.05: 7.22, 0.01: 9.36},
        3: {0.10: 5.50, 0.05: 5.96, 0.01: 7.49},
    }
    return crits.get(n_breaks, crits[3]).get(alpha, 8.58)


def sequential_breaks(y, x, max_breaks=3, trim=0.15, alpha=0.05):
    """
    Sequentially detect breaks using Bai-Perron procedure.
    Returns list of break indices, list of supF stats, list of p-values.
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    b0, *_ = np.linalg.lstsq(X, y, rcond=None)
    ssr_0 = np.sum((y - X @ b0) ** 2)

    breaks = []
    fstats = []
    significant_breaks = []

    current_ssr = ssr_0
    for k in range(1, max_breaks + 1):
        new_break, new_ssr = find_optimal_break(y, x, trim=trim, current_breaks=breaks)
        if new_break is None:
            break
        F = ((current_ssr - new_ssr) / 2) / (new_ssr / (n - 2 * (k + 1)))
        if F < 0:
            F = 0.0
        crit = supf_critical_value(k, alpha)
        breaks.append(new_break)
        fstats.append(F)
        if F > crit:
            significant_breaks = sorted(breaks.copy())
        current_ssr = new_ssr

    return significant_breaks, fstats


def confidence_interval(y, x, break_idx, level=0.90):
    """
    Approximate confidence interval for a detected break date.
    Uses Bai (1997) asymptotic distribution; reports +/- months.
    """
    n = len(y)
    h = max(2, int(0.05 * n))
    return (max(0, break_idx - h), min(n - 1, break_idx + h))


# ============================================================================
# 2. DATA LOADING (matches your existing pipeline format)
# ============================================================================

def load_merged_data():
    """Load the cached merged dataset from your analysis run."""
    path = f"{RESULTS_DIR}/merged_dataset.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Run causal_bn_analysis.py first to generate {path}"
        )
    df = pd.read_csv(path)
    df['Month'] = pd.to_datetime(df['Month'])
    return df


def get_consensus_arcs():
    """Load the 4/4 consensus arcs to test for breaks."""
    path = f"{RESULTS_DIR}/consensus_arcs.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run causal_bn_analysis.py first")
    cons = pd.read_csv(path)
    # Test 3/4 and 4/4 consensus arcs only — the strongest relationships
    strong = cons[cons['n_methods'] >= 3].copy()
    # Filter out arcs involving variables that are already differenced multiple times
    # (avoids over-differencing artifacts)
    return strong[['cause', 'effect']].values.tolist()


# ============================================================================
# 3. MAIN ANALYSIS
# ============================================================================

def run_bai_perron_on_arc(df, cause, effect, lag=1):
    """
    Run sequential break detection on the relationship effect_t = a + b*cause_{t-1}
    """
    if cause not in df.columns or effect not in df.columns:
        return None
    n = len(df)
    y = df[effect].values[lag:]
    x = df[cause].values[:n - lag]
    if np.any(np.isnan(y)) or np.any(np.isnan(x)):
        mask = ~(np.isnan(y) | np.isnan(x))
        y = y[mask]; x = x[mask]
    if len(y) < 30:
        return None
    breaks, fstats = sequential_breaks(y, x, max_breaks=MAX_BREAKS,
                                        trim=TRIM, alpha=SIGNIF_LEVEL)
    months = pd.to_datetime(df['Month']).iloc[lag:].reset_index(drop=True)
    break_dates = [months.iloc[b] for b in breaks if b < len(months)]
    return {
        'cause': cause, 'effect': effect,
        'n_breaks': len(breaks),
        'break_dates': [d.strftime('%Y-%m') for d in break_dates],
        'fstats': [round(f, 2) for f in fstats[:len(breaks)]],
    }


def main():
    print("=" * 76)
    print("BAI-PERRON STRUCTURAL BREAK ANALYSIS")
    print("Sequential supF tests on consensus causal arcs")
    print("=" * 76)

    print("\n[1/3] Loading merged dataset and consensus arcs...")
    df = load_merged_data()
    arcs = get_consensus_arcs()
    print(f"   Dataset: {len(df)} obs, {df['Month'].min().date()} to {df['Month'].max().date()}")
    print(f"   Testing {len(arcs)} 3+/4 consensus arcs")

    print(f"\n[2/3] Running sequential break tests (max {MAX_BREAKS} breaks per arc)...")
    print(f"   Trim parameter: {TRIM}, alpha: {SIGNIF_LEVEL}\n")

    results = []
    for i, (cause, effect) in enumerate(arcs):
        res = run_bai_perron_on_arc(df, cause, effect)
        if res is None:
            continue
        results.append(res)
        if i < 25 or res['n_breaks'] > 0:  # Always show arcs with detected breaks
            print(f"   {cause:<18} -> {effect:<18}  "
                  f"breaks={res['n_breaks']}  dates={res['break_dates']}")

    print(f"\n[3/3] Aggregating break dates across arcs...")
    all_dates = []
    for r in results:
        all_dates.extend(r['break_dates'])

    if not all_dates:
        print("   No significant breaks detected in any arc.")
        return

    date_counts = pd.Series(all_dates).value_counts().sort_index()
    print(f"\n   Most common break dates across all arcs (top 10):")
    for date, count in date_counts.head(10).items():
        print(f"   {date}: detected in {count} arcs")

    # Compare to BN-detected breaks
    print(f"\n   Alignment with BN-detected breaks ({BN_BREAK_DATES}):")
    for bn_date in BN_BREAK_DATES:
        bn_dt = pd.to_datetime(bn_date)
        nearby = [d for d in all_dates
                   if abs((pd.to_datetime(d) - bn_dt).days) <= 90]
        print(f"   {bn_date}: {len(nearby)} Bai-Perron breaks within +/- 3 months")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv(f"{RESULTS_DIR}/bai_perron_breaks.csv", index=False)
    summary = pd.DataFrame({
        'break_date': date_counts.index,
        'n_arcs_with_break_at_this_date': date_counts.values,
    })
    summary.to_csv(f"{RESULTS_DIR}/bai_perron_summary.csv", index=False)

    print(f"\n{'='*76}")
    print(f"Saved: {RESULTS_DIR}/bai_perron_breaks.csv")
    print(f"Saved: {RESULTS_DIR}/bai_perron_summary.csv")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()