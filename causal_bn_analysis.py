"""
================================================================================
Causal Inference in Multi-Asset Pricing Models  [v4 — extended sample]
================================================================================
SAMPLE:
    Main analysis: ~250 observations, 2004-2025
    17 variables (no DXY, no Bitcoin, no CreditSpread)
    Bitcoin handled separately via pre_post_bitcoin_comparison.py

CHANGES FROM v3:
    - Sample 2004-2025 (was 2014-2025)
    - DXY removed (only goes back to 2006)
    - CreditSpread removed (only goes back to 2023)
    - Five-regime structural break: pre-crisis, GFC, mid, COVID/QE, tightening
================================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from itertools import permutations
from functools import reduce
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIG
# ============================================================================

BASE = "./data/"

ASSETS = {
    "Gold":        "Gold_monthly.csv",
    "CrudeOil":    "CrudeOil_monthly.csv",
    "Russell3000": "Russell3000_monthly.csv",
    "REIT":        "REIT_monthly.csv",
    "VIX":         "VIX_monthly.csv",
    "Nasdaq":      "Nasdaq_monthly.csv",
    "SP500":       "SP500_monthly.csv",
    "PoundUSD":    "PoundUSD_monthly.csv",
    "EuroUSD":     "EuroUSD_monthly.csv",
}
MACRO = {
    "CPI":          ("CPI.csv",          "CPIAUCSL"),
    "FEDFUNDS":     ("FEDFUNDS.csv",     "FEDFUNDS"),
    "UNRATE":       ("UNRATE.csv",       "UNRATE"),
    "M2":           ("M2.csv",           "M2SL"),
    "PCE":          ("PCE.csv",          "PCE"),
    "INDPRO":       ("INDPRO.csv",       "INDPRO"),
    "ConsumerConf": ("ConsumerConf.csv", "UMCSENT"),
}
SPREADS = {"YieldSpread":  ("YieldSpread.csv",  "T10Y2Y")}
UNCERTAINTY = {"EPU": ("EPU.csv", "USEPUINDXM"), "GPR": ("GPR.csv", "GPR")}


def normalize_to_month_start(date_series):
    dates = pd.to_datetime(date_series, errors='coerce')
    return dates.dt.to_period('M').dt.to_timestamp()


def load_asset(fname, varname, verbose=True):
    path = BASE + fname
    if not os.path.exists(path):
        if verbose: print(f"   [MISSING] {fname}")
        return None
    df = pd.read_csv(path)
    month_col = next((c for c in df.columns if 'month' in c.lower() or 'date' in c.lower()), df.columns[0])
    ret_col = next((c for c in df.columns if 'return' in c.lower() or 'log' in c.lower()),
                    df.columns[1] if len(df.columns) > 1 else None)
    if ret_col is None: return None
    df = df[[month_col, ret_col]].copy()
    df.columns = ["Month", varname]
    df["Month"] = normalize_to_month_start(df["Month"])
    df = df.dropna(subset=["Month"]).drop_duplicates(subset=["Month"])
    if verbose:
        print(f"   [OK] {varname:<18} ({len(df):>4} rows)  {df['Month'].min().date()} to {df['Month'].max().date()}")
    return df


def load_fred(fname, col_name, out_name, verbose=True):
    path = BASE + fname
    if not os.path.exists(path):
        if verbose: print(f"   [MISSING] {fname}")
        return None
    df = pd.read_csv(path)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = col_name if col_name in df.columns else df.columns[1]
    df = df[[date_col, val_col]].copy()
    df.columns = ["Month", out_name]
    df["Month"] = normalize_to_month_start(df["Month"])
    df[out_name] = pd.to_numeric(df[out_name], errors="coerce")
    df = df.dropna(subset=["Month"]).drop_duplicates(subset=["Month"])
    if verbose:
        print(f"   [OK] {out_name:<18} ({len(df):>4} rows)  {df['Month'].min().date()} to {df['Month'].max().date()}")
    return df.dropna()


def build_dataset():
    print("\n   Loading asset CSVs:")
    dfs = []
    for var, fname in ASSETS.items():
        df = load_asset(fname, var)
        if df is not None: dfs.append(df)
    print("\n   Loading macro CSVs:")
    for var, (fname, col) in MACRO.items():
        df = load_fred(fname, col, var)
        if df is not None: dfs.append(df)
    print("\n   Loading spread CSVs:")
    for var, (fname, col) in SPREADS.items():
        df = load_fred(fname, col, var)
        if df is not None:
            df = df.set_index("Month").resample("MS").mean().reset_index()
            df["Month"] = normalize_to_month_start(df["Month"])
            df = df.dropna()
            dfs.append(df)
    print("\n   Loading uncertainty CSVs:")
    for var, (fname, col) in UNCERTAINTY.items():
        df = load_fred(fname, col, var)
        if df is not None: dfs.append(df)

    print(f"\n   Merging {len(dfs)} datasets...")
    all_data = reduce(lambda a, b: pd.merge(a, b, on="Month", how="outer"), dfs)
    all_data = all_data.sort_values("Month").reset_index(drop=True)
    print(f"   After merge (before dropna): {len(all_data)} rows × {len(all_data.columns)} cols")

    print("\n   Non-null counts per column:")
    for col in all_data.columns:
        if col != 'Month':
            non_null = all_data[col].notna().sum()
            print(f"      {col:<20} {non_null:>4} / {len(all_data)} rows")

    for c in list(MACRO.keys()) + list(UNCERTAINTY.keys()):
        if c in all_data.columns:
            all_data[c + "_chg"] = np.log(all_data[c] / all_data[c].shift(1))
    for c in SPREADS.keys():
        if c in all_data.columns:
            all_data[c + "_chg"] = all_data[c].diff()

    final = all_data.dropna().reset_index(drop=True)
    print(f"\n   Final dataset after dropna: {len(final)} rows")
    return final


def get_var_cols():
    return [
        "Gold", "CrudeOil", "Russell3000", "REIT",
        "VIX", "Nasdaq", "SP500",
        "PoundUSD", "EuroUSD",
        "CPI_chg", "FEDFUNDS_chg", "UNRATE_chg", "M2_chg",
        "PCE_chg", "INDPRO_chg", "ConsumerConf_chg",
        "YieldSpread_chg",
        "EPU_chg", "GPR_chg",
    ]


def adf_test(x):
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 20: return np.nan, np.nan
    dx = np.diff(x); x_lag = x[:-1]
    if len(dx) > 1:
        dx_lag = dx[:-1]; y = dx[1:]
        X = np.column_stack([np.ones(len(y)), x_lag[1:], dx_lag])
    else:
        y = dx; X = np.column_stack([np.ones(len(y)), x_lag])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    sigma2 = np.sum(resid**2) / (len(y) - X.shape[1])
    se = np.sqrt(sigma2 * np.diag(np.linalg.inv(X.T @ X)))
    t_stat = b[1] / se[1]
    if t_stat < -3.43: p = 0.01
    elif t_stat < -2.86: p = 0.05
    elif t_stat < -2.57: p = 0.10
    else: p = 0.20
    return round(t_stat, 3), p


def stationarity_check(X, var_names):
    print("\n   ADF test results:")
    print(f"   {'Variable':<20} {'ADF stat':>10} {'p-value':>10} {'Stationary?':>12}")
    print("   " + "-" * 55)
    out = []
    for i, name in enumerate(var_names):
        stat, p = adf_test(X[:, i])
        is_stat = "YES" if not np.isnan(p) and p <= 0.05 else "NO"
        s = f"{stat:.3f}" if not np.isnan(stat) else "NaN"
        ps = f"{p:.3f}" if not np.isnan(p) else "NaN"
        print(f"   {name:<20} {s:>10} {ps:>10} {is_stat:>12}")
        out.append({'variable': name, 'adf_stat': stat, 'p_value': p,
                    'stationary': is_stat == 'YES'})
    return pd.DataFrame(out)


def granger_test(y, x, max_lag=3):
    n = len(y); results = {}
    for lag in range(1, max_lag + 1):
        Y = y[lag:]; n2 = len(Y)
        Xr = np.column_stack([np.ones(n2)] + [y[lag-k:n-k] for k in range(1, lag+1)])
        Xu = np.column_stack([Xr] + [x[lag-k:n-k] for k in range(1, lag+1)])
        br, *_ = np.linalg.lstsq(Xr, Y, rcond=None)
        bu, *_ = np.linalg.lstsq(Xu, Y, rcond=None)
        rss_r = np.sum((Y - Xr @ br) ** 2)
        rss_u = np.sum((Y - Xu @ bu) ** 2)
        df1, df2 = lag, n2 - Xu.shape[1]
        if df2 <= 0 or rss_u <= 0: continue
        F = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p = 1 - stats.f.cdf(F, df1, df2)
        results[lag] = (F, p)
    return results


def run_granger(X, var_names, max_lag=3, alpha=0.05):
    df_num = pd.DataFrame(X, columns=var_names)
    sig = []
    for cause, effect in permutations(var_names, 2):
        r = granger_test(df_num[effect].values, df_num[cause].values, max_lag=max_lag)
        for lag, (F, p) in r.items():
            if p < alpha:
                sig.append({'cause': cause, 'effect': effect, 'lag': lag,
                            'F': round(F, 3), 'pval': round(p, 5)})
    seen = {}
    for row in sig:
        key = (row['cause'], row['effect'])
        if key not in seen or row['pval'] < seen[key]['pval']:
            seen[key] = row
    return pd.DataFrame(list(seen.values())).sort_values('pval').reset_index(drop=True)


def partial_corr(X_mat, x_idx, y_idx, cond_idx):
    if len(cond_idx) == 0:
        r, p = stats.pearsonr(X_mat[:, x_idx], X_mat[:, y_idx])
        return r, p
    Z = X_mat[:, cond_idx]
    Z = np.column_stack([np.ones(len(Z)), Z])
    def resid(v):
        b, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ b
    rx = resid(X_mat[:, x_idx])
    ry = resid(X_mat[:, y_idx])
    return stats.pearsonr(rx, ry)


def run_pcmci_plus(X, var_names, max_lag=2, alpha=0.05):
    T, d = X.shape
    current = X[max_lag:]
    lagged = [X[max_lag-k:T-k] for k in range(1, max_lag+1)]
    combined = np.hstack([current] + lagged)
    results = []
    for j in range(d):
        for lag in range(1, max_lag + 1):
            for i in range(d):
                src_idx = lag * d + i
                tgt_idx = j
                cond = [lag * d + j] if (lag * d + j) != src_idx else []
                r, pval = partial_corr(combined, src_idx, tgt_idx, cond)
                if pval < alpha:
                    results.append({'cause': var_names[i], 'effect': var_names[j],
                                     'lag': lag, 'r': round(r, 4), 'pval': round(pval, 5)})
    return pd.DataFrame(results).sort_values('pval').reset_index(drop=True)


def dynotears(X, p_lags=1, lambda_w=0.05, lambda_a=0.05,
              max_iter=300, lr=0.001, tol=1e-6, edge_thresh=0.03):
    T, d = X.shape
    Y = X[p_lags:]
    Z = np.hstack([X[p_lags-k-1:T-k-1] for k in range(p_lags)])
    T2 = Y.shape[0]
    Y_n = Y / (Y.std(axis=0) + 1e-10)
    Z_n = Z / (Z.std(axis=0) + 1e-10)
    W = np.zeros((d, d))
    A = np.zeros((p_lags * d, d))
    def grad_acyclicity(W):
        M = np.eye(d) + W * W / d
        E = np.linalg.matrix_power(M, d - 1)
        return E.T * (2 * W / d)
    rho = 1.0
    for _ in range(max_iter):
        R = Y_n - Y_n @ W - Z_n @ A
        g_W = -(1/T2) * Y_n.T @ R + lambda_w * np.sign(W) + rho * grad_acyclicity(W)
        g_A = -(1/T2) * Z_n.T @ R + lambda_a * np.sign(A)
        np.fill_diagonal(g_W, 0)
        W_new = W - lr * g_W
        A_new = A - lr * g_A
        np.fill_diagonal(W_new, 0)
        W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - lambda_w * lr, 0)
        A_new = np.sign(A_new) * np.maximum(np.abs(A_new) - lambda_a * lr, 0)
        if np.max(np.abs(W_new - W)) + np.max(np.abs(A_new - A)) < tol:
            break
        W, A = W_new, A_new
    W[np.abs(W) < edge_thresh] = 0
    A[np.abs(A) < edge_thresh] = 0
    return W, A


def var_lingam(X, max_lag=1):
    T, d = X.shape
    Y = X[max_lag:]
    Z = np.hstack([X[max_lag-k:T-k] for k in range(1, max_lag+1)])
    Z_aug = np.column_stack([np.ones(len(Y)), Z])
    B_hat = np.linalg.lstsq(Z_aug, Y, rcond=None)[0]
    B_lags = B_hat[1:]
    residuals = Y - Z_aug @ B_hat
    def negentropy(v): return abs(stats.kurtosis(v, fisher=True))
    remaining = list(range(d)); causal_order = []
    resids = residuals.copy()
    for _ in range(d):
        if not remaining: break
        sub = resids[:, remaining]
        scores = [negentropy(sub[:, j]) for j in range(sub.shape[1])]
        best_local = int(np.argmax(scores))
        best_global = remaining[best_local]
        causal_order.append(best_global)
        x_best = resids[:, best_global]
        for j in remaining:
            if j != best_global:
                reg = np.column_stack([np.ones(len(x_best)), x_best])
                b, *_ = np.linalg.lstsq(reg, resids[:, j], rcond=None)
                resids[:, j] = resids[:, j] - reg @ b
        remaining.remove(best_global)
    B_inst = np.zeros((d, d))
    for k, var in enumerate(causal_order[1:], 1):
        predecessors = causal_order[:k]
        if predecessors:
            X_pred = np.column_stack(
                [np.ones(len(residuals))] + [residuals[:, p] for p in predecessors])
            b, *_ = np.linalg.lstsq(X_pred, residuals[:, var], rcond=None)
            for idx, pred in enumerate(predecessors):
                if abs(b[idx + 1]) > 0.02:
                    B_inst[var, pred] = b[idx + 1]
    return causal_order, B_inst, B_lags, residuals


def consensus_arcs(granger_df, pcmci_df, dy_W, dy_A, lingam_inst, lingam_lag,
                   var_names, dy_thresh=0.05, lingam_thresh=0.1):
    granger_edges = set(zip(granger_df['cause'], granger_df['effect']))
    pcmci_edges = set(zip(pcmci_df['cause'], pcmci_df['effect']))
    d = len(var_names)
    dy_w_edges = set()
    rows, cols = np.where(np.abs(dy_W) > dy_thresh)
    for r, c in zip(rows, cols):
        if r != c: dy_w_edges.add((var_names[c], var_names[r]))
    dy_a_edges = set()
    rows, cols = np.where(np.abs(dy_A[:d, :]) > dy_thresh - 0.01)
    for r, c in zip(rows, cols):
        dy_a_edges.add((var_names[c], var_names[r]))
    dy_edges = dy_w_edges | dy_a_edges
    lingam_edges_inst = set()
    rows, cols = np.where(np.abs(lingam_inst) > lingam_thresh)
    for r, c in zip(rows, cols):
        if r != c: lingam_edges_inst.add((var_names[c], var_names[r]))
    lingam_edges_lag = set()
    rows, cols = np.where(np.abs(lingam_lag[:d, :]) > lingam_thresh)
    for r, c in zip(rows, cols):
        lingam_edges_lag.add((var_names[c], var_names[r]))
    lingam_edges = lingam_edges_inst | lingam_edges_lag
    all_pairs = granger_edges | pcmci_edges | dy_edges | lingam_edges
    out = []
    for pair in all_pairs:
        if pair[0] == pair[1]: continue
        score = sum([pair in granger_edges, pair in pcmci_edges,
                     pair in dy_edges, pair in lingam_edges])
        out.append({'cause': pair[0], 'effect': pair[1], 'n_methods': score,
                    'granger': pair in granger_edges, 'pcmci': pair in pcmci_edges,
                    'dynotears': pair in dy_edges, 'lingam': pair in lingam_edges})
    return pd.DataFrame(out).sort_values(['n_methods', 'cause'],
                                          ascending=[False, True]).reset_index(drop=True)


def quantify_effect(X, var_names, cause, effect, lag=1):
    n = len(X)
    y = X[lag:, var_names.index(effect)]
    x = X[:n - lag, var_names.index(cause)]
    Xc = np.column_stack([np.ones(n - lag), x])
    b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ b
    sigma2 = np.sum(resid ** 2) / (n - lag - 2)
    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    t_stat = b[1] / se[1]
    r2 = 1 - np.var(resid) / np.var(y)
    std_X = X.std(axis=0)
    std_effect = b[1] * std_X[var_names.index(cause)] / std_X[var_names.index(effect)]
    return {'cause': cause, 'effect': effect, 'lag': lag,
            'coef': round(b[1], 5), 't_stat': round(t_stat, 2),
            'R2': round(r2, 3), 'std_effect': round(std_effect, 3)}


def regime_analysis(df, regime_var, asset_cols, n_quantiles=3):
    labels = ['Low', 'Mid', 'High'] if n_quantiles == 3 else None
    tertile = pd.qcut(df[regime_var], q=n_quantiles, labels=labels)
    return df.assign(regime=tertile).groupby('regime', observed=True)[asset_cols].mean()


def five_regime_break(df, var_names, arcs_to_test):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Month"])
    pre_crisis = df[df["Date"] < "2008-09-01"]
    gfc = df[(df["Date"] >= "2008-09-01") & (df["Date"] < "2015-01-01")]
    mid = df[(df["Date"] >= "2015-01-01") & (df["Date"] < "2020-03-01")]
    qe = df[(df["Date"] >= "2020-03-01") & (df["Date"] < "2022-03-01")]
    tight = df[df["Date"] >= "2022-03-01"]
    print(f"\n   Sub-sample sizes:")
    print(f"   Pre-crisis (2004-Aug 2008):    n = {len(pre_crisis)}")
    print(f"   GFC era    (Sep 2008-2014):    n = {len(gfc)}")
    print(f"   Mid-cycle  (2015-Feb 2020):    n = {len(mid)}")
    print(f"   COVID/QE   (Mar 2020-Feb 2022): n = {len(qe)}")
    print(f"   Tightening (Mar 2022 onwards): n = {len(tight)}")

    def quick_g(X_, cause, effect, lag=1):
        ci = var_names.index(cause); ei = var_names.index(effect)
        if X_.shape[0] < 5: return 0.0, 1.0
        y = X_[:, ei]; x = X_[:, ci]
        Y = y[lag:]; n2 = len(Y)
        Xr = np.column_stack([np.ones(n2), y[:n2]])
        Xu = np.column_stack([Xr, x[:n2]])
        try:
            br, *_ = np.linalg.lstsq(Xr, Y, rcond=None)
            bu, *_ = np.linalg.lstsq(Xu, Y, rcond=None)
            rss_r = np.sum((Y - Xr @ br) ** 2)
            rss_u = np.sum((Y - Xu @ bu) ** 2)
            if n2 - 3 <= 0 or rss_u <= 0: return 0.0, 1.0
            F = ((rss_r - rss_u) / 1) / (rss_u / (n2 - 3))
            return round(F, 2), round(1 - stats.f.cdf(F, 1, n2 - 3), 4)
        except Exception:
            return 0.0, 1.0

    rows = []
    for cause, effect in arcs_to_test:
        f_pre, p_pre = quick_g(pre_crisis[var_names].values, cause, effect)
        f_gfc, p_gfc = quick_g(gfc[var_names].values, cause, effect)
        f_mid, p_mid = quick_g(mid[var_names].values, cause, effect)
        f_qe, p_qe = quick_g(qe[var_names].values, cause, effect)
        f_t, p_t = quick_g(tight[var_names].values, cause, effect)
        rows.append({'cause': cause, 'effect': effect,
                     'pre_F': f_pre, 'gfc_F': f_gfc, 'mid_F': f_mid,
                     'qe_F': f_qe, 'tight_F': f_t,
                     'pre_p': p_pre, 'gfc_p': p_gfc, 'mid_p': p_mid,
                     'qe_p': p_qe, 'tight_p': p_t})
    return pd.DataFrame(rows)


def main():
    print("=" * 76)
    print("CAUSAL DISCOVERY PIPELINE v4 — Extended sample 2004-2025")
    print("17 variables (no DXY, no Bitcoin) | 5 monetary regimes | n~250")
    print("=" * 76)

    print("\n[1/10] Loading data from ./data/ subfolder...")
    df = build_dataset()
    var_cols = [c for c in get_var_cols() if c in df.columns]
    X = df[var_cols].values
    n, d = X.shape
    print(f"\n   FINAL Dataset: {n} observations × {d} variables")
    print(f"   Period: {df['Month'].min().date()} to {df['Month'].max().date()}")

    if n < 100:
        print(f"\n   [ABORT] Only {n} observations after merge.")
        return

    print("\n[2/10] Stationarity tests (ADF)...")
    stat_df = stationarity_check(X, var_cols)

    print("\n[3/10] Granger causality...")
    granger_df = run_granger(X, var_cols, max_lag=3, alpha=0.05)
    print(f"   Found {len(granger_df)} significant pairs.")

    print("\n[4/10] PCMCI+...")
    pcmci_df = run_pcmci_plus(X, var_cols, max_lag=2, alpha=0.05)
    print(f"   Found {len(pcmci_df)} significant links.")

    print("\n[5/10] DYNOTEARS...")
    W_dy, A_dy = dynotears(X, p_lags=1, lambda_w=0.04, lambda_a=0.04)
    print(f"   Found {int((W_dy != 0).sum())} W edges + {int((A_dy != 0).sum())} A edges.")

    print("\n[6/10] VAR-LiNGAM...")
    causal_order, B_inst, B_lags, _ = var_lingam(X, max_lag=1)
    print(f"   Top 5 exogenous: {[var_cols[i] for i in causal_order[:5]]}")

    print("\n[7/10] Consensus...")
    cons_df = consensus_arcs(granger_df, pcmci_df, W_dy, A_dy,
                              B_inst, B_lags, var_cols)
    print(f"   4/4 arcs: {(cons_df['n_methods']==4).sum()}")
    print(f"   3/4 arcs: {(cons_df['n_methods']==3).sum()}")

    print("\n[8/10] Effect sizes (3+/4 consensus)...")
    top_arcs = cons_df[cons_df['n_methods'] >= 3][['cause','effect']].values.tolist()
    effects = pd.DataFrame([quantify_effect(X, var_cols, c, e) for c, e in top_arcs])
    if not effects.empty:
        effects = effects.sort_values('std_effect', key=abs, ascending=False)
        print(effects.head(20).to_string(index=False))

    print("\n[9/10] Regime analysis (EPU and GPR tertiles)...")
    asset_list = [a for a in ["Gold","CrudeOil","Russell3000","REIT"] if a in df.columns]
    if "EPU" in df.columns:
        print("\n   By EPU regime (mean monthly returns × 100):")
        print((regime_analysis(df, 'EPU', asset_list) * 100).round(2).to_string())
    if "GPR" in df.columns:
        print("\n   By GPR regime:")
        print((regime_analysis(df, 'GPR', asset_list) * 100).round(2).to_string())

    print("\n[10/10] Five-regime structural break...")
    arcs_to_test = []
    for cause in ["CrudeOil","EPU_chg","Russell3000","M2_chg","GPR_chg",
                  "FEDFUNDS_chg","YieldSpread_chg","VIX","SP500"]:
        for effect in ["PCE_chg","FEDFUNDS_chg","UNRATE_chg","INDPRO_chg",
                       "CPI_chg","Russell3000"]:
            if cause in var_cols and effect in var_cols and cause != effect:
                arcs_to_test.append((cause, effect))
    arcs_to_test = list(set(arcs_to_test))[:12]
    break_df = five_regime_break(df, var_cols, arcs_to_test)
    print("\n   " + break_df.to_string(index=False).replace("\n", "\n   "))

    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)
    granger_df.to_csv(f"{out_dir}/granger_results.csv", index=False)
    pcmci_df.to_csv(f"{out_dir}/pcmci_results.csv", index=False)
    cons_df.to_csv(f"{out_dir}/consensus_arcs.csv", index=False)
    effects.to_csv(f"{out_dir}/effect_sizes.csv", index=False)
    break_df.to_csv(f"{out_dir}/structural_break_5regime.csv", index=False)
    stat_df.to_csv(f"{out_dir}/stationarity_tests.csv", index=False)
    np.savez(f"{out_dir}/dynotears_matrices.npz", W=W_dy, A=A_dy)
    df.to_csv(f"{out_dir}/merged_dataset.csv", index=False)

    print(f"\n{'='*76}")
    print(f"Pipeline complete. Results in {out_dir}/")
    print(f"NEXT: run pre_post_bitcoin_comparison.py for Bitcoin sub-analysis")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()