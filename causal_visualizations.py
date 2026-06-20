"""
================================================================================
COMPLETE VISUALIZATION SUITE — n=207 DATASET
================================================================================
Generates all figures for the IRFA paper, fresh on the current 207-observation
2004-2025 sample with 17 variables (no Bitcoin, no DXY in main analysis).

FIGURES PRODUCED:
  Figure 1.  Time series of all variables (with regime shading)
  Figure 2.  Full-sample consensus DAG (paper Figure 1)
  Figure 3.  Granger F-statistic heatmap
  Figure 4.  Top effect sizes bar chart (paper Table 1 visualized)
  Figure 5.  VAR-LiNGAM causal ordering
  Figure 6.  ADF stationarity tests
  Figure 7.  Regime DAGs side-by-side (paper Figure 2) [READS regime_dag_pre/post_v3.csv]
  Figure 8.  Invariant arc effect-size slopes (paper Figure 3) [READS regime_invariant_effects_v3.csv]
  Figure 9.  Bai-Perron break date histogram (paper Figure 4) [READS bai_perron_breaks.csv]

USAGE:
  1. DELETE old figures: Remove-Item -Recurse -Force ./figures
  2. python causal_visualizations.py

Reads cached results from ./results/. Run these analyses first if not already:
  python causal_bn_analysis.py
  python regime_bn_comparison_v3.py
  python bai_perron_breaks.py
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import warnings
warnings.filterwarnings('ignore')

from causal_bn_analysis import (
    build_dataset, get_var_cols,
    run_granger, run_pcmci_plus, dynotears, var_lingam,
    consensus_arcs, quantify_effect,
    stationarity_check, partial_corr,
)


# ============================================================================
# STYLING
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.4,
})

PAL = {
    'macro': '#D85A30', 'asset': '#378ADD', 'equity': '#1D9E75',
    'spread': '#BA7517', 'uncert': '#7F77DD',
    'pos': '#1D9E75', 'neg': '#E24B4A', 'neutral': '#888780',
    'pre': '#7AAEDC', 'post': '#E24B4A', 'invariant': '#444444',
}


def vcolor(v):
    if v in ["Russell3000", "REIT", "SP500", "Nasdaq"]: return PAL['equity']
    if v in ["Gold", "CrudeOil", "PoundUSD", "EuroUSD"]: return PAL['asset']
    if v in ["VIX", "EPU", "GPR", "EPU_chg", "GPR_chg"]: return PAL['uncert']
    if "Spread" in v: return PAL['spread']
    return PAL['macro']


def short(v):
    return (v.replace('_chg', '').replace('Russell3000', 'R3000')
            .replace('YieldSpread', 'Yield').replace('ConsumerConf', 'ConsConf')
            .replace('FEDFUNDS', 'FedFds'))


def save(fig, name):
    fig.savefig(f'figures/{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
    fig.savefig(f'figures/{name}.pdf', bbox_inches='tight', pad_inches=0.4)
    plt.close(fig)
    print(f"   Saved: figures/{name}.[png|pdf]")


# ============================================================================
# FIGURE 1: Time series with regime shading
# ============================================================================
def fig1_timeseries(df, var_cols):
    df = df.copy()
    df['Month'] = pd.to_datetime(df['Month'])
    n_vars = len(var_cols)
    n_cols = 4
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.2 * n_rows))
    axes = axes.flatten()

    covid_start = pd.Timestamp('2020-03-01')
    covid_end = pd.Timestamp('2022-03-01')
    tight_end = pd.Timestamp('2025-09-01')

    for i, var in enumerate(var_cols):
        ax = axes[i]
        if var not in df.columns:
            ax.axis('off'); continue
        ax.plot(df['Month'], df[var], color=vcolor(var), linewidth=0.9)
        ax.axvspan(covid_start, covid_end, alpha=0.12, color='#E24B4A')
        ax.axvspan(covid_end, tight_end, alpha=0.10, color='#BA7517')
        ax.axhline(0, color='black', linewidth=0.4, alpha=0.5)
        ax.set_title(short(var), fontsize=10)
        ax.tick_params(labelsize=8)
    for j in range(n_vars, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Time series of all variables (red = COVID/QE, orange = tightening)',
                  fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 2: Full-sample consensus DAG (paper Figure 1)
# ============================================================================
def fig2_consensus_dag(cons_df, var_cols):
    fig, ax = plt.subplots(figsize=(16, 11))

    macro = [v for v in var_cols if v.endswith('_chg') and v not in
             ['EPU_chg', 'GPR_chg', 'YieldSpread_chg']]
    spread = [v for v in var_cols if v == 'YieldSpread_chg']
    uncert = [v for v in var_cols if v in ['EPU_chg', 'GPR_chg', 'VIX']]
    equity = [v for v in var_cols if v in ['Russell3000', 'REIT', 'SP500', 'Nasdaq']]
    asset = [v for v in var_cols if v in ['Gold', 'CrudeOil', 'PoundUSD', 'EuroUSD']]

    pos = {}
    for i, v in enumerate(macro):   pos[v] = (-5, 5 - i * 0.95)
    for i, v in enumerate(spread):  pos[v] = (-2.5, 0.5)
    for i, v in enumerate(uncert):  pos[v] = (-1 + i * 1.8, 6.0)
    for i, v in enumerate(equity):  pos[v] = (4, 4.5 - i * 1.5)
    for i, v in enumerate(asset):   pos[v] = (7, 4.5 - i * 1.5)

    # Nodes
    for var in var_cols:
        if var not in pos: continue
        x, y = pos[var]
        c = Circle((x, y), 0.45, color=vcolor(var), alpha=0.9, zorder=3,
                    edgecolor='black', linewidth=1)
        ax.add_patch(c)
        ax.text(x, y, short(var), ha='center', va='center', fontsize=8,
                fontweight='bold', color='white', zorder=4)

    # Arcs
    strong = cons_df[cons_df['n_methods'] >= 3]
    for _, row in strong.iterrows():
        if row['cause'] not in pos or row['effect'] not in pos: continue
        x1, y1 = pos[row['cause']]
        x2, y2 = pos[row['effect']]
        if row['n_methods'] == 4:
            color = 'black'; lw = 1.8; alpha = 0.8
        else:
            color = '#888780'; lw = 1.0; alpha = 0.45
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle='->', mutation_scale=15,
                                  color=color, linewidth=lw, alpha=alpha,
                                  zorder=2, connectionstyle="arc3,rad=0.1")
        ax.add_patch(arrow)

    ax.set_xlim(-7, 9)
    ax.set_ylim(-2.5, 7)
    ax.set_aspect('equal'); ax.axis('off')

    n4 = (cons_df['n_methods'] == 4).sum()
    n3 = (cons_df['n_methods'] == 3).sum()
    ax.set_title(f'Full-sample consensus DAG (n=207, 17 variables).\n'
                  f'{n4} arcs at 4/4 consensus (black, bold), {n3} arcs at 3/4 consensus (gray, thin).',
                  fontsize=14, pad=15)

    # Legend
    legend_items = [
        (PAL['macro'], 'Macro variables'),
        (PAL['equity'], 'Equity / REIT'),
        (PAL['asset'], 'Assets / FX'),
        (PAL['spread'], 'Yield spread'),
        (PAL['uncert'], 'Uncertainty / VIX'),
    ]
    for i, (c, lab) in enumerate(legend_items):
        ax.scatter(-6.5 + i * 2.9, -2, s=200, color=c, edgecolor='black', zorder=5)
        ax.text(-6.2 + i * 2.9, -2, lab, fontsize=11, va='center')

    return fig


# ============================================================================
# FIGURE 3: Granger F heatmap
# ============================================================================
def fig3_granger_heatmap(granger_df, var_cols):
    d = len(var_cols)
    F_mat = np.zeros((d, d))
    p_mat = np.ones((d, d))
    for _, row in granger_df.iterrows():
        if row['cause'] in var_cols and row['effect'] in var_cols:
            i = var_cols.index(row['effect'])
            j = var_cols.index(row['cause'])
            F_mat[i, j] = row['F']
            p_mat[i, j] = row['pval']

    # Show only significant
    F_show = np.where(p_mat < 0.05, F_mat, np.nan)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(F_show, cmap='YlOrRd', vmin=0, vmax=50, aspect='auto')

    ax.set_xticks(range(d)); ax.set_xticklabels(var_cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(d)); ax.set_yticklabels(var_cols, fontsize=9)
    ax.set_xlabel('Cause'); ax.set_ylabel('Effect')

    for i in range(d):
        for j in range(d):
            if not np.isnan(F_show[i, j]) and F_show[i, j] >= 5:
                ax.text(j, i, f'{F_show[i, j]:.0f}', ha='center', va='center',
                        fontsize=7, color='black' if F_show[i, j] < 25 else 'white')

    cbar = plt.colorbar(im, ax=ax, label='Granger F-statistic')
    ax.set_title('Granger F-statistics (significant pairs, p < 0.05)',
                  fontsize=12, pad=10)
    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 4: Top effect sizes
# ============================================================================
def fig4_effect_sizes(effects_df):
    if effects_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'No effect data', ha='center')
        return fig

    df = effects_df.head(20).copy()
    df['label'] = df['cause'] + ' → ' + df['effect']
    df = df.sort_values('std_effect')
    colors = [PAL['pos'] if x > 0 else PAL['neg'] for x in df['std_effect']]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(df['label'], df['std_effect'], color=colors, edgecolor='white', linewidth=0.7)
    ax.axvline(0, color='black', linewidth=0.6)
    ax.set_xlabel('Standardized Effect (σ change in effect per 1σ change in cause)')
    ax.set_title('Effect Sizes for Top Consensus Arcs (lag = 1 month)',
                  fontsize=12, pad=10)

    for i, v in enumerate(df['std_effect']):
        ax.text(v + (0.02 if v > 0 else -0.02), i, f'{v:+.3f}',
                va='center', ha='left' if v > 0 else 'right', fontsize=8)

    ax.set_xlim(-1, 1)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 5: VAR-LiNGAM ordering
# ============================================================================
def fig5_lingam_ordering(causal_order, var_cols):
    ordered = [var_cols[i] for i in causal_order]
    n = len(ordered)
    fig, ax = plt.subplots(figsize=(10, n * 0.4 + 1))

    cmap = plt.get_cmap('RdYlGn_r')
    for i, var in enumerate(ordered):
        color = cmap(i / max(n - 1, 1))
        ax.barh(n - 1 - i, 1, color=color, edgecolor='white', alpha=0.85)
        ax.text(0.5, n - 1 - i, f'{i+1}. {short(var)}',
                ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 1); ax.set_ylim(-0.5, n - 0.5)
    ax.axis('off')
    ax.text(-0.02, n - 0.5, 'MOST EXOGENOUS\n(causes others)',
             fontsize=9, ha='right', va='top', color=PAL['neg'], fontweight='bold')
    ax.text(-0.02, -0.5, 'MOST ENDOGENOUS\n(caused by others)',
             fontsize=9, ha='right', va='bottom', color=PAL['equity'], fontweight='bold')
    ax.set_title('VAR-LiNGAM Causal Ordering\n(non-Gaussianity-based)',
                  fontsize=12, pad=10)
    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 6: ADF stationarity
# ============================================================================
def fig6_stationarity(stat_df):
    df = stat_df.sort_values('adf_stat').copy()
    df['label'] = df['variable'].apply(short)
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ['#1D9E75' if x < -2.86 else '#E24B4A' for x in df['adf_stat']]
    ax.barh(df['label'], df['adf_stat'], color=colors, edgecolor='white', linewidth=0.6)
    ax.axvline(-2.86, color='#888', linestyle='--', linewidth=1.0, label='ADF 5% critical (-2.86)')
    ax.axvline(-3.43, color='red', linestyle='--', linewidth=1.0, label='ADF 1% critical (-3.43)')
    ax.set_xlabel('ADF Test Statistic (more negative = more stationary)')
    ax.set_title('Augmented Dickey-Fuller Stationarity Tests', fontsize=12, pad=10)
    for i, v in enumerate(df['adf_stat']):
        ax.text(v - 0.1, i, f'{v:.2f}', va='center', ha='right', fontsize=7)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 7: Regime DAGs side by side (paper Fig 2)
# ============================================================================
def fig7_regime_dags(pre_dag, post_dag, var_cols, jaccard_val):
    # 16x9 (not 22x22): less downscaling in the paper => labels stay legible
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    macro = [v for v in var_cols if v.endswith('_chg') and v not in
             ['EPU_chg', 'GPR_chg', 'YieldSpread_chg']]
    spread = [v for v in var_cols if v == 'YieldSpread_chg']
    uncert = [v for v in var_cols if v in ['EPU_chg', 'GPR_chg', 'VIX']]
    equity = [v for v in var_cols if v in ['Russell3000', 'REIT', 'SP500', 'Nasdaq']]
    asset = [v for v in var_cols if v in ['Gold', 'CrudeOil', 'PoundUSD', 'EuroUSD']]

    pos = {}
    for i, v in enumerate(macro):   pos[v] = (-4, 4.2 - i * 1.05)
    for i, v in enumerate(spread):  pos[v] = (-1.8, -0.6)
    for i, v in enumerate(uncert):  pos[v] = (-0.3 + i * 1.6, 5.2)
    for i, v in enumerate(equity):  pos[v] = (4, 3.6 - i * 1.45)
    for i, v in enumerate(asset):   pos[v] = (6.6, 3.6 - i * 1.45)

    pre_arcs = pre_dag[pre_dag['n_methods'] >= 3] if not pre_dag.empty else pd.DataFrame()
    post_arcs = post_dag[post_dag['n_methods'] >= 3] if not post_dag.empty else pd.DataFrame()
    pre_set = set(zip(pre_arcs['cause'], pre_arcs['effect'])) if not pre_arcs.empty else set()
    post_set = set(zip(post_arcs['cause'], post_arcs['effect'])) if not post_arcs.empty else set()
    invariant = pre_set & post_set

    NODE_R = 0.52      # larger nodes (was 0.4)
    LABEL_FS = 11      # larger node label font (was 7)

    for ax, regime_name, dag, regime_color, title in zip(axes,
        ['pre', 'post'],
        [pre_arcs, post_arcs],
        [PAL['pre'], PAL['post']],
        [
            f'(a) Pre-COVID DAG (Dec 2004 - Feb 2020, n=167)\n{len(pre_set)} consensus arcs',
            f'(b) Post-COVID DAG (Mar 2020 - Sep 2025, n=40)\n{len(post_set)} consensus arcs'
        ]):
        # arcs first (under nodes)
        for _, row in dag.iterrows():
            if row['cause'] not in pos or row['effect'] not in pos: continue
            x1, y1 = pos[row['cause']]; x2, y2 = pos[row['effect']]
            edge = (row['cause'], row['effect'])
            if edge in invariant:
                color = PAL['invariant']; lw = 1.6; alpha = 0.75
            else:
                color = regime_color
                lw = 1.5 if row['n_methods'] == 4 else 0.8
                alpha = 0.7 if row['n_methods'] == 4 else 0.40
            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                                      mutation_scale=11, color=color, linewidth=lw,
                                      alpha=alpha, zorder=2, connectionstyle="arc3,rad=0.1")
            ax.add_patch(arrow)

        # nodes on top
        for var in var_cols:
            if var not in pos: continue
            x, y = pos[var]
            c = Circle((x, y), NODE_R, color=vcolor(var), alpha=0.9, zorder=3,
                        edgecolor='black', linewidth=1.2)
            ax.add_patch(c)
            ax.text(x, y, short(var), ha='center', va='center', fontsize=LABEL_FS,
                    fontweight='bold', color='white', zorder=4)

        ax.set_xlim(-5.5, 8); ax.set_ylim(-3, 6.2)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=13, pad=8)

    fig.suptitle(f'Causal architecture before vs after March 2020. '
                  f'Jaccard similarity = {jaccard_val:.3f} at >= 3/4 consensus.',
                  fontsize=15, y=1.01)

    fig.text(0.5, 0.04,
              f'Pre-COVID specific (blue)   |   Post-COVID specific (red)   |   '
              f'Invariant arcs ({len(invariant)}, dark gray)',
              ha='center', fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.99])
    return fig

# ============================================================================
# FIGURE 8: Invariant arc slopes (paper Fig 3)
# ============================================================================
def fig8_invariant_slopes(invariant_effects_df):
    if invariant_effects_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No invariant effects data', ha='center')
        return fig

    df = invariant_effects_df.sort_values('change_in_std_effect', ascending=True).copy()
    n = len(df)
    fig, ax = plt.subplots(figsize=(11, max(7, 0.45 * n)))

    for _, row in df.iterrows():
        pre_v = row['pre_std_effect']; post_v = row['post_std_effect']
        label = f"{row['cause']} → {row['effect']}"
        if abs(post_v - pre_v) < 0.05:    color = PAL['neutral']
        elif post_v > pre_v:                color = PAL['pos']
        else:                               color = PAL['neg']
        ax.plot([0, 1], [pre_v, post_v], color=color, linewidth=2.2,
                marker='o', markersize=8, alpha=0.85)
        ax.text(1.02, post_v, label, va='center', fontsize=8.5)
        ax.text(-0.04, pre_v, f'{pre_v:+.2f}', va='center', ha='right',
                fontsize=7, color=color)
        ax.text(0.96, post_v, f'{post_v:+.2f}', va='center', ha='right',
                fontsize=7, color=color)

    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-COVID\n(Dec 2004 – Feb 2020)',
                         'Post-COVID\n(Mar 2020 – Sep 2025)'], fontsize=10)
    ax.set_ylabel('Standardized Effect Size (σ)', fontsize=11)
    ax.set_title('How effect sizes of invariant arcs shifted across the COVID break.\n'
                  'Green = strengthened, Red = weakened, Gray = stable.',
                  fontsize=15
                  , pad=15)
    ax.set_xlim(-0.15, 1.55)
    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 9: Bai-Perron timeline (paper Fig 4)
# ============================================================================
def fig9_bai_perron_timeline(bp_breaks_df):
    if bp_breaks_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No Bai-Perron data', ha='center')
        return fig

    all_dates = []
    for _, row in bp_breaks_df.iterrows():
        dates_str = row.get('break_dates', '[]')
        if isinstance(dates_str, str) and dates_str != '[]':
            cleaned = dates_str.strip('[]').replace("'", '').replace('"', '')
            if cleaned.strip():
                all_dates.extend([d.strip() for d in cleaned.split(',')])
        elif isinstance(dates_str, list):
            all_dates.extend(dates_str)

    if not all_dates:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No break dates', ha='center')
        return fig

    dates = pd.to_datetime(all_dates, errors='coerce').dropna()
    counts = dates.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axvspan(pd.Timestamp('2004-12-01'), pd.Timestamp('2008-09-01'),
                alpha=0.10, color='#888', label='Pre-crisis')
    ax.axvspan(pd.Timestamp('2008-09-01'), pd.Timestamp('2015-01-01'),
                alpha=0.12, color='#BA7517', label='GFC era')
    ax.axvspan(pd.Timestamp('2015-01-01'), pd.Timestamp('2020-03-01'),
                alpha=0.08, color='#7AAEDC', label='Mid-cycle')
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2022-03-01'),
                alpha=0.15, color='#E24B4A', label='COVID/QE')
    ax.axvspan(pd.Timestamp('2022-03-01'), pd.Timestamp('2025-10-01'),
                alpha=0.10, color='#BA7517', label='Tightening')

    ax.bar(counts.index, counts.values, width=20, color='#2A2A2A',
            alpha=0.85, edgecolor='white', linewidth=0.5)

    top5 = counts.nlargest(5)
    for date, ct in top5.items():
        ax.annotate(f'{date.strftime("%b %Y")}: {ct}',
                    xy=(date, ct), xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')

    ax.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--',
                linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Detected Break Date')
    ax.set_ylabel('Number of consensus arcs detecting break here')
    ax.set_title('Bai-Perron sequential structural break dates across consensus arcs.\n'
                  'Concentration around March-May 2020 confirms the principal regime break.',
                  fontsize=12, pad=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(pd.Timestamp('2004-01-01'), pd.Timestamp('2025-10-01'))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs('figures', exist_ok=True)
    print("=" * 76)
    print("COMPLETE VISUALIZATION SUITE — n=207 DATASET")
    print("=" * 76)

    results_dir = './results'

    # Load main data
    print("\n[1/9] Loading merged dataset and main results...")
    df = pd.read_csv(f'{results_dir}/merged_dataset.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    var_cols = [c for c in get_var_cols() if c in df.columns]
    print(f"   n = {len(df)}, variables = {len(var_cols)}")

    cons_df = pd.read_csv(f'{results_dir}/consensus_arcs.csv')
    granger_df = pd.read_csv(f'{results_dir}/granger_results.csv')
    effects = pd.read_csv(f'{results_dir}/effect_sizes.csv')
    stat_df = pd.read_csv(f'{results_dir}/stationarity_tests.csv')

    # Re-run var_lingam to get causal order (it's not cached)
    print("   Re-running VAR-LiNGAM for causal order...")
    X = df[var_cols].values
    causal_order, _, _, _ = var_lingam(X, max_lag=1)

    # Figures 1-6
    print("\n[2/9] Figure 1: Time series with regime shading")
    save(fig1_timeseries(df, var_cols), 'fig1_time_series')

    print("\n[3/9] Figure 2: Full-sample consensus DAG")
    save(fig2_consensus_dag(cons_df, var_cols), 'fig2_consensus_dag')

    print("\n[4/9] Figure 3: Granger F-statistic heatmap")
    save(fig3_granger_heatmap(granger_df, var_cols), 'fig3_granger_heatmap')

    print("\n[5/9] Figure 4: Top effect sizes")
    save(fig4_effect_sizes(effects), 'fig4_effect_sizes')

    print("\n[6/9] Figure 5: VAR-LiNGAM causal ordering")
    save(fig5_lingam_ordering(causal_order, var_cols), 'fig5_lingam_ordering')

    print("\n[7/9] Figure 6: ADF stationarity tests")
    save(fig6_stationarity(stat_df), 'fig6_stationarity')

    # Figures 7-9 require regime + Bai-Perron results
    regime_files = [
        f'{results_dir}/regime_dag_pre_v3.csv',
        f'{results_dir}/regime_dag_post_v3.csv',
        f'{results_dir}/regime_invariant_effects_v3.csv',
        f'{results_dir}/regime_similarity_v3.csv',
    ]
    bp_file = f'{results_dir}/bai_perron_breaks.csv'

    if all(os.path.exists(f) for f in regime_files):
        print("\n[8/9] Figure 7: Regime DAGs side-by-side")
        pre_dag = pd.read_csv(f'{results_dir}/regime_dag_pre_v3.csv')
        post_dag = pd.read_csv(f'{results_dir}/regime_dag_post_v3.csv')
        sim_df = pd.read_csv(f'{results_dir}/regime_similarity_v3.csv')
        jaccard_val = sim_df['jaccard_3plus'].iloc[0]
        save(fig7_regime_dags(pre_dag, post_dag, var_cols, jaccard_val), 'fig7_regime_dags')

        print("\n   Figure 8: Invariant arc slope chart")
        inv_eff = pd.read_csv(f'{results_dir}/regime_invariant_effects_v3.csv')
        save(fig8_invariant_slopes(inv_eff), 'fig8_invariant_slopes')
    else:
        print("\n[!] Skipping Figures 7-8 — regime comparison not yet run")
        print("    Run: python regime_bn_comparison_v3.py")

    if os.path.exists(bp_file):
        print("\n[9/9] Figure 9: Bai-Perron break date timeline")
        bp = pd.read_csv(bp_file)
        save(fig9_bai_perron_timeline(bp), 'fig9_bai_perron_timeline')
    else:
        print("\n[!] Skipping Figure 9 — Bai-Perron not yet run")
        print("    Run: python bai_perron_breaks.py")

    print(f"\n{'=' * 76}")
    print(f"All figures saved to ./figures/ (PNG + PDF)")
    print(f"{'=' * 76}")


if __name__ == "__main__":
    main()