# Causal Inference in Multi-Asset Pricing Models — Code Repository

**Author:** Rajan Shukla
**Supervisor:** Dr. Raju Mainali
**Institution:** Missouri State University

## Files

| File | Purpose |
|------|---------|
| `causal_bn_analysis.py` | Master Python pipeline — all four algorithms, consensus, effect sizes, regimes, structural break |
| `README.md` | This file |
| `Section4_Results.docx` | Drafted Section 4 of the paper |

## Quick start

```bash
pip install numpy pandas scipy scikit-learn matplotlib
python causal_bn_analysis.py
```

The script expects monthly CSV files in `/mnt/user-data/uploads/` (adjust the `BASE` constant at the top to point to your local data directory).

## Required data files

**Asset returns** (one row per month with `Month`, `Log_Return` columns):
- `Gold_monthly.csv`, `CrudeOil_monthly.csv`, `Bitcoin_monthly.csv`,
  `Russell3000_monthly.csv`, `REIT_monthly.csv`,
  `PoundUSD_monthly.csv`, `EuroUSD_monthly.csv`, `VIX_monthly.csv`

**Macro variables** (FRED-style with `observation_date` + value column):
- `CPI.csv` (CPIAUCSL), `FEDFUNDS.csv`, `UNRATE.csv`,
  `WM2NS.csv` (M2), `PCE.csv`, `GDPC1.csv` (GDP, quarterly)

**EPU and GPR** are currently hardcoded as approximate published values.
**Before publication, replace these with real downloads:**
- EPU: https://fred.stlouisfed.org/series/USEPUINDXM
- GPR: https://www.matteoiacoviello.com/gpr.htm

## Algorithms implemented

1. **Granger causality** (full F-test with AIC lag selection up to lag 3)
2. **PCMCI+** (simplified — for production use the `tigramite` package)
3. **DYNOTEARS** (basic gradient-descent with L1 — for production use `causalnex`)
4. **VAR-LiNGAM** (DirectLiNGAM with kurtosis non-Gaussianity score — for production use `lingam`)

## Output CSVs

- `granger_results.csv` — significant Granger pairs
- `pcmci_results.csv` — significant PCMCI+ links
- `consensus_arcs.csv` — cross-algorithm consensus (sorted by `n_methods`)
- `effect_sizes.csv` — OLS effect-size quantification
- `covid_structural_break.csv` — pre/post-2020 Granger comparison

## Production checklist before submission

Critical fixes needed before journal submission:

1. Extend sample period through end of 2024 (target n ≈ 120)
2. Download real EPU and GPR CSVs (replace hardcoded values)
3. Install and use `causalnex`, `tigramite`, `lingam` for proper algorithm implementations
4. Add ADF and KPSS stationarity tests for all variables
5. Add bootstrap confidence intervals on all reported F-statistics (B = 500)
6. Add out-of-sample forecast evaluation (rolling-window CV)
7. Add LPCMCI run as latent-confounder robustness check

## Citing

If using this code, please cite:
- Pamfil et al. (2020) — DYNOTEARS, AISTATS
- Runge et al. (2019) — PCMCI, Nature Communications
- Hyvärinen et al. (2010) — VAR-LiNGAM, JMLR
- Granger (1969) — Econometrica
- Baker, Bloom & Davis (2016) — EPU, QJE
- Caldara & Iacoviello (2022) — GPR, AER
