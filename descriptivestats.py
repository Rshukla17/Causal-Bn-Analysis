"""
================================================================================
DESCRIPTIVE STATISTICS TABLE GENERATOR
================================================================================
Reads results/merged_dataset.csv, computes mean / std / min / max for the 19
modeling variables, and writes a ready-to-paste LaTeX table to
results/descriptive_stats_table.tex

Usage:
  python make_descriptive_table.py
================================================================================
"""

import pandas as pd

# ----------------------------------------------------------------------------
# The 19 modeling variables, in paper order, with human-readable descriptions.
# (9 asset-price series entered as log returns; 10 macro/uncertainty/spread
#  series entered in log-difference form, suffix _chg.)
# ----------------------------------------------------------------------------
VAR_INFO = [
    # column name          description                                     unit
    ("Gold",               "Gold ETF return (SPDR GLD)",                   "log return"),
    ("CrudeOil",           "WTI Crude Oil return",                         "log return"),
    ("Russell3000",        "Russell 3000 return (iShares IWV)",            "log return"),
    ("REIT",               "REIT return (Vanguard VNQ)",                   "log return"),
    ("VIX",                "CBOE Volatility Index change",                 "log return"),
    ("Nasdaq",             "NASDAQ Composite return",                      "log return"),
    ("SP500",              "S\\&P 500 return",                              "log return"),
    ("PoundUSD",           "Pound-USD exchange rate return",               "log return"),
    ("EuroUSD",            "Euro-USD exchange rate return",                "log return"),
    ("CPI_chg",            "Consumer Price Index change",                  "log diff."),
    ("FEDFUNDS_chg",       "Federal Funds rate change",                    "log diff."),
    ("UNRATE_chg",         "Unemployment rate change",                     "log diff."),
    ("M2_chg",             "M2 money supply change",                       "log diff."),
    ("PCE_chg",            "Personal consumption expenditures change",     "log diff."),
    ("INDPRO_chg",         "Industrial production change",                 "log diff."),
    ("ConsumerConf_chg",   "U. Michigan consumer confidence change",       "log diff."),
    ("YieldSpread_chg",    "Term spread (10y--2y Treasury) change",        "log diff."),
    ("EPU_chg",            "Economic Policy Uncertainty index change",     "log diff."),
    ("GPR_chg",            "Geopolitical Risk index change",               "log diff."),
]

# LaTeX-safe display names for the first column
DISPLAY_NAME = {
    "Gold": "Gold", "CrudeOil": "CrudeOil", "Russell3000": "Russell3000",
    "REIT": "REIT", "VIX": "VIX", "Nasdaq": "Nasdaq", "SP500": "SP500",
    "PoundUSD": "PoundUSD", "EuroUSD": "EuroUSD",
    "CPI_chg": "CPI\\_chg", "FEDFUNDS_chg": "FEDFUNDS\\_chg",
    "UNRATE_chg": "UNRATE\\_chg", "M2_chg": "M2\\_chg", "PCE_chg": "PCE\\_chg",
    "INDPRO_chg": "INDPRO\\_chg", "ConsumerConf_chg": "ConsumerConf\\_chg",
    "YieldSpread_chg": "YieldSpread\\_chg", "EPU_chg": "EPU\\_chg",
    "GPR_chg": "GPR\\_chg",
}


def main():
    df = pd.read_csv("results/merged_dataset.csv")
    n = len(df)

    rows = []
    for col, desc, unit in VAR_INFO:
        if col not in df.columns:
            print(f"[!] WARNING: column '{col}' not in dataset, skipping")
            continue
        s = df[col].dropna()
        rows.append((
            DISPLAY_NAME.get(col, col),
            desc,
            s.mean(),
            s.std(),
            s.min(),
            s.max(),
        ))

    # ---- Write LaTeX ----
    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{Descriptive Statistics for the Nineteen Modeling Variables "
                 f"(Monthly, Dec 2004 -- Sep 2025, $n={n}$)}}")
    lines.append("\\label{tab:descriptive}")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Variable} & \\textbf{Description} & \\textbf{Mean} & "
                 "\\textbf{Std.\\ Dev.} & \\textbf{Min} & \\textbf{Max} \\\\")
    lines.append("\\midrule")
    for name, desc, mean, std, mn, mx in rows:
        lines.append(f"{name} & {desc} & {mean:.4f} & {std:.4f} & {mn:.4f} & {mx:.4f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\\\[6pt]")
    lines.append("\\begin{minipage}{0.95\\textwidth}")
    lines.append("\\footnotesize \\textit{Notes:} Asset-price series (Gold through "
                 "Euro-USD) enter as monthly log returns; macroeconomic, term-spread, "
                 "and uncertainty series enter in log-difference form (suffix "
                 "\\texttt{\\_chg}). All series are stationary at the 1\\% level by "
                 "Augmented Dickey-Fuller test.")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")

    out = "\n".join(lines)
    with open("results/descriptive_stats_table.tex", "w") as f:
        f.write(out)

    print("Wrote results/descriptive_stats_table.tex\n")
    print(out)


if __name__ == "__main__":
    main()