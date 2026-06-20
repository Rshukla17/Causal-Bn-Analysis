"""
================================================================================
DATA FETCHER v4 — FRED + yfinance hybrid (final working version)
================================================================================
Strategy:
  - FRED for everything FRED has (macro, FX, VIX, Crude Oil, Nasdaq, DXY)
  - yfinance for assets FRED no longer provides (Gold, Bitcoin, REIT, equity indices)

Why yfinance: FRED has discontinued daily gold/equity/REIT series. Stooq has
added bot blocking. Yahoo Finance via yfinance library properly handles their
session/cookie auth. Once data is saved to local CSVs, the source no longer
matters for reproducibility.

Setup:
    pip install pandas requests numpy openpyxl yfinance
    python data_fetcher_v4.py
================================================================================
"""

import os
import pandas as pd
import numpy as np
import requests
from io import StringIO

# ============================================================================
# CONFIGURATION
# ============================================================================

FRED_API_KEY = "0f7e83718c0bedcb414d311fbd18fb21"   # ← Your FRED key
START_DATE        = "2000-01-01"
BITCOIN_START     = "2014-10-01"
END_DATE          = "2025-10-31"
OUTPUT_DIR        = "./data"
GPR_FILE          = "./gpr_export.xlsx"
 
 
# ============================================================================
# 1. FRED FETCHER
# ============================================================================
 
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
 
 
def fetch_fred(series_id, start=START_DATE, end=END_DATE,
                force_monthly=False, agg_method="avg"):
    """Pull a FRED series with optional monthly aggregation."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    if force_monthly:
        params["frequency"] = "m"
        params["aggregation_method"] = agg_method
 
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        raise ValueError(f"No data for {series_id}")
    df = pd.DataFrame(obs)[["date", "value"]]
    df["Month"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    df = df[df["value"] > 0]
    return df[["Month", "value"]].reset_index(drop=True)
 
 
def save_macro(series_id, output_name, force_monthly=False, start=START_DATE):
    df = fetch_fred(series_id, start=start, force_monthly=force_monthly)
    df.columns = ["observation_date", series_id]
    out = os.path.join(OUTPUT_DIR, output_name)
    df.to_csv(out, index=False)
    print(f"   [OK] {output_name:<25} ({len(df)} obs)  [{series_id}]")
 
 
def save_asset_from_fred(series_id, output_name, start=START_DATE):
    """FRED daily price -> end-of-month -> log returns."""
    df = fetch_fred(series_id, start=start, force_monthly=True, agg_method="eop")
    df["Log_Return"] = np.log(df["value"] / df["value"].shift(1))
    df = df.dropna()
    df["Month"] = df["Month"].dt.strftime("%Y-%m")
    out_df = df[["Month", "Log_Return"]]
    out = os.path.join(OUTPUT_DIR, output_name)
    out_df.to_csv(out, index=False)
    print(f"   [OK] {output_name:<25} ({len(out_df)} obs)  [FRED:{series_id}]")
 
 
# ============================================================================
# 2. YFINANCE FETCHER (for assets FRED doesn't have)
# ============================================================================
 
def save_yfinance_asset(symbol, output_name, start=START_DATE):
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")
 
    print(f"   ... fetching {symbol} from Yahoo Finance...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=END_DATE, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {symbol}")
 
    df = df[["Close"]].copy()
    df = df[df["Close"] > 0]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    monthly = df["Close"].resample("ME").last()
    log_ret = np.log(monthly / monthly.shift(1)).dropna()
    out_df = pd.DataFrame({
        "Month": log_ret.index.strftime("%Y-%m"),
        "Log_Return": log_ret.values
    })
    out = os.path.join(OUTPUT_DIR, output_name)
    out_df.to_csv(out, index=False)
    print(f"   [OK] {output_name:<25} ({len(out_df)} obs)  [Yahoo:{symbol}]")
 
 
# ============================================================================
# 3. GPR PARSER
# ============================================================================
 
def save_gpr():
    if not os.path.exists(GPR_FILE):
        print(f"   [SKIP] {GPR_FILE} not found")
        return False
    try:
        df = pd.read_excel(GPR_FILE)
    except ImportError:
        print(f"   [SKIP] xlrd missing. Run: pip install xlrd")
        return False
 
    date_col = next((c for c in df.columns if "month" in str(c).lower()
                     or "date" in str(c).lower()), df.columns[0])
    gpr_col = next((c for c in df.columns if str(c).strip().upper() == "GPR"), None)
    if gpr_col is None:
        gpr_col = next((c for c in df.columns
                        if "gpr" in str(c).lower() and "_" not in str(c).lower()), None)
    if gpr_col is None:
        print(f"   [SKIP] GPR column not found")
        return False
 
    df = df[[date_col, gpr_col]].copy()
    df.columns = ["observation_date", "GPR"]
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df[(df["observation_date"] >= START_DATE)
            & (df["observation_date"] <= END_DATE)]
    df = df.dropna().reset_index(drop=True)
    out = os.path.join(OUTPUT_DIR, "GPR.csv")
    df.to_csv(out, index=False)
    print(f"   [OK] GPR.csv                  ({len(df)} obs)  [Iacoviello]")
    return True
 
 
# ============================================================================
# 4. MAIN
# ============================================================================
 
def main():
    print("=" * 76)
    print("DATA FETCHER v5 — Extended sample 2000-2025")
    print("Multi-crisis analysis: dot-com, GFC, Eurozone, COVID, tightening")
    print("=" * 76)
 
    if FRED_API_KEY == "PUT_YOUR_FRED_KEY_HERE":
        print("\n[ERROR] Set FRED_API_KEY first")
        return
 
    try:
        import yfinance
    except ImportError:
        print("\n[ERROR] yfinance not installed. Run: pip install yfinance")
        return
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput:        {os.path.abspath(OUTPUT_DIR)}")
    print(f"Main range:    {START_DATE} to {END_DATE}")
    print(f"Bitcoin range: {BITCOIN_START} to {END_DATE} (separate file)\n")
 
    # ── ASSETS FROM FRED (back to 2000) ─────────────────────────────────
    print("[1/4] Pulling assets that FRED has (2000-2025)...")
    fred_assets = [
        ("DCOILWTICO", "CrudeOil_monthly.csv"),
        ("DEXUSUK",    "PoundUSD_monthly.csv"),
        ("DEXUSEU",    "EuroUSD_monthly.csv"),       # Note: only from 1999
        ("VIXCLS",     "VIX_monthly.csv"),
        ("NASDAQCOM",  "Nasdaq_monthly.csv"),
        ("DTWEXBGS",   "DXY_monthly.csv"),           # Goes back to 2006
    ]
    for series_id, fname in fred_assets:
        try:
            save_asset_from_fred(series_id, fname)
        except Exception as e:
            print(f"   [FAIL] {fname:<25} {str(e)[:70]}")
 
    # ── ASSETS FROM YFINANCE (back to 2000, except Bitcoin) ──────────────
    print("\n[2/4] Pulling assets from yfinance (2000-2025, Bitcoin separate)...")
    yf_assets_full = [
        ("GLD",    "Gold_monthly.csv"),
        ("IWV",    "Russell3000_monthly.csv"),
        ("VNQ",    "REIT_monthly.csv"),
        ("^GSPC",  "SP500_monthly.csv"),
    ]
    for symbol, fname in yf_assets_full:
        try:
            save_yfinance_asset(symbol, fname, start=START_DATE)
        except Exception as e:
            print(f"   [FAIL] {fname:<25} {str(e)[:70]}")
 
    # Bitcoin separately (2014 onwards)
    print("\n   Bitcoin (2014 onwards, separate file):")
    try:
        save_yfinance_asset("BTC-USD", "Bitcoin_monthly.csv", start=BITCOIN_START)
    except Exception as e:
        print(f"   [FAIL] Bitcoin_monthly.csv  {str(e)[:70]}")
 
    # ── MACRO ─────────────────────────────────────────────────────────────
    print("\n[3/4] Pulling macro from FRED (2000-2025)...")
    macro_native = [
        ("CPIAUCSL",   "CPI.csv"),
        ("FEDFUNDS",   "FEDFUNDS.csv"),
        ("UNRATE",     "UNRATE.csv"),
        ("M2SL",       "M2.csv"),
        ("PCE",        "PCE.csv"),
        ("INDPRO",     "INDPRO.csv"),
        ("UMCSENT",    "ConsumerConf.csv"),
        ("USEPUINDXM", "EPU.csv"),
    ]
    for series_id, fname in macro_native:
        try:
            save_macro(series_id, fname)
        except Exception as e:
            print(f"   [FAIL] {fname:<25} {str(e)[:70]}")
 
    # Daily series forced to monthly
    macro_daily = [
        ("T10Y2Y",       "YieldSpread.csv"),
        # CreditSpread (BAMLH0A0HYM2) only goes back to 1996, but it had issues
        # in v4 — we'll skip it for the extended sample to avoid bottlenecks
    ]
    for series_id, fname in macro_daily:
        try:
            save_macro(series_id, fname, force_monthly=True)
        except Exception as e:
            print(f"   [FAIL] {fname:<25} {str(e)[:70]}")
 
    # ── GPR ──────────────────────────────────────────────────────────────
    print("\n[4/4] Loading manually-downloaded GPR Excel...")
    save_gpr()
 
    print("\n" + "=" * 76)
    print(f"Done. Files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Main analysis sample: 2000-2025 (~309 obs, 18 variables)")
    print(f"Bitcoin sub-sample:   2014-2025 (~132 obs, 19 variables)")
    print("=" * 76)
 
 
if __name__ == "__main__":
    main()