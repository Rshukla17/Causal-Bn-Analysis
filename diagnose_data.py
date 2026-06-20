"""
Diagnostic script — run this FIRST to see what's in your data files
and why the merge is producing only 12 rows.

Usage:
    python diagnose_data.py
"""

import os
import pandas as pd

DATA_DIR = "./data"

print("=" * 76)
print("DATA DIRECTORY DIAGNOSTIC")
print("=" * 76)

if not os.path.exists(DATA_DIR):
    print(f"\n[FAIL] {DATA_DIR} does not exist!")
    exit()

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
print(f"\nFound {len(files)} CSV files in {DATA_DIR}:\n")

for f in files:
    path = os.path.join(DATA_DIR, f)
    try:
        df = pd.read_csv(path)
        n_rows = len(df)
        cols = list(df.columns)
        # Detect date column
        date_col = None
        for c in cols:
            if 'date' in c.lower() or 'month' in c.lower():
                date_col = c
                break
        first_date = df[date_col].iloc[0] if date_col else "N/A"
        last_date = df[date_col].iloc[-1] if date_col else "N/A"
        print(f"  {f:<30} rows={n_rows:>4}  cols={cols}")
        print(f"  {'':<30} date_col={date_col!r}  first={first_date!r}  last={last_date!r}")
        print()
    except Exception as e:
        print(f"  {f:<30} ERROR: {e}\n")

# Check GPR specifically
print("\n" + "=" * 76)
print("GPR CHECK")
print("=" * 76)
gpr_path = os.path.join(DATA_DIR, 'GPR.csv')
gpr_root = './gpr_export.xlsx'
print(f"  ./data/GPR.csv exists?     {os.path.exists(gpr_path)}")
print(f"  ./gpr_export.xlsx exists?  {os.path.exists(gpr_root)}")

# Show date format diversity
print("\n" + "=" * 76)
print("DATE FORMAT COMPARISON")
print("=" * 76)
print("This is the key diagnostic — if formats differ, the merge fails:\n")

samples = {}
for f in files[:6]:
    path = os.path.join(DATA_DIR, f)
    try:
        df = pd.read_csv(path)
        date_col = next((c for c in df.columns
                          if 'date' in c.lower() or 'month' in c.lower()), None)
        if date_col:
            samples[f] = (date_col, df[date_col].iloc[0], df[date_col].iloc[-1])
    except Exception:
        pass

for f, (col, first, last) in samples.items():
    print(f"  {f:<30}  {col!r:<20}  '{first}'  ...  '{last}'")
