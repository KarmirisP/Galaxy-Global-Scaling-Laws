#!/usr/bin/env python3
"""
full_comparison_fixed.py
Clean 5-fold CV for six models.
Fixes:
- Replaces deprecated delim_whitespace with sep=r'\s+'
- Guards against sqrt(negative) in Unified & MOND models
- Adds numeric type conversion for data columns
Author: P. Karmiris, 2025-08
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution
from sklearn.model_selection import KFold

# -------------------- DATA --------------------
CSV_FILE = "SPARC_Lelli2016_masses_cleaned.csv"
DATA_DIR = "sparc_data"
Path("figures").mkdir(exist_ok=True)
Path("tables").mkdir(exist_ok=True)

MIN_INC, MAX_QUAL = 30.0, 2
G_ASTRO = 4.3009e-6
EPS = 1e-12


def load_sparc():
    meta = pd.read_csv(CSV_FILE, quotechar='"', low_memory=False)
    for c in ['Rdisk', 'MHI', 'L3.6', 'Type', 'i', 'Qual']:
        meta[c] = pd.to_numeric(meta[c], errors='coerce')
    meta['Name_std'] = meta['Name'].str.strip().str.replace(
        r'\s+', '', regex=True).str.upper()
    meta.set_index('Name_std', inplace=True)

    galaxies = []
    skipped = 0
    for fp in glob.glob(os.path.join(DATA_DIR, "*.dat")):
        name_std = "".join(Path(fp).stem.split('_')[0].split()).upper()
        if name_std not in meta.index:
            continue
        m = meta.loc[name_std]
        if m['Qual'] > MAX_QUAL or m['i'] < MIN_INC or pd.isna([m['Rdisk'], m['i']]).any():
            skipped += 1
            continue

        df = pd.read_csv(fp, sep=r'\s+', comment='#',
                         names=['R', 'V_obs', 'e_V_obs', 'V_gas', 'V_disk', 'V_bulge'])
        
        # Convert all columns to numeric - FIX ADDED HERE
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        if len(df) <= 5:
            skipped += 1
            continue

        sin_i = np.sin(np.radians(m['i']))
        # Ensure division works with floats
        df['V_obs'] = df['V_obs'].astype(float) / sin_i
        df['e_V_obs'] = df['e_V_obs'].astype(float) / sin_i
        
        # Convert components to float before operations
        V_gas = df['V_gas'].astype(float)
        V_disk = df['V_disk'].astype(float)
        V_bulge = df['V_bulge'].astype(float)
        df['v_bary'] = np.sqrt(np.maximum(0, V_gas**2 + 0.5*V_disk**2 + 0.7*V_bulge**2))

        stellar_mass = m['L3.6'] * 0.5
        total_bary = m['MHI'] + stellar_mass
        log_Eg = np.log10(abs(-G_ASTRO * total_bary**2 / (m['Rdisk']+EPS)))

        galaxies.append({'name': m['Name'], 'df': df,
                        'T': m['Type'], 'log_Eg': log_Eg})
    print(f"Loaded {len(galaxies)} galaxies, skipped {skipped}")
    return galaxies


# -------------------- MODELS --------------------
MODELS = {
    "UC_Base": {
        "func": lambda df, p, feat: df['v_bary']*np.sqrt(np.maximum(0.01, p[0]+p[1]*feat['T'])),
        "bounds": [(0.1, 2), (-0.2, 0.2)]
    },
    "Partanen_Energy": {
        "func": lambda df, p, feat: df['v_bary']*np.sqrt(np.maximum(0.01,
                                                                    1+(p[0]+p[1]*feat['log_Eg'])/(1+p[2]*feat['log_Eg']+EPS))),
        "bounds": [(-5, 5), (-5, 5), (-5, 5)]
    },
    "Unified_Interaction": {   # non-linear coupling, 4 pars
        "func": lambda df, p, feat: df['v_bary']*np.sqrt(np.maximum(0,
                                                                    1 + (p[0]+p[1]*feat['T'])*(p[2]+p[3]*feat['log_Eg']))),
        "bounds": [(0.1, 2), (-0.2, 0.2), (-0.5, 0.5), (-0.2, 0.2)]
    },
    "ΛCDM_NFW": {
        "func": lambda df, p, feat: np.sqrt(df['v_bary']**2 + p[0]**2 *
                                            (np.log(1+df['R']/p[1])-df['R']/p[1]/(1+df['R']/p[1]))/(df['R']/p[1])),
        "bounds": [(10, 400), (0.1, 50)]
    },
    "RAR": {
        "func": lambda df, p, feat: np.sqrt(df['R']*df['v_bary']**2/(df['R']+EPS) /
                                            (1-np.exp(-np.sqrt(df['v_bary']**2/(df['R']+EPS)/p[0])))),
        "bounds": [(1000, 5000)]
    },
    "MOND_Simple": {
        "func": lambda df, p, feat: np.sqrt(df['v_bary']**2 + np.sqrt(df['v_bary']**2 * p[0] * df['R'])),
        "bounds": [(1000, 5000)]
    }
}


def chi2(pars, df, model, features):
    pred = model(df, pars, features)
    err = np.maximum(df['e_V_obs'], 3.0)
    return np.sum(((df['V_obs'] - pred)/err)**2)

# -------------------- 5-FOLD CV --------------------


def run_cv(galaxies):
    gal = np.array(galaxies, dtype=object)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(kf.split(gal), 1):
        for name, m in MODELS.items():
            def obj(p): return sum(
                chi2(p, g['df'], m['func'], g) for g in gal[tr])
            res = differential_evolution(
                obj, m['bounds'], maxiter=100, tol=0.01, seed=42)
            for g in gal[te]:
                ch = chi2(res.x, g['df'], m['func'], g)
                dof = max(1, len(g['df']) - len(res.x))
                rows.append(
                    {'galaxy': g['name'], 'model': name, 'chi2_red': ch/dof, 'fold': fold})
    return pd.DataFrame(rows)

# -------------------- OUTPUT --------------------


def summarize(df):
    summary = (df.groupby('model')['chi2_red']
                 .agg(success=lambda x: (x < 3).mean()*100, median=np.median)
                 .sort_values('median'))
    print("\n🏆 FINAL RANKING:")
    print(summary)
    summary.to_latex('tables/full_comparison.tex', float_format='%.2f')


if __name__ == "__main__":
    galaxies = load_sparc()
    results = run_cv(galaxies)
    summarize(results)