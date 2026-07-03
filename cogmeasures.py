"""
cogmeasures.py — Cognitive measures coverage check + partial correlation analysis
==================================================================================
Measures (all controlling age + sex):
  - KBIT-2 Verbal (crystallised)
  - KBIT-2 Nonverbal
  - KBIT-2 IQ
  - NIH Toolbox Flanker (inhibitory control / attention)
"""

import numpy as np
import pandas as pd
import matplotlib
import h5py 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import pearsonr
from scipy.stats import t as t_dist

NIMH_DIR = Path('/Users/anwesha151200/Documents/specparam/nimh')
OUT_DIR  = Path('/Users/anwesha151200/Documents/specparam/results')
DATA_DIR = Path('/Users/anwesha151200/Documents/specparam/nimh/subject PSD') 
FIG_DIR  = OUT_DIR / 'figures'

# =============================================================================
# LOAD + MERGE
# =============================================================================

summary = pd.read_csv(OUT_DIR / 'nimh_subject_summary.csv').dropna(subset=['age', 'sex'])
n_meg   = len(summary) #number of usable MEG subjects after dropping unavailable age/sex 

def load_psd(filepath):
    with h5py.File(filepath, 'r') as f:
        freq = f['freq_out'][:].squeeze()
        pow_ = f['pow_out'][:].T             # transpose to (channels, freqs)
    return freq, pow_
 
pow_data = {}
freq     = None
for sub in summary['subject']:
    f, p = load_psd(DATA_DIR / f'{sub}_psd.mat')
    pow_data[sub] = p
    if freq is None:
        freq = f

kbit = pd.read_csv(NIMH_DIR / 'kbit2_vas.csv',   sep='\t').replace(-999, np.nan)
nih  = pd.read_csv(NIMH_DIR / 'nih_toolbox.csv',  sep='\t').replace(-999, np.nan)

# keep only MEG subjects and scores from baseline visit
kbit = (kbit[kbit['participant_id'].isin(summary['subject'])]
            .sort_values('visit')
            .drop_duplicates(subset='participant_id', keep='first'))
nih  = (nih [nih ['participant_id'].isin(summary['subject'])]
            .sort_values('visit')
            .drop_duplicates(subset='participant_id', keep='first'))

KBIT_COLS = ['participant_id', 'stand_verb', 'stand_nonverb', 'stand_IQ']
NIH_COLS  = [
    'participant_id',
    'FLANKER_INHIB_CTL_ATTN_AGE_CRE']

df = (summary
      .merge(kbit[KBIT_COLS], left_on='subject', right_on='participant_id', how='left')
      .drop(columns='participant_id')
      .merge(nih[NIH_COLS],  left_on='subject', right_on='participant_id', how='left')
      .drop(columns='participant_id'))

# =============================================================================
# OVERALL SUMMARY
# =============================================================================
kbit_subs = set(kbit['participant_id'])
nih_subs  = set(nih ['participant_id'])
all_subs  = set(summary['subject'])

print(f"\n{'='*60}")
print(f"Coverage summary (n={n_meg} MEG subjects)")
print(f"{'Subject':<22} {'KBIT-2':>8} {'NIH Flanker':>12}")
print(f"{'-'*55}")
for sub in sorted(all_subs):
    k = 'yes' if sub in kbit_subs else '*** MISSING ***'
    n = 'yes' if sub in nih_subs  else '*** MISSING ***'
    print(f"{sub:<22} {k:>8} {n:>12}")

#count of subjects who actually have a (non-NaN) value for each measure.
print(f"\nMEG total:                     {n_meg}")
print(f"MEG + KBIT-2 verbal:           {df['stand_verb'].notna().sum()}")
print(f"MEG + NIH Flanker:             {df['FLANKER_INHIB_CTL_ATTN_AGE_CRE'].notna().sum()}")

# =============================================================================
# PARTIAL CORRELATION 
# =============================================================================

def partial_r(x, y, covariates):
    """
    Partial r between x and y controlling for covariates.
    Residualises both on [intercept + covariates], then correlates residuals.
    """
    data = pd.DataFrame({'x': x, 'y': y})
    for i, c in enumerate(covariates):
        data[f'cov{i}'] = c
    data = data.dropna()
    n = len(data)
    if n < 5:
        return np.nan, np.nan, n

    X_cov = np.column_stack([np.ones(n)] + [data[f'cov{i}'].values
                                             for i in range(len(covariates))]) #covariates matrix

    def residualise(v):  #Least-squares fit
        b, _, _, _ = np.linalg.lstsq(X_cov, v, rcond=None)
        return v - X_cov @ b        #residual = actual - predicted

#Pearson correlation of the two residual sets (partial correlation)
    r, p = pearsonr(residualise(data['x'].values),
                    residualise(data['y'].values))
    return r, p, n

# =============================================================================
# MEASURES + ANALYSIS
# =============================================================================

MEASURES = [
    # KBIT-2
    ('KBIT-2 Verbal (crystallised)',  'stand_verb',                    'kbit'),
    ('KBIT-2 Nonverbal',              'stand_nonverb',                  'kbit'),
    ('KBIT-2 IQ',                     'stand_IQ',                       'kbit'),
    # NIH Toolbox (n≈102 each)
    ('Flanker (attention/inhibition)', 'FLANKER_INHIB_CTL_ATTN_AGE_CRE',  'nih')]

BLOCK_COLORS = {'kbit': '#E06C75', 'nih': '#61AFEF'} #group by dataset

ages    = df['age'].values
sex_bin = (df['sex'] == 'female').astype(float).values  # sex as 0/1 (female=1) for the model
exp     = df['mean_exponent'].values                    # aperiodic exponent

all_results = []

print("Partial correlations: exponent ~ cognitive measure (controlling age + sex)")
print(f"{'='*60}")
print(f"{'Measure':<34} {'r':>7} {'p':>9} {'n':>5} {'sig':>5}")
print(f"{'-'*60}")

#For each measure: partial r between that measure and the exponent, controlling age + sex.
for name, col, block in MEASURES:
    r, p, n = partial_r(df[col].values, exp, [ages, sex_bin])
    sig  = '**' if (not np.isnan(p) and p < 0.01) else \
           ('*'  if (not np.isnan(p) and p < 0.05) else 'ns')
    pstr = f'{p:.4f}' if not np.isnan(p) else '   NaN'
    rstr = f'{r:.3f}'  if not np.isnan(r) else '   NaN'
    print(f"{name:<34} {rstr:>7} {pstr:>9} {n:>5} {sig:>5}")
    all_results.append({'label': name, 'col': col, 'block': block,
                        'r': r, 'p': p, 'n': n})

results_df = pd.DataFrame(all_results)

# =============================================================================
# SEX AS CONFOUND — OLS: exponent ~ age + sex + cognitive_measure
# =============================================================================

def ols_full(y, X):
    """OLS returning beta, se, t, p for each predictor."""
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)      # coefficients minimising squared residuals
    resid = y - X @ b                                   # residuals = actual - predicted
    n, k  = X.shape                                     # n = subjects, k = predictors
    mse   = (resid @ resid) / (n - k)                   # unbiased error variance
    cov   = mse * np.linalg.inv(X.T @ X)                # covariance matrix of the coefficients
    se    = np.sqrt(np.diag(cov))                       # standard errors = sqrt of its diagonal
    tv    = b / se                                      # t-stat per coefficient
    pv    = 2 * t_dist.sf(np.abs(tv), df=n - k)         # two-tailed p-value per coefficient
    return b, se, tv, pv

print(f"\n{'='*75}")
print("Sex as confound: OLS  exponent ~ age + sex + cognitive_measure")
print(f"{'='*75}")
print(f"{'Measure':<34} {'β_sex':>8} {'p_sex':>9} {'β_cog':>8} {'p_cog':>9} {'sex_sig':>8}")
print(f"{'-'*75}")

for name, col, block in MEASURES:
    data = pd.DataFrame({
        'exp': exp, 'cog': df[col].values,
        'age': ages, 'sex': sex_bin,
    }).dropna()
    n = len(data)
    if n < 5:
        print(f"{name:<34}  {'(too few)':>8}")
        continue
    X = np.column_stack([np.ones(n), data['age'], data['sex'], data['cog']])
    b, se, tv, pv = ols_full(data['exp'].values, X)
    # b[0]=intercept, b[1]=age, b[2]=sex, b[3]=cog
    sex_sig = '**' if pv[2] < 0.01 else ('*' if pv[2] < 0.05 else 'ns')
    print(f"{name:<34} {b[2]:>8.4f} {pv[2]:>9.4f} {b[3]:>8.4f} {pv[3]:>9.4f} {sex_sig:>8}")

# =============================================================================
# FIGURE 1: Forest plot
# =============================================================================
plot_df = results_df[results_df['r'].notna()].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.7 + 1.5)))

for i, row in plot_df.iterrows():
    color  = BLOCK_COLORS[row['block']]     # colour by dataset
    marker = 'D' if (not np.isnan(row['p']) and row['p'] < 0.05) else 'o'
    ax.scatter(row['r'], i, color=color, s=90, marker=marker, zorder=3)
    ax.axhline(i, color='lightgrey', linewidth=0.5, zorder=1)

# set xlim symmetrically after all points are plotted
abs_max = plot_df['r'].abs().max()
ax.set_xlim(-abs_max * 1.4, abs_max * 1.4)

ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.set_yticks(range(len(plot_df)))
# include n in tick labels so no need in-plot text
yticklabels = [f"{row['label']}  (n={int(row['n'])})" for _, row in plot_df.iterrows()]
ax.set_yticklabels(yticklabels, fontsize=10)
ax.set_xlabel('Partial r (controlling age + sex)', fontsize=10)
ax.set_title('Exponent ~ cognitive measures  (◆ = p<0.05)', fontsize=11)
ax.grid(axis='x', alpha=0.3)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E06C75', markersize=10, label='KBIT-2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#61AFEF', markersize=10, label='NIH Toolbox'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig(FIG_DIR / 'cog_partial_r_forest.png', dpi=150)
plt.close()
print("\nSaved: cog_partial_r_forest.png")

# =============================================================================
# COGNITION × OSCILLATORY POWER  (raw vs aperiodic-corrected, all measures)
# =============================================================================

# ── load per-subject PSDs (so raw band power can be computed here too) ─────────
# Mirrors the loading in addplots.py. pow_data[subject] = (n_channels, n_freqs);
# freq = frequency vector shared across subjects.

BANDS_COG = ['alpha', 'beta', 'gamma']
BAND_HZ   = {'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}

# corrected peak power per band — read from summary CSV (already computed)
corr_pw = {b: df[f'{b}_PW'].values for b in BANDS_COG}

# raw band power per band — log-mean across channels per subject
# (requires pow_data + freq in scope, as in addplots.py)
def _raw_band(lo, hi):
    idx = (freq >= lo) & (freq <= hi)
    return np.array([np.log10(pow_data[s][:, idx]).mean()
                     for s in df['subject']])
raw_pw = {b: _raw_band(*BAND_HZ[b]) for b in BANDS_COG}

print(f"\n{'='*84}")
print("Cognition × band power  (partial r, controlling age + sex):  raw vs corrected")
print(f"{'='*84}")
print(f"{'Measure':<26} {'Band':<7} {'raw r':>8} {'raw p':>9} {'corr r':>8} {'corr p':>9} {'n':>5}")
print(f"{'-'*84}")

cog_power_results = []
for name, col, block in MEASURES:                 # loops all four cognitive measures
    cog = df[col].values
    for b in BANDS_COG:
        r_raw,  p_raw,  _ = partial_r(raw_pw[b],  cog, [ages, sex_bin])
        r_corr, p_corr, n = partial_r(corr_pw[b], cog, [ages, sex_bin])
        star = '*' if (not np.isnan(p_corr) and p_corr < 0.05) else ' '
        print(f"{name:<26} {b:<7} {r_raw:>8.3f} {p_raw:>9.4f} "
              f"{r_corr:>8.3f} {p_corr:>9.4f} {n:>5} {star}")
        cog_power_results.append({'measure': name, 'band': b,
                                  'raw_r': r_raw,  'raw_p': p_raw,
                                  'corr_r': r_corr, 'corr_p': p_corr, 'n': n})
    print(f"{'-'*84}")

cog_power_df = pd.DataFrame(cog_power_results)

# =============================================================================
# FIGURE: Cognition × band power — raw vs corrected (paired dots per measure×band)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, max(4, len(cog_power_df) * 0.5 + 1.5)))

labels = [f"{r['measure'].split(' (')[0]} · {r['band']}" for _, r in cog_power_df.iterrows()]
ypos   = np.arange(len(cog_power_df))

for i, (_, r) in enumerate(cog_power_df.iterrows()):
    # line linking raw -> corrected so the shift is visible
    ax.plot([r['raw_r'], r['corr_r']], [i, i], color='lightgrey', linewidth=1.5, zorder=1)
    ax.scatter(r['raw_r'],  i, color='#E06C75', s=70, zorder=3,
               label='Raw' if i == 0 else None)
    ax.scatter(r['corr_r'], i, color='#61AFEF', s=70, zorder=3,
               marker='D' if (not np.isnan(r['corr_p']) and r['corr_p'] < 0.05) else 'o',
               label='Corrected' if i == 0 else None)

ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9)
abs_max = np.nanmax(np.abs(cog_power_df[['raw_r', 'corr_r']].values))
ax.set_xlim(-abs_max * 1.3, abs_max * 1.3)
ax.set_xlabel('Partial r with cognition (controlling age + sex)', fontsize=10)
ax.set_title('Cognition × band power: raw vs corrected  (\u25c6 = corrected p<0.05)', fontsize=11)
ax.grid(axis='x', alpha=0.3); ax.invert_yaxis()
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig(FIG_DIR / 'cog_power_raw_vs_corrected.png', dpi=150)
plt.close()
print("Saved: cog_power_raw_vs_corrected.png")