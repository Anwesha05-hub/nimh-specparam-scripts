"""
Raw vs corrected band power comparison
=======================================================
Depends on outputs from nimh_specparam.py:
  - results/nimh_subject_summary.csv
  - results/nimh_peaks.csv
  - nimh/*_psd.mat  (raw PSD files)
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

from specparam import SpectralModel
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from matplotlib.patches import Patch

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path('/Users/a.chakraborty.2@bham.ac.uk/Documents/specparam/nimh/subject PSD')
OUT_DIR  = Path('/Users/a.chakraborty.2@bham.ac.uk/Documents/specparam/results')
FIG_DIR  = OUT_DIR / 'figures'

BANDS = {
    'Delta\n(1-4 Hz)'  : (1,  4),
    'Theta\n(4-8 Hz)'  : (4,  8),
    'Alpha\n(8-13 Hz)' : (8,  13),
    'Beta\n(13-30 Hz)' : (13, 30),
    'Gamma\n(30-45 Hz)': (30, 45),
}

# ── load CSV outputs from nimh_specparam.py ───────────────────────────────────
summary  = pd.read_csv(OUT_DIR / 'nimh_subject_summary.csv').dropna(subset=['age'])
df_peaks = pd.read_csv(OUT_DIR / 'nimh_peaks.csv')
df_ap    = pd.read_csv(OUT_DIR / 'nimh_aperiodic.csv')

n_subs    = len(summary)
ages      = summary['age'].values
sexes     = summary['sex'].values
exponents = summary['mean_exponent'].values
colors    = ['#E06C75' if s == 'female' else '#61AFEF' for s in sexes]

# ── load raw PSDs once per subject ────────────────────────────────────────────
def load_psd(filepath):
    with h5py.File(filepath, 'r') as f:
        freq = f['freq_out'][:].squeeze()
        pow_ = f['pow_out'][:].T       # (n_channels x n_freqs)
    return freq, pow_

pow_data = {}
freq     = None
for sub in summary['subject']:
    f, p = load_psd(DATA_DIR / f'{sub}_psd.mat')
    pow_data[sub] = p
    if freq is None:
        freq = f

##function to compute mean log10 power in a frequency band throughout the script
#Band edges inclusive on both ends (needs changing?)
def _raw_band(lo, hi):
    """Log-transform per channel first, then average — mean(log10(x))"""
    idx = (freq >= lo) & (freq <= hi)
    return np.array([np.log10(pow_data[s][:, idx]).mean()
                     for s in summary['subject']])

# read corrected peak power directly from summary CSV as already computed via get_band_peak_group
band_map = {
    'Delta\n(1-4 Hz)'  : 'delta',
    'Theta\n(4-8 Hz)'  : 'theta',
    'Alpha\n(8-13 Hz)' : 'alpha',
    'Beta\n(13-30 Hz)' : 'beta',
    'Gamma\n(30-45 Hz)': 'gamma',}

BANDS_CORR = ['delta', 'theta', 'alpha', 'beta', 'gamma']

corr_pw = {band: summary[f'{band}_PW'].values
    for band in BANDS_CORR}                     #corrected peak power for each band (previosuly computed)

# n_ch per band also already in summary — use for reporting
corr_nch = {band: summary[f'{band}_n_ch'].values
    for band in BANDS_CORR}                     #no. of channels where peak is detected for each band

print(summary[['alpha_PW', 'beta_PW']].describe())
# =============================================================================
# PRINT R^2 TABLE
# =============================================================================
print("=" * 60)
print(f"{'Measure':<35} {'r':>7}  {'r2':>7}  {'p':>9}")

for band_label, (lo, hi) in BANDS.items():
    band_clean = band_label.replace('\n', ' ')
    raw_pow = _raw_band(lo, hi)
    r, p = pearsonr(ages, raw_pow)
    pstr = f'{p:.3f}' if p >= 0.001 else '<0.001'
    print(f"Raw {band_clean:<31} {r:>7.3f}  {r**2:>7.3f}  {pstr:>9}")

print()
for band_label, bname in band_map.items():
    band_clean = band_label.replace('\n', ' ')
    corr_pow = corr_pw[bname]
    valid = ~np.isnan(corr_pow)
    r, p  = pearsonr(ages[valid], corr_pow[valid]) #correlation b/w age & corrected peak power for each band
    pstr  = f'{p:.3f}' if p >= 0.001 else '<0.001'
    print(f"Corrected {band_clean:<27} {r:>7.3f}  {r**2:>7.3f}  {pstr:>9}")

print()
r, p = pearsonr(ages, exponents)
pstr = f'{p:.3f}' if p >= 0.001 else '<0.001'
print(f"{'Aperiodic exponent':<35} {r:>7.3f}  {r**2:>7.3f}  {pstr:>9}")

# =============================================================================
# FIGURE 1: raw vs corrected alpha scatter
# =============================================================================
raw_alpha  = _raw_band(8, 13)
corr_alpha = summary['alpha_PW'].values

raw_beta  = _raw_band(13, 30)
corr_beta = summary['beta_PW'].values

# shared y ranges for better comparison
_raw_all  = np.concatenate([raw_alpha[~np.isnan(raw_alpha)],   raw_beta[~np.isnan(raw_beta)]])
_corr_all = np.concatenate([corr_alpha[~np.isnan(corr_alpha)], corr_beta[~np.isnan(corr_beta)]])
raw_ylim  = (_raw_all.min()  - 0.05, _raw_all.max()  + 0.05)
corr_ylim = (_corr_all.min() - 0.02, _corr_all.max() + 0.02)

def _scatter_panel(ax, ages, y, colors, ylabel, title, ylim):
    valid = ~np.isnan(y)
    r, p  = pearsonr(ages[valid], y[valid])
    pstr  = f'p={p:.3f}' if p >= 0.001 else 'p<0.001'
    ax.scatter(ages[valid], y[valid],
               c=[colors[i] for i in np.where(valid)[0]],
               s=60, alpha=0.85, edgecolors='white', linewidths=0.5)
    ax.plot(np.unique(ages[valid]),
            np.poly1d(np.polyfit(ages[valid], y[valid], 1))(np.unique(ages[valid])),
            'k-', linewidth=1.5)
    ax.set_title(f'{title}\nr={r:.3f}, {pstr}', fontsize=11)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3)

legend_els = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#E06C75',
                     markersize=8, label='female'),
              Line2D([0],[0], marker='o', color='w', markerfacecolor='#61AFEF',
                     markersize=8, label='male')]

# Figure 1: alpha
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
_scatter_panel(axes[0], ages, raw_alpha,  colors, 'Alpha power (log10)',              'Raw alpha (8–13 Hz)',       raw_ylim)
_scatter_panel(axes[1], ages, corr_alpha, colors, 'Alpha peak power above aperiodic', 'Corrected alpha (8–13 Hz)', corr_ylim)
axes[1].legend(handles=legend_els, fontsize=9)
plt.suptitle(f'Effect of aperiodic correction on alpha power-age relationship  (n={n_subs})',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'raw_vs_corrected_alpha.png', dpi=150)
plt.close()
print(f"\nSaved: raw_vs_corrected_alpha.png")

# Figure 1b: beta
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
_scatter_panel(axes[0], ages, raw_beta,  colors, 'Beta power (log10)',              'Raw beta (13–30 Hz)',       raw_ylim)
_scatter_panel(axes[1], ages, corr_beta, colors, 'Beta peak power above aperiodic', 'Corrected beta (13–30 Hz)', corr_ylim)
axes[1].legend(handles=legend_els, fontsize=9)
plt.suptitle(f'Effect of aperiodic correction on beta power-age relationship  (n={n_subs})',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'raw_vs_corrected_beta.png', dpi=150)
plt.close()
print(f"Saved: raw_vs_corrected_beta.png")

# =============================================================================
# FIGURE 2: r for all bands — raw vs corrected bar chart
# =============================================================================
raw_r  = []
corr_r = []
for band_label, (lo, hi) in BANDS.items():
    raw_pow = _raw_band(lo, hi)
    raw_r.append(pearsonr(ages, raw_pow)[0])

    corr_pow = corr_pw[band_map[band_label]]
    valid    = ~np.isnan(corr_pow)
    corr_r.append(pearsonr(ages[valid], corr_pow[valid])[0])

x     = np.arange(len(BANDS))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width/2, raw_r,  width, label='Raw band power',
       color='#E06C75', alpha=0.8)
ax.bar(x + width/2, corr_r, width, label='Corrected peak power (specparam)',
       color='#61AFEF', alpha=0.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(list(BANDS.keys()), fontsize=10)
ax.set_ylabel('Pearson r with age')
ax.set_title(f'Age correlations: raw band power vs aperiodic-corrected power  (n={n_subs})')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'multiband_correction.png', dpi=150)
plt.close()
print(f"Saved: multiband_correction.png")

# =============================================================================
# FIGURE 3: Fit quality validation
# =============================================================================
all_r2 = df_ap['r_squared'].values
mean_r2_per_sub = df_ap.groupby('subject')['r_squared'].mean().sort_values().values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(all_r2, bins=40, color='#3B4B7A', alpha=0.8, edgecolor='white')
axes[0].axvline(np.mean(all_r2), color='red', linestyle='--',
                label=f'mean = {np.mean(all_r2):.3f}')
axes[0].axvline(0.90, color='orange', linestyle=':',
                label='exclusion threshold (R²=0.90)')
axes[0].set_xlabel('R² (goodness of fit)')
axes[0].set_ylabel('Channel count')
axes[0].set_title(f'specparam fit quality — all channels, all participants (n={n_subs})')
axes[0].legend()

axes[1].bar(range(len(mean_r2_per_sub)), mean_r2_per_sub, color='#3B4B7A', alpha=0.8)
axes[1].axhline(0.90, color='orange', linestyle=':', label='exclusion threshold')
axes[1].axhline(np.mean(mean_r2_per_sub), color='red', linestyle='--',
                label=f'mean = {np.mean(mean_r2_per_sub):.3f}')
axes[1].set_xlabel('Participant (sorted by R²)')
axes[1].set_ylabel('Mean R² across channels')
axes[1].set_title(f'Mean fit quality per participant (n={n_subs})')
axes[1].legend()
axes[1].set_ylim([0.85, 1.0])

plt.suptitle('specparam model fit validation', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fit_quality_all_subjects.png', dpi=150)
plt.close()
print("Saved: fit_quality_all_subjects.png")

# =============================================================================
# FIGURE 4: Youngest vs oldest participant PSD decomposition
# =============================================================================
FREQ_RANGE      = [1, 45]
PEAK_WIDTH_LIMS = [1, 8]
MAX_N_PEAKS     = 4
MIN_PEAK_HEIGHT = 0.03
APERIODIC_MODE  = 'fixed'       #identical parameters as main pipeline

def ap_line(fm, freqs):
    ap = fm.get_params('aperiodic')
    return np.power(10, ap[0] - ap[1] * np.log10(freqs))

youngest_row = summary.loc[summary['age'].idxmin()]
oldest_row   = summary.loc[summary['age'].idxmax()]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, row, color in [
    (axes[0], youngest_row, '#3B4B7A'),
    (axes[1], oldest_row,   '#B05030'),
]:
    sub      = row['subject']
    age      = row['age']
    pow_mean = pow_data[sub].mean(axis=0)

    fm = SpectralModel(peak_width_limits=PEAK_WIDTH_LIMS, max_n_peaks=MAX_N_PEAKS,
                       min_peak_height=MIN_PEAK_HEIGHT, aperiodic_mode=APERIODIC_MODE,
                       verbose=False)
    fm.fit(freq, pow_mean, FREQ_RANGE)
    ap_params = fm.get_params('aperiodic')

    ax.semilogy(freq, pow_mean, 'k-', linewidth=1.5, label='Mean PSD', alpha=0.8)
    ax.semilogy(freq, ap_line(fm, freq), '--', color=color, linewidth=2,
                label=f'Aperiodic fit (χ={ap_params[1]:.3f})')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title(f'Age {age:.0f}y  ({sub.replace("sub-", "")})')
    ax.legend(fontsize=9)
    ax.set_xlim(FREQ_RANGE)
    ax.grid(alpha=0.3)

plt.suptitle('Aperiodic decomposition: youngest vs oldest participant', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'young_vs_old_decomposition.png', dpi=150)
plt.close()
print("Saved: young_vs_old_decomposition.png")

# =============================================================================
# FIGURE 5: Grand average PSD — younger vs older age group
# =============================================================================
median_age = np.median(ages)
young_mask = summary['age'] <= median_age
old_mask   = summary['age']  > median_age
 
young_subs = summary.loc[young_mask, 'subject'].values
old_subs   = summary.loc[old_mask,   'subject'].values
 
grand_young = np.mean([pow_data[s].mean(axis=0) for s in young_subs], axis=0)
grand_old   = np.mean([pow_data[s].mean(axis=0) for s in old_subs],   axis=0)
 
mean_young_age = summary.loc[young_mask, 'age'].mean()
mean_old_age   = summary.loc[old_mask,   'age'].mean()
 
def reconstruct_aperiodic(subs, df_ap, freqs):
    """
    Reconstruct group aperiodic curve from mean per-channel parameters.
    Averages offset and exponent across all channels for subjects in subs,
    then generates the aperiodic line — consistent with per-channel pipeline.
    Returns: curve (log10 power), mean_offset, mean_exponent
    """
    group         = df_ap[df_ap['subject'].isin(subs)]
    mean_offset   = group['offset'].mean()
    mean_exponent = group['exponent'].mean()
    curve         = mean_offset - mean_exponent * np.log10(freqs)
    return curve, mean_offset, mean_exponent
 
freqs_ap = np.linspace(1, 45, 500)   # smooth frequency axis for reconstruction
 
ap_young, b_young, chi_young = reconstruct_aperiodic(young_subs, df_ap, freqs_ap)
ap_old,   b_old,   chi_old   = reconstruct_aperiodic(old_subs,   df_ap, freqs_ap)
 
fig, ax = plt.subplots(figsize=(10, 6))
 
ax.semilogy(freq, grand_young, color='#3B4B7A', linewidth=2,
            label=f'Younger (n={len(young_subs)}, mean age={mean_young_age:.0f}y)')
ax.semilogy(freq, grand_old,   color='#B05030', linewidth=2,
            label=f'Older (n={len(old_subs)}, mean age={mean_old_age:.0f}y)')
 
# overlay reconstructed aperiodic lines (in linear power for semilogy axis)
ax.semilogy(freqs_ap, np.power(10, ap_young), '--', color='#3B4B7A', alpha=0.7,
            linewidth=2, label=f'Aperiodic young (χ={chi_young:.3f})')
ax.semilogy(freqs_ap, np.power(10, ap_old),   '--', color='#B05030', alpha=0.7,
            linewidth=2, label=f'Aperiodic older (χ={chi_old:.3f})')
 
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (log scale)')
ax.set_title(
    f'Grand average PSD: younger vs older adults  (n={n_subs})\n'
    'Aperiodic fits reconstructed from mean per-channel parameters',
    fontsize=11
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim([1, 45])
 
plt.tight_layout()
plt.savefig(FIG_DIR / 'grand_average_decomposition.png', dpi=150)
plt.close()
print("Saved: grand_average_decomposition.png")

# =============================================================================
# FIGURE 6: Effect-size forest plot with bootstrap 95% CIs
# =============================================================================
def bootstrap_r_ci(x, y, n_boot=5000, ci=95):
    rng = np.random.default_rng(42)   #fixed seed for reproducibility (random number generator)
    n   = len(x) 
    rs  = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rs.append(pearsonr(x[idx], y[idx])[0])
    return np.percentile(rs, [(100-ci)/2, 100-(100-ci)/2])
 
offset     = summary['mean_offset'].values
raw_gamma  = _raw_band(30, 45)
corr_gamma = summary['gamma_PW'].values         #other variables already defined above 
 
measures = [
    # (label,               x_vals,    y_vals,     group)
    ('Aperiodic exponent',  ages,       exponents,  'aperiodic'),
    ('Aperiodic offset',    ages,       offset,     'aperiodic'),
    ('Raw alpha (8-13 Hz)', ages,       raw_alpha,  'raw'),
    ('Raw beta (13-30 Hz)', ages,       raw_beta,   'raw'),
    ('Raw gamma (30-45 Hz)',ages,       raw_gamma,  'raw'),
    ('Corrected alpha',     ages,       corr_alpha, 'corrected'),
    ('Corrected beta',      ages,       corr_beta,  'corrected'),
    ('Corrected gamma',     ages,       corr_gamma, 'corrected'),]
 
GROUP_COLORS = {'aperiodic': '#3B4B7A', 'raw': '#E06C75', 'corrected': '#61AFEF'}
 
r_vals, lo_vals, hi_vals, colors_es, labels_es, p_vals = [], [], [], [], [], []
 
for label, x, y, group in measures:
    valid   = ~np.isnan(x) & ~np.isnan(y)
    x_v, y_v = x[valid], y[valid]
    r, p    = pearsonr(x_v, y_v)
    lo, hi  = bootstrap_r_ci(x_v, y_v)
    r_vals.append(r)
    lo_vals.append(lo)
    hi_vals.append(hi)
    colors_es.append(GROUP_COLORS[group])
    labels_es.append(label)
    p_vals.append(p)

# ── Bonferroni correction across the full measure set ─────────────────────────
k = len(measures)
bonf_reject, p_bonf, _, _ = multipletests(p_vals, alpha=0.05, method='bonferroni')
print(f"\n{'='*65}")
print(f"Multiple comparison correction — Bonferroni (k={k}, α={0.05/k:.5f})")
print(f"{'='*65}")
print(f"{'Measure':<30} {'r':>7} {'p_raw':>9} {'p_bonf':>9} {'sig':>6}")
print(f"{'-'*65}")
for label, r, p_raw, p_c, reject in zip(labels_es, r_vals, p_vals, p_bonf, bonf_reject):
    sig = '**' if reject and p_raw < 0.01 else ('*' if reject else 'ns')
    print(f"{label:<30} {r:>7.3f} {p_raw:>9.4f} {p_c:>9.4f} {sig:>6}")
print(f"{'='*65}\n")

# significance tier:  2 = survives Bonferroni (●)   1 = uncorrected only (◆)   0 = ns (○)
sig_tier = [2 if reject else (1 if p_raw < 0.05 else 0)
            for p_raw, reject in zip(p_vals, bonf_reject)]

y_pos = np.arange(len(measures))
 
fig, ax = plt.subplots(figsize=(9, 6))
 
for i, (r, lo, hi, color, tier) in enumerate(
        zip(r_vals, lo_vals, hi_vals, colors_es, sig_tier)):
    if tier == 2:
        fmt, ms, mfc = 'o', 9,  color   # filled circle
    elif tier == 1:
        fmt, ms, mfc = 'D', 8,  color   # filled diamond
    else:
        fmt, ms, mfc = 'o', 8,  'white' # open circle
 
    ax.errorbar(r, i, xerr=[[r - lo], [hi - r]],
                fmt=fmt, color=color, ecolor=color,
                markerfacecolor=mfc, markeredgecolor=color,
                markersize=ms, capsize=4, linewidth=1.5,
                zorder=3)
 
ax.axvline(0, color='black', linewidth=0.9, linestyle='--', zorder=2)
ax.axvspan(-0.1, 0.1, alpha=0.06, color='grey', zorder=0)
 
abs_max = max(abs(v) for v in lo_vals + hi_vals)
abs_max = np.ceil(abs_max * 10) / 10
ax.set_xlim(-abs_max, abs_max)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_es, fontsize=10)
ax.set_xlabel('Pearson r with age  (95% bootstrap CI)', fontsize=11)
ax.set_title(f'Effect sizes: specparam parameters vs age  (n={n_subs})',
             fontweight='bold', fontsize=12)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()
 
legend_els = [
    Patch(color=GROUP_COLORS['aperiodic'], label='Aperiodic'),
    Patch(color=GROUP_COLORS['raw'],       label='Raw band power'),
    Patch(color=GROUP_COLORS['corrected'], label='Corrected peak power'),
    Line2D([0],[0], marker='o', color='grey', markerfacecolor='grey',
           markersize=9,  label='Bonferroni p<0.05', linestyle='none'),
    Line2D([0],[0], marker='D', color='grey', markerfacecolor='grey',
           markersize=8,  label='Uncorrected p<0.05 only', linestyle='none'),
    Line2D([0],[0], marker='o', color='grey', markerfacecolor='white',
           markersize=8,  label='ns', linestyle='none'),
]
ax.legend(handles=legend_els, fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(FIG_DIR / 'effect_sizes.png', dpi=150)
plt.close()
print("Saved: effect_sizes.png")

# =============================================================================
# BETA PEAK PREVALENCE VS AGE
# =============================================================================

# total channels fitted per subject
total_ch_per_sub = (
    df_ap.groupby('subject').size()
         .reindex(summary['subject'].values)
         .values.astype(float))

beta_n_ch  = corr_nch['beta'].astype(float)
beta_rate  = beta_n_ch / total_ch_per_sub   # proportion of channels with a detected beta peak

valid_prev = ~np.isnan(beta_rate)
r_count, p_count = pearsonr(ages[valid_prev], beta_n_ch[valid_prev])
r_rate,  p_rate  = pearsonr(ages[valid_prev], beta_rate[valid_prev])

print("\n" + "=" * 60)
print("BETA PEAK PREVALENCE vs AGE")
print(f"  Mean channels with beta peak: {beta_n_ch[valid_prev].mean():.1f} "
      f"/ {total_ch_per_sub[valid_prev].mean():.1f} total")
print(f"  Detection rate: {beta_rate[valid_prev].mean():.3f} ± {beta_rate[valid_prev].std():.3f}")
pstr = '<0.001' if p_count < 0.001 else f'{p_count:.3f}'
print(f"  r(age, beta_n_ch)  = {r_count:.3f},  p = {pstr}")
pstr = '<0.001' if p_rate < 0.001 else f'{p_rate:.3f}'
print(f"  r(age, beta_rate)  = {r_rate:.3f},  p = {pstr}")
print("=" * 60)

sex_legend_els = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#E06C75', markersize=8, label='female'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#61AFEF', markersize=8, label='male'),]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
_scatter_panel(axes[0], ages, beta_n_ch,
               colors, 'Channels with beta peak (n)',
               'Beta peak count vs age',
               (np.nanmin(beta_n_ch) - 1, np.nanmax(beta_n_ch) + 1))
_scatter_panel(axes[1], ages, beta_rate,
               colors, 'Beta peak detection rate (proportion)',
               'Beta peak prevalence vs age',
               (max(0, np.nanmin(beta_rate) - 0.05), min(1, np.nanmax(beta_rate) + 0.05)))
axes[1].legend(handles=sex_legend_els, fontsize=9)
plt.suptitle(f'Beta peak prevalence vs age  (n={n_subs})', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'beta_prevalence_vs_age.png', dpi=150)
plt.close()
print("Saved: beta_prevalence_vs_age.png")

# =============================================================================
# BETA PEAK POWER: AGE WITH PEAK COUNT AS COVARIATE
# =============================================================================
print("BETA PEAK POWER — AGE WITH N_CH AS COVARIATE")

df_reg = pd.DataFrame({
    'beta_PW':   corr_pw['beta'],
    'age':       ages,
    'beta_n_ch': beta_n_ch,
    'beta_rate': beta_rate,
}).dropna()

# simple model
B0 = smf.ols('beta_PW ~ age', data=df_reg).fit()
# age + peak count covariate
B1 = smf.ols('beta_PW ~ age + beta_n_ch', data=df_reg).fit()

print(f"\n  Simple (beta_PW ~ age):")
print(f"    beta_age = {B0.params['age']:.5f},  p = {B0.pvalues['age']:.4f},  R² = {B0.rsquared:.3f}")

print(f"\n  With covariate (beta_PW ~ age + beta_n_ch):")
print(f"    beta_age   = {B1.params['age']:.5f},  p = {B1.pvalues['age']:.4f}")
print(f"    beta_n_ch  = {B1.params['beta_n_ch']:.5f},  p = {B1.pvalues['beta_n_ch']:.4f}")
print(f"    R² = {B1.rsquared:.3f}  (Δ = {B1.rsquared - B0.rsquared:+.3f})")

# partial correlation: r(age, beta_PW | beta_n_ch) via residuals
resid_age = smf.ols('age ~ beta_n_ch',     data=df_reg).fit().resid
resid_pw  = smf.ols('beta_PW ~ beta_n_ch', data=df_reg).fit().resid
r_partial, p_partial = pearsonr(resid_age, resid_pw)
r_zero,    _         = pearsonr(df_reg['age'], df_reg['beta_PW'])
pstr = '<0.001' if p_partial < 0.001 else f'{p_partial:.3f}'
print(f"\n  Partial r(age, beta_PW | beta_n_ch) = {r_partial:.3f},  p = {pstr}")
print(f"  Zero-order r(age, beta_PW)           = {r_zero:.3f}")
print("=" * 60)
