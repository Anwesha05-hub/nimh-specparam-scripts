"""
NIMH MEG Dataset — Spectral Parameterisation Pipeline
"""
# ── standard imports ──────────────────────────────────────────────────────────
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ── specparam imports (from tutorials 11, 12, failed_fits, data_exporting) ───
from specparam import SpectralGroupModel, Bands
from specparam import __version__ as specparam_version
from specparam.data.periodic import get_band_peak_group
from specparam.reports.methods import methods_report_info, methods_report_text

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR  = Path('/Users/a.chakraborty.2@bham.ac.uk/Documents/specparam/nimh/subject PSD')
META_FILE = Path('/Users/a.chakraborty.2@bham.ac.uk/Documents/NIMH participants.xlsx')
OUT_DIR   = Path('/Users/a.chakraborty.2@bham.ac.uk/Documents/specparam/results')
FIG_DIR   = OUT_DIR / 'figures'
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# Fitting settings
FREQ_RANGE      = [1, 40]   # trimmed range (to avoid high-freq noise)
PEAK_WIDTH_LIMS = [1, 8]
MAX_N_PEAKS     = 6
MIN_PEAK_HEIGHT = 0.05      # lowered from 0.05 to capture theta/small peaks
APERIODIC_MODE  = 'fixed'   # no bend, thus no knee mode. 

# Frequency bands (from tutorial 11)
bands = Bands({
    'delta' : [1,  4],
    'theta' : [4,  8],
    'alpha' : [8,  13],
    'beta'  : [13, 30],
    'gamma' : [30, 45],})

# =============================================================================
# Loading per subjects's HDF5 .mat file
# =============================================================================

def load_psd(filepath):
    with h5py.File(filepath, 'r') as f:
        freq   = f['freq_out'][:].squeeze()
        pow_   = f['pow_out'][:].T          # MATLAB HDF5 is column-major → transpose
        labels = []
        for i in range(f['chan_labels'].shape[1]):
            ref  = f['chan_labels'][0, i]
            name = ''.join(chr(c) for c in f[ref][:].flatten())
            labels.append(name)
    return freq, pow_, labels  #extract frequency vector, the per-channel power spectra, and channel-names

if __name__ == '__main__':
    print(f"specparam version: {specparam_version}")

# =============================================================================
# MAIN FITTING LOOP
# =============================================================================

ap_rows   = []   # aperiodic params per channel
peak_rows = []   # band peaks per channel
null_log  = []   # failed fits log

mat_files = sorted(DATA_DIR.glob('*_psd.mat'))
print(f"Found {len(mat_files)} subjects\n")

for idx, mat_path in enumerate(mat_files):
    sub = mat_path.stem.replace('_psd', '')
    print(f"Fitting {sub} ...", end=' ', flush=True)

    freq, pow_, labels = load_psd(mat_path)

    fg = SpectralGroupModel(
        peak_width_limits = PEAK_WIDTH_LIMS,
        max_n_peaks       = MAX_N_PEAKS,
        min_peak_height   = MIN_PEAK_HEIGHT,
        aperiodic_mode    = APERIODIC_MODE,
        verbose           = False,)
    fg.fit(freq, pow_, FREQ_RANGE, n_jobs=1)

    # ── print methods report once, from the first subject's fit (tutorial 12) ──
    if idx == 0:
            print("\n" + "="*70)
            print("METHODS REPORT")
            print("="*70)
            methods_report_info(fg)
            methods_report_text(fg)
        
    # ── check for failed fits (from manage/plot_failed_fits) ──────────────────
    n_null   = fg.results.n_null
    null_idx = fg.results.null_inds
    if n_null > 0:
        print(f"\n  WARNING: {n_null} null fits at channel indices {null_idx}")
        null_log.append({'subject': sub, 'n_null': n_null, 'null_indices': str(null_idx)})

    # ── aperiodic params : get_metrics/get_params pull arrays across the whole group fit ────────────────────
    offset   = fg.get_params('aperiodic', 'offset')
    exponent = fg.get_params('aperiodic', 'exponent')
    knee     = fg.get_params('aperiodic', 'knee') if APERIODIC_MODE == 'knee' else None
    r2     = fg.get_metrics('gof_rsquared')
    adj_r2 = fg.get_metrics('gof_adjrsquared')
    mae    = fg.get_metrics('error_mae')
    rmse   = np.sqrt(fg.get_metrics('error_mse'))   # RMSE = sqrt(MSE)

    # ── every fitted peak (not just one-per-band) for CF/PW distributions ─────
    # get_params('peak') returns [CF, PW, BW, ch_index] pooled across the group.
    all_pk = fg.get_params('peak')
        if all_pk is not None and np.size(all_pk):
            all_pk = np.atleast_2d(all_pk)
            n_peaks_per_ch = np.bincount(all_pk[:, 3].astype(int), minlength=len(labels)) #detected peak per channel is counted
            for cf, pw, bw, _ci in all_pk:
                allpeak_rows.append({'subject': sub, 'CF': cf, 'PW': pw, 'BW': bw}) 
        else:
            n_peaks_per_ch = np.zeros(len(labels), dtype=int)


    for ch_idx, label in enumerate(labels):
        ap_rows.append({
            'subject'  : sub,
            'channel'  : label,
            'ch_index' : ch_idx,
            'offset'   : ap[ch_idx, 0],
            'offset'    : offset[ch_idx],
            'knee'      : knee[ch_idx] if knee is not None else np.nan,
            'exponent'  : exponent[ch_idx],
            'r_squared' : r2[ch_idx],
            'adj_r_squared' : adj_r2[ch_idx],
            'r2_adj_gap' : r2[ch_idx] - adj_r2[ch_idx],   # overfitting indicator (tutorial 06)
            'error_mae' : mae[ch_idx],
            'rmse'      : rmse[ch_idx],
            'n_peaks'   : int(n_peaks_per_ch[ch_idx]),})

    # ── band peaks using get_band_peak_group (from tutorial 11) ──────────────
    # Returns (n_ch, 3) array of [CF, PW, BW]; NaN where no peak found
    for band_name, band_range in bands:
        band_peaks = get_band_peak_group(fg, band_range)  # (n_ch, 3)
        for ch_idx, label in enumerate(labels):
            cf, pw, bw = band_peaks[ch_idx]
            peak_rows.append({
                'subject'  : sub,
                'channel'  : label,
                'ch_index' : ch_idx,
                'band'     : band_name,
                'CF'       : cf if not np.isnan(cf) else np.nan,
                'PW'       : pw if not np.isnan(pw) else np.nan,
                'BW'       : bw if not np.isnan(bw) else np.nan,
            })

    # ── per-subject figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(sub, fontsize=12, fontweight='bold')

    best_ch = int(np.argmax(r2))
    fg.get_model(best_ch, regenerate=True).plot(ax=axes[0])
    axes[0].set_title(f'Best fit — {labels[best_ch]}  (R²={r2[best_ch]:.3f})') #plot best channel fit for subject

    worst_ch = int(np.argmin(r2))
    fg.get_model(worst_ch, regenerate=True).plot(ax=axes[1])
    axes[1].set_title(f'Worst fit — {labels[worst_ch]}  (R²={r2[worst_ch]:.3f})') #plot worst channel fit for subject

    axes[2].hist(r2, bins=30, color='steelblue', edgecolor='white')
    axes[2].axvline(r2.mean(), color='red', linestyle='--',
                    label=f'mean = {r2.mean():.3f}')
    axes[2].set_xlabel('R²')
    axes[2].set_ylabel('Channels')
    axes[2].set_title('Goodness of fit across channels')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / f'{sub}_summary.png', dpi=120)
    plt.close()

    print(f"done  ({len(labels)} ch,  mean R²={r2.mean():.3f},  null={n_null})")

# =============================================================================
# SAVE CHANNEL-LEVEL CSVs
# =============================================================================

df_ap    = pd.DataFrame(ap_rows)
df_peaks = pd.DataFrame(peak_rows)

df_ap.to_csv(OUT_DIR / 'nimh_aperiodic.csv', index=False)
df_peaks.to_csv(OUT_DIR / 'nimh_peaks.csv', index=False)
print(f"\nSaved nimh_aperiodic.csv  ({len(df_ap)} rows)")
print(f"Saved nimh_peaks.csv       ({len(df_peaks)} rows)")

if null_log:
    pd.DataFrame(null_log).to_csv(OUT_DIR / 'failed_fits.csv', index=False)
    print(f"Saved failed_fits.csv ({len(null_log)} subjects had null fits)")

# =============================================================================
# SUBJECT-LEVEL SUMMARY + AGE MERGE
# =============================================================================

# ── group goodness of fit report (Tutorial 06) ───────────────────────────────
print("\n" + "="*60)
print("GOODNESS OF FIT — all channels, all subjects")
print("="*60)
print(f"  Mean R²:              {df_ap['r_squared'].mean():.4f}")
print(f"  Median R²:            {df_ap['r_squared'].median():.4f}")
print(f"  Min R²:               {df_ap['r_squared'].min():.4f}")
print(f"  Channels with R²<0.90: {(df_ap['r_squared'] < 0.90).sum()}")
print(f"  Mean MAE:             {df_ap['error_mae'].mean():.5f}")
print(f"  Median MAE:           {df_ap['error_mae'].median():.5f}")
print("="*60 + "\n")

# aperiodic means
ap_summary = df_ap.groupby('subject').agg(
    mean_exponent = ('exponent',   'mean'),
    sd_exponent   = ('exponent',   'std'),
    mean_offset   = ('offset',     'mean'),
    sd_offset     = ('offset',     'std'),
    mean_r2       = ('r_squared',  'mean'),
    mean_mae      = ('error_mae',  'mean'),
    n_channels    = ('channel',    'count'),
).reset_index()

# per-band peak means (CF and PW only; drop channels with no peak in band)
for band_name in ['delta','theta','alpha','beta','gamma']:
    band = df_peaks[df_peaks['band'] == band_name]
    bsum = (band.dropna(subset=['CF'])
                .groupby('subject')
                .agg(**{
                    f'{band_name}_CF'    : ('CF', 'mean'),
                    f'{band_name}_PW'    : ('PW', 'mean'),
                    f'{band_name}_n_ch'  : ('CF', 'count'),
                })
                .reset_index())
    ap_summary = ap_summary.merge(bsum, on='subject', how='left')

# merge age + sex from metadata
meta = pd.read_excel(META_FILE, header=1)[['participant_id','age','sex']]
meta = meta.rename(columns={'participant_id': 'subject'})
summary = ap_summary.merge(meta, on='subject', how='left')
summary['subject_id'] = summary['subject'].str.replace('sub-', '')

summary.to_csv(OUT_DIR / 'nimh_subject_summary.csv', index=False)
print(f"Saved nimh_subject_summary.csv  ({len(summary)} subjects)")

# =============================================================================
# GROUP FIGURES
# =============================================================================

# ── 0. Goodness of fit distributions (Tutorial 06) ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Goodness of Fit — All Channels & Subjects', fontweight='bold')

axes[0].hist(df_ap['r_squared'], bins=50, color='#3B4B7A', edgecolor='white', alpha=0.85)
axes[0].axvline(df_ap['r_squared'].mean(),   color='red',    linestyle='--',
                label=f"mean={df_ap['r_squared'].mean():.3f}")
axes[0].axvline(df_ap['r_squared'].median(), color='orange', linestyle=':',
                label=f"median={df_ap['r_squared'].median():.3f}")
axes[0].axvline(0.90, color='black', linestyle=':', linewidth=1.2, label='R²=0.90')
axes[0].set_xlabel('R²'); axes[0].set_ylabel('Count')
axes[0].set_title('R² distribution'); axes[0].legend(fontsize=9)

axes[1].hist(df_ap['error_mae'], bins=50, color='#98C379', edgecolor='white', alpha=0.85)
axes[1].axvline(df_ap['error_mae'].mean(), color='red', linestyle='--',
                label=f"mean={df_ap['error_mae'].mean():.5f}")
axes[1].set_xlabel('MAE'); axes[1].set_ylabel('Count')
axes[1].set_title('MAE distribution'); axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / 'goodness_of_fit.png', dpi=150)
plt.close()

# ── 1. Aperiodic distributions ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Aperiodic Component — All Channels & Subjects', fontweight='bold')

for ax, col, label, color in [
    (axes[0], 'exponent', 'Aperiodic Exponent (χ)', 'steelblue'),
    (axes[1], 'offset',   'Aperiodic Offset (b)',    'salmon'),
]:
    ax.hist(df_ap[col], bins=60, color=color, edgecolor='white', linewidth=0.5)
    ax.axvline(df_ap[col].mean(), color='red',    linestyle='--', linewidth=1.5,
               label=f"mean={df_ap[col].mean():.3f}")
    ax.axvline(df_ap[col].median(), color='orange', linestyle=':',  linewidth=1.5,
               label=f"median={df_ap[col].median():.3f}")
    ax.set_xlabel(label); ax.set_ylabel('Count'); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / 'aperiodic_distributions.png', dpi=150)
plt.close()

# ── 2. Peak CF distribution per band ─────────────────────────────────────────
band_colors = {'delta':'#4878d0','theta':'#6acc65','alpha':'#d65f5f',
               'beta':'#ee854a','gamma':'#956cb4'}
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
fig.suptitle('Peak Centre Frequency by Band — All Channels & Subjects', fontweight='bold')

for ax, (band_name, band_range) in zip(axes, bands):
    band_cf = df_peaks[(df_peaks['band'] == band_name)]['CF'].dropna()
    ax.hist(band_cf, bins=30, color=band_colors[band_name], edgecolor='white')
    ax.axvline(band_cf.mean(), color='black', linestyle='--', linewidth=1.2,
               label=f'mean={band_cf.mean():.1f} Hz')
    ax.set_xlabel('CF (Hz)')
    ax.set_ylabel('Count')
    ax.set_title(f'{band_name.capitalize()}\n({band_range[0]}–{band_range[1]} Hz)\nn={len(band_cf)}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / 'peak_cf_by_band.png', dpi=150)
plt.close()

# ── 3. Age correlations (4 metrics) ──────────────────────────────────────────
age_metrics = [
    ('mean_exponent', 'Aperiodic Exponent (χ)', 'steelblue'),
    ('mean_offset',   'Aperiodic Offset (b)',    'salmon'),
    ('alpha_CF',      'Alpha Peak CF (Hz)',       'mediumpurple'),
    ('alpha_PW',      'Alpha Peak Power',         'mediumseagreen'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Specparam Parameters vs Age  (NIMH MEG, n={len(summary)})',
             fontsize=13, fontweight='bold')

for ax, (col, label, color) in zip(axes.flat, age_metrics):
    sub = summary[['age', col, 'sex']].dropna()
    x, y = sub['age'].values, sub[col].values
    y_z  = (y - y.mean()) / y.std()   # z-score so all panels share same axis

    # colour by sex
    for sex, marker, ec in [('female', 'o', 'white'), ('male', '^', 'white')]:
        mask = sub['sex'] == sex
        ax.scatter(x[mask.values], y_z[mask.values], color=color, alpha=0.8,
                   s=70, marker=marker, edgecolors=ec, linewidths=0.8,
                   label=sex, zorder=3)

    # regression on z-scored values
    slope, intercept, r, p, _ = stats.linregress(x, y_z)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, slope * xline + intercept, color='black', linewidth=1.5, zorder=2)

    pstr = f'p={p:.3f}' if p >= 0.001 else 'p<0.001'
    ax.set_title(f'{label}\nr={r:.3f},  {pstr}', fontsize=10)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('z-score')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# apply same ylim to all panels
all_axes = list(axes.flat)
y_min = min(ax.get_ylim()[0] for ax in all_axes)
y_max = max(ax.get_ylim()[1] for ax in all_axes)
for ax in all_axes:
    ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig(FIG_DIR / 'age_correlations.png', dpi=150)
plt.close()

# ── 4. Exponent boxplot per subject ──────────────────────────────────────────
subs     = sorted(df_ap['subject'].unique())
sub_data = [df_ap.loc[df_ap['subject'] == s, 'exponent'].values for s in subs]
labels_s = [s.replace('sub-', '') for s in subs]
 
fig, ax = plt.subplots(figsize=(16, 5))
bp = ax.boxplot(sub_data, patch_artist=True, showfliers=False,
                medianprops=dict(color='red', linewidth=1.5))
for patch in bp['boxes']:
    patch.set_facecolor('steelblue'); patch.set_alpha(0.6)

ax.set_xticks(range(1, len(subs) + 1))
ax.set_xticklabels(labels_s, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Aperiodic Exponent (χ)')
ax.set_title('Aperiodic Exponent per Subject  (all channels, no outliers)')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'exponent_per_subject.png', dpi=150)
plt.close()

print("\nAll figures saved to:", FIG_DIR)
print("\n=== DONE ===")
print(summary[['subject_id','age','mean_exponent','mean_offset',
               'alpha_CF','alpha_PW','mean_r2']].to_string(index=False))

"""
Outputs
-------
results/nimh_aperiodic.csv    — offset, knee (NaN in fixed mode), exponent, r_squared, adj_r_squared, r2_adj_gap, error_mae, rmse, n_peaks
                                per channel per subject
results/nimh_peaks.csv        — CF, PW, BW per band per channel per subject
results/nimh_subject_summary.csv — subject-level means + age merged
results/figures/              — per-subject and group figures
"""
