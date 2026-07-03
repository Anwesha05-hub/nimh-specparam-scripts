"""
nimh_specparam_walkthrough.py
==============================
Applies the specparam pipeline to NIMH MEG data following the exact structure
of the tutorial notebooks:

  Tutorial 03 — Algorithm (7 steps)
  Tutorial 04 — Periodic fitting
  Tutorial 05 — Aperiodic fitting
  Tutorial 06 — Metrics & goodness of fit
  Tutorial 08 — Group fits
  Tutorial 11 — Further analysis (band peaks)
  Tutorial 12 — Reporting

One representative subject/channel is used to illustrate each algorithm step.
All subjects are then fitted as a group, matching nimh_specparam.py outputs.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from specparam import SpectralModel, SpectralGroupModel, Bands
from specparam import __version__ as specparam_version
from specparam.data.periodic import get_band_peak_group
from specparam.reports.methods import methods_report_info, methods_report_text
from specparam.metrics.definitions import check_metrics

print(f"specparam version: {specparam_version}")

# =============================================================================
# PATHS & SETTINGS
# =============================================================================

DATA_DIR  = Path('/Users/a.chakraborty.2@bham.ac.uk/Documents/specparam/nimh/subject PSD')
META_FILE = Path('/Users/a.chakraborty.2@bham.ac.uk/Documents/NIMH participants.xlsx')
OUT_DIR   = Path('/Users/a.chakraborty.2@bham.ac.uk/Documents/specparam/results')
FIG_DIR   = OUT_DIR / 'figures' / 'walkthrough'
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

FREQ_RANGE      = [1, 45]
PEAK_WIDTH_LIMS = [1, 8]
MAX_N_PEAKS     = 6
MIN_PEAK_HEIGHT = 0.03
APERIODIC_MODE  = 'fixed'

bands = Bands({
    'delta' : [1,  4],
    'theta' : [4,  8],
    'alpha' : [8,  13],
    'beta'  : [13, 30],
    'gamma' : [30, 45],
})

# =============================================================================
# HELPER: load one subject's HDF5 .mat file
# =============================================================================

def load_psd(filepath):
    with h5py.File(filepath, 'r') as f:
        freq   = f['freq_out'][:].squeeze()
        pow_   = f['pow_out'][:].T          # MATLAB HDF5 transpose
        labels = []
        for i in range(f['chan_labels'].shape[1]):
            ref  = f['chan_labels'][0, i]
            name = ''.join(chr(c) for c in f[ref][:].flatten())
            labels.append(name)
    return freq, pow_, labels

# =============================================================================
# TUTORIAL 12 — REPORTING: methods text
# =============================================================================

print("\n" + "="*70)
print("TUTORIAL 12 — Methods report")
print("="*70)

_demo = SpectralModel(
    peak_width_limits=PEAK_WIDTH_LIMS, max_n_peaks=MAX_N_PEAKS,
    min_peak_height=MIN_PEAK_HEIGHT, aperiodic_mode=APERIODIC_MODE,
    verbose=False,
)
methods_report_info(_demo)
methods_report_text(_demo)

# =============================================================================
# TUTORIAL 06 — METRICS: check available metrics
# =============================================================================

print("\n" + "="*70)
print("TUTORIAL 06 — Available specparam metrics")
print("="*70)
check_metrics()

# =============================================================================
# PICK REPRESENTATIVE SUBJECT & CHANNEL
# (median R² subject, median R² channel within that subject)
# Used for the step-by-step algorithm illustration (Tutorial 03)
# =============================================================================

mat_files = sorted(DATA_DIR.glob('*_psd.mat'))
print(f"\nFound {len(mat_files)} subjects")

# quick pass to find median-quality subject
r2_means = {}
for mat_path in mat_files:
    sub = mat_path.stem.replace('_psd', '')
    freq, pow_, labels = load_psd(mat_path)
    fg_quick = SpectralGroupModel(
        peak_width_limits=PEAK_WIDTH_LIMS, max_n_peaks=MAX_N_PEAKS,
        min_peak_height=MIN_PEAK_HEIGHT, aperiodic_mode=APERIODIC_MODE,
        verbose=False,
    )
    fg_quick.fit(freq, pow_, FREQ_RANGE)
    r2s = np.array([r.metrics['gof_rsquared'] for r in fg_quick.results.group_results])
    r2_means[sub] = (r2s.mean(), fg_quick, freq, pow_, labels, r2s)

median_sub  = sorted(r2_means, key=lambda s: r2_means[s][0])[len(r2_means)//2]
_, fg_ex, freq, pow_ex, labels_ex, r2s_ex = r2_means[median_sub]
mid_ch = int(np.argsort(r2s_ex)[len(r2s_ex)//2])
example_spectrum = pow_ex[mid_ch]
example_label    = labels_ex[mid_ch]
print(f"\nExample subject: {median_sub}  |  channel: {example_label}"
      f"  |  R²={r2s_ex[mid_ch]:.3f}")

# =============================================================================
# TUTORIAL 03 — ALGORITHM: 7-step walkthrough on example channel
# =============================================================================

print("\n" + "="*70)
print("TUTORIAL 03 — Algorithm walkthrough (7 steps)")
print("="*70)

fm = SpectralModel(
    peak_width_limits=PEAK_WIDTH_LIMS, max_n_peaks=MAX_N_PEAKS,
    min_peak_height=MIN_PEAK_HEIGHT, aperiodic_mode=APERIODIC_MODE,
    verbose=False,
)

# Add data (Tutorial 03: data can be added independently of fitting)
fm.add_data(freq, example_spectrum, FREQ_RANGE)

# Fit full model (steps 1–7 happen internally; we then recreate each for plots)
fm.fit(freq, example_spectrum, FREQ_RANGE)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(f'Tutorial 03 — Algorithm steps\n{median_sub} | {example_label}',
             fontweight='bold', fontsize=13)

plot_freqs = fm.data.freqs
log_freq   = np.log10(plot_freqs)

# Step 0: raw spectrum
ax = axes[0, 0]
ax.semilogy(plot_freqs, np.power(10, fm.data.power_spectrum), 'k-', linewidth=1.5)
ax.set_title('Step 0: Raw power spectrum')
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Power'); ax.grid(alpha=0.3)

# Step 1: initial aperiodic fit
init_ap = fm.results.model.get_component('aperiodic')

ax = axes[0, 1]
ax.semilogy(plot_freqs, np.power(10, fm.data.power_spectrum), 'k-', linewidth=1.5,
            label='PSD', alpha=0.6)
ax.semilogy(plot_freqs, np.power(10, init_ap), 'b--', linewidth=2,
            label='Aperiodic fit')
ax.set_title('Step 1: Initial aperiodic fit')
ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Step 2: flattened spectrum
flat_spec = fm.data.power_spectrum - init_ap

ax = axes[0, 2]
ax.plot(plot_freqs, flat_spec, 'k-', linewidth=1.5)
ax.axhline(0, color='blue', linestyle='--', linewidth=1)
ax.set_title('Step 2: Flattened spectrum\n(PSD − aperiodic)')
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('log power (residual)'); ax.grid(alpha=0.3)

# Step 3: detected peaks
peak_fit = fm.results.model.get_component('peak')

ax = axes[0, 3]
ax.plot(plot_freqs, flat_spec, 'k-', linewidth=1.5, label='Flattened', alpha=0.7)
ax.plot(plot_freqs, peak_fit,  'r-', linewidth=2,   label='Peak fit')
ax.axhline(0, color='blue', linestyle='--', linewidth=0.8)
ax.set_title('Step 3 & 4: Peak detection\n& full peak fit')
ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Step 5: peak-removed spectrum
peak_removed = fm.get_data('aperiodic')

ax = axes[1, 0]
ax.semilogy(plot_freqs, np.power(10, fm.data.power_spectrum), 'k-',
            linewidth=1.5, label='Original', alpha=0.5)
ax.semilogy(plot_freqs, np.power(10, peak_removed), 'purple',
            linewidth=1.5, label='Peak removed')
ax.set_title('Step 5: Peak-removed spectrum')
ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Step 6: final aperiodic fit
final_ap = fm.results.model.get_component('aperiodic')

ax = axes[1, 1]
ax.semilogy(plot_freqs, np.power(10, peak_removed), 'purple',
            linewidth=1.5, label='Peak removed', alpha=0.7)
ax.semilogy(plot_freqs, np.power(10, final_ap), 'b-',
            linewidth=2, label='Final aperiodic fit')
ap_p = fm.get_params('aperiodic')
ax.set_title(f'Step 6: Final aperiodic fit\noffset={ap_p[0]:.3f}, χ={ap_p[1]:.3f}')
ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Step 7: full model
full_model = fm.results.model.modeled_spectrum

ax = axes[1, 2]
ax.semilogy(plot_freqs, np.power(10, fm.data.power_spectrum), 'k-',
            linewidth=1.5, label='PSD', alpha=0.7)
ax.semilogy(plot_freqs, np.power(10, full_model), 'r-',
            linewidth=2, label='Full model fit')
ax.semilogy(plot_freqs, np.power(10, final_ap), 'b--',
            linewidth=1.5, label='Aperiodic')
r2_val  = fm.get_metrics('gof', 'squared')
mae_val = fm.get_metrics('error', 'mae')
ax.set_title(f'Step 7: Full model\nR²={r2_val:.3f}, MAE={mae_val:.4f}')
ax.set_xlabel('Frequency (Hz)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Step 7b: standard specparam plot
ax = axes[1, 3]
fm.plot(ax=ax)
ax.set_title('Step 7: specparam model plot')

plt.tight_layout()
plt.savefig(FIG_DIR / 'tutorial03_algorithm_steps.png', dpi=150)
plt.close()
print("Saved: tutorial03_algorithm_steps.png")

# Print model results (Tutorial 03 step)
print("\n--- fm.print_results() ---")
fm.print_results()

# =============================================================================
# TUTORIAL 06 — METRICS: goodness of fit on example channel
# =============================================================================

print("\n" + "="*70)
print("TUTORIAL 06 — Metrics & goodness of fit")
print("="*70)

metrics = ['error_mae', 'error_mse', 'gof_rsquared']
fm_metrics = SpectralModel(
    peak_width_limits=PEAK_WIDTH_LIMS, max_n_peaks=MAX_N_PEAKS,
    min_peak_height=MIN_PEAK_HEIGHT, aperiodic_mode=APERIODIC_MODE,
    metrics=metrics, verbose=False,
)
fm_metrics.fit(freq, example_spectrum, FREQ_RANGE)

print(f"MAE:  {fm_metrics.get_metrics('error', 'mae'):.5f}")
print(f"MSE:  {fm_metrics.get_metrics('error', 'mse'):.5f}")
print(f"R²:   {fm_metrics.get_metrics('gof', 'squared'):.5f}")
print("\nFull metrics object:")
print(fm_metrics.results.metrics.results)

# =============================================================================
# TUTORIAL 04 — PERIODIC FITTING: inspect detected peaks
# =============================================================================

print("\n" + "="*70)
print("TUTORIAL 04 — Periodic component")
print("="*70)

peaks = fm.get_params('peak')
if peaks is not None and len(np.atleast_1d(peaks)) > 0:
    peaks = np.atleast_2d(peaks)
    print(f"Detected peaks in {example_label}:")
    print(f"  {'CF (Hz)':>10} {'PW':>8} {'BW (Hz)':>10}")
    for row in peaks:
        print(f"  {row[0]:>10.2f} {row[1]:>8.4f} {row[2]:>10.2f}")
else:
    print(f"No peaks detected in {example_label}")

# =============================================================================
# TUTORIAL 05 — APERIODIC FITTING: fixed vs knee
# (illustrative only — we use fixed for all NIMH fits)
# =============================================================================

print("\n" + "="*70)
print("TUTORIAL 05 — Aperiodic modes (fixed vs knee, illustrative)")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, mode, color in [(axes[0], 'fixed', '#3B4B7A'), (axes[1], 'knee', '#B05030')]:
    fm_ap = SpectralModel(
        peak_width_limits=PEAK_WIDTH_LIMS, max_n_peaks=MAX_N_PEAKS,
        min_peak_height=MIN_PEAK_HEIGHT, aperiodic_mode=mode,
        verbose=False,
    )
    try:
        fm_ap.fit(freq, example_spectrum, FREQ_RANGE)
        ap    = fm_ap.get_params('aperiodic')
        r2_ap = fm_ap.get_metrics('gof', 'squared')
        label = f'{mode}: offset={ap[0]:.2f}, χ={ap[-1]:.2f}  R²={r2_ap:.3f}'
        fm_ap.plot(ax=ax)
        ax.set_title(f'Aperiodic mode: {mode}\n{label}', fontsize=10)
    except Exception as e:
        ax.set_title(f'{mode} — fit failed: {e}', fontsize=9)

fig.suptitle(f'Tutorial 05 — Aperiodic modes\n{median_sub} | {example_label}',
             fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'tutorial05_aperiodic_modes.png', dpi=150)
plt.close()
print("Saved: tutorial05_aperiodic_modes.png")
print("(fixed mode used for all NIMH analysis — knee caused degenerate fits)")

# =============================================================================
# TUTORIAL 08 — GROUP FITS: full pipeline across all subjects
# =============================================================================

print("\n" + "="*70)
print("TUTORIAL 08 — Group fits (all subjects)")
print("="*70)

ap_rows, peak_rows, null_log = [], [], []
mat_files = sorted(DATA_DIR.glob('*_psd.mat'))

for mat_path in mat_files:
    sub = mat_path.stem.replace('_psd', '')
    print(f"  {sub} ...", end=' ', flush=True)

    freq, pow_, labels = load_psd(mat_path)

    fg = SpectralGroupModel(
        peak_width_limits=PEAK_WIDTH_LIMS, max_n_peaks=MAX_N_PEAKS,
        min_peak_height=MIN_PEAK_HEIGHT, aperiodic_mode=APERIODIC_MODE,
        verbose=False,
    )
    fg.fit(freq, pow_, FREQ_RANGE)

    n_null   = fg.results.n_null
    null_idx = fg.results.null_inds
    if n_null > 0:
        print(f"\n    WARNING: {n_null} null fits at indices {null_idx}")
        null_log.append({'subject': sub, 'n_null': n_null, 'null_indices': str(null_idx)})

    ap    = fg.get_params('aperiodic')
    r2    = np.array([r.metrics['gof_rsquared'] for r in fg.results.group_results])
    mae   = np.array([r.metrics['error_mae']     for r in fg.results.group_results])

    for ch_idx, label in enumerate(labels):
        ap_rows.append({
            'subject': sub, 'channel': label, 'ch_index': ch_idx,
            'offset': ap[ch_idx, 0], 'exponent': ap[ch_idx, 1],
            'r_squared': r2[ch_idx], 'error_mae': mae[ch_idx],
        })

    # Tutorial 11 — band peaks
    for band_name, band_range in bands:
        band_peaks = get_band_peak_group(fg, band_range)
        for ch_idx, label in enumerate(labels):
            cf, pw, bw = band_peaks[ch_idx]
            peak_rows.append({
                'subject': sub, 'channel': label, 'ch_index': ch_idx,
                'band': band_name, 'CF': cf, 'PW': pw, 'BW': bw,
            })

    print(f"done  ({len(labels)} ch, mean R²={r2.mean():.3f}, null={n_null})")

# =============================================================================
# TUTORIAL 06 — GROUP METRICS: R² distribution across all channels
# =============================================================================

df_ap    = pd.DataFrame(ap_rows)
df_peaks = pd.DataFrame(peak_rows)

print(f"\n--- Tutorial 06: Group goodness of fit ---")
print(f"Mean R²  across all channels & subjects: {df_ap['r_squared'].mean():.4f}")
print(f"Median R²:                               {df_ap['r_squared'].median():.4f}")
print(f"Channels with R² < 0.90:                 {(df_ap['r_squared'] < 0.90).sum()}")
print(f"Mean MAE:                                {df_ap['error_mae'].mean():.5f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_ap['r_squared'], bins=50, color='#3B4B7A', edgecolor='white', alpha=0.8)
axes[0].axvline(df_ap['r_squared'].mean(),   color='red',    linestyle='--',
                label=f"mean={df_ap['r_squared'].mean():.3f}")
axes[0].axvline(df_ap['r_squared'].median(), color='orange', linestyle=':',
                label=f"median={df_ap['r_squared'].median():.3f}")
axes[0].axvline(0.90, color='black', linestyle=':', linewidth=1.2, label='R²=0.90 threshold')
axes[0].set_xlabel('R²'); axes[0].set_ylabel('Count')
axes[0].set_title('Tutorial 06 — R² distribution (all channels, all subjects)')
axes[0].legend()

axes[1].hist(df_ap['error_mae'], bins=50, color='#98C379', edgecolor='white', alpha=0.8)
axes[1].axvline(df_ap['error_mae'].mean(), color='red', linestyle='--',
                label=f"mean={df_ap['error_mae'].mean():.5f}")
axes[1].set_xlabel('MAE'); axes[1].set_ylabel('Count')
axes[1].set_title('Tutorial 06 — MAE distribution (all channels, all subjects)')
axes[1].legend()

plt.suptitle('Goodness of fit metrics — NIMH MEG dataset', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'tutorial06_group_metrics.png', dpi=150)
plt.close()
print("Saved: tutorial06_group_metrics.png")

# =============================================================================
# SAVE CSVs + SUBJECT SUMMARY (same as nimh_specparam.py)
# =============================================================================

df_ap.to_csv(OUT_DIR / 'nimh_aperiodic.csv',  index=False)
df_peaks.to_csv(OUT_DIR / 'nimh_peaks.csv',   index=False)
print(f"\nSaved nimh_aperiodic.csv  ({len(df_ap)} rows)")
print(f"Saved nimh_peaks.csv      ({len(df_peaks)} rows)")

if null_log:
    pd.DataFrame(null_log).to_csv(OUT_DIR / 'failed_fits.csv', index=False)

ap_summary = df_ap.groupby('subject').agg(
    mean_exponent=('exponent', 'mean'), sd_exponent=('exponent', 'std'),
    mean_offset=('offset', 'mean'),     sd_offset=('offset', 'std'),
    mean_r2=('r_squared', 'mean'),      n_channels=('channel', 'count'),
).reset_index()

for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
    band = df_peaks[df_peaks['band'] == band_name]
    bsum = (band.dropna(subset=['CF'])
                .groupby('subject')
                .agg(**{
                    f'{band_name}_CF'   : ('CF', 'mean'),
                    f'{band_name}_PW'   : ('PW', 'mean'),
                    f'{band_name}_n_ch' : ('CF', 'count'),
                }).reset_index())
    ap_summary = ap_summary.merge(bsum, on='subject', how='left')

meta    = pd.read_excel(META_FILE, header=1)[['participant_id', 'age', 'sex']]
meta    = meta.rename(columns={'participant_id': 'subject'})
summary = ap_summary.merge(meta, on='subject', how='left')
summary['subject_id'] = summary['subject'].str.replace('sub-', '')

summary.to_csv(OUT_DIR / 'nimh_subject_summary.csv', index=False)
print(f"Saved nimh_subject_summary.csv  ({len(summary)} subjects)")

print("\n=== DONE ===")
