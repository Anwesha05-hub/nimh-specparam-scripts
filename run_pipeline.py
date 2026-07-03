import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from specparam import SpectralGroupModel

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path('/Users/a.chakraborty.2@bham.ac.uk/Documents/specparam/nimh')
OUT_DIR  = Path('/Users/a.chakraborty.2@bham.ac.uk/Documents/specparam/results')
FIG_DIR  = OUT_DIR / 'figures'
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ── specparam settings ────────────────────────────────────────────────────────
FREQ_RANGE      = [1, 45]
PEAK_WIDTH_LIMS = [1, 8]
MAX_N_PEAKS     = 6
MIN_PEAK_HEIGHT = 0.05
APERIODIC_MODE  = 'fixed'   # change to 'knee' if you see a bend at low freqs

# ── helper: load one subject's PSD file ───────────────────────────────────────
def load_psd(filepath):
    with h5py.File(filepath, 'r') as f:
        freq   = f['freq_out'][:].squeeze()
        pow_   = f['pow_out'][:].T          # HDF5 from MATLAB is transposed
        labels = []
        for i in range(f['chan_labels'].shape[1]):
            ref  = f['chan_labels'][0, i]
            name = ''.join(chr(c) for c in f[ref][:].flatten())
            labels.append(name)
    return freq, pow_, labels

# ── main loop ─────────────────────────────────────────────────────────────────
all_rows  = []
all_peaks = []
mat_files = sorted(DATA_DIR.glob('*_psd.mat'))
print(f"Found {len(mat_files)} subjects\n")

for mat_path in mat_files:
    sub = mat_path.stem.replace('_psd', '')
    print(f"Fitting {sub} ...", end=' ', flush=True)

    freq, pow_, labels = load_psd(mat_path)

    fg = SpectralGroupModel(
        peak_width_limits = PEAK_WIDTH_LIMS,
        max_n_peaks       = MAX_N_PEAKS,
        min_peak_height   = MIN_PEAK_HEIGHT,
        aperiodic_mode    = APERIODIC_MODE,
        verbose           = False,
    )
    fg.fit(freq, pow_, FREQ_RANGE)

    # save model to disk so you can reload without re-fitting
    fg.save(str(OUT_DIR / f'{sub}_model'), save_results=True, save_settings=True)

    # ── extract aperiodic params & metrics ────────────────────────────────────
    ap = fg.get_params('aperiodic')   # (n_ch, 2): offset, exponent
    group_results = fg.results.group_results

    r2  = np.array([r.metrics['gof_rsquared'] for r in group_results])
    mae = np.array([r.metrics['error_mae']     for r in group_results])

    for ch_idx, label in enumerate(labels):
        all_rows.append({
            'subject'  : sub,
            'channel'  : label,
            'ch_index' : ch_idx,
            'offset'   : ap[ch_idx, 0],
            'exponent' : ap[ch_idx, 1],
            'r_squared': r2[ch_idx],
            'error_mae': mae[ch_idx],
        })

    # ── extract peaks (CF, PW, BW) per channel ────────────────────────────────
    for ch_idx, (label, res) in enumerate(zip(labels, group_results)):
        for peak in res.peak_converted:   # (n_peaks_found, 3)
            all_peaks.append({
                'subject'  : sub,
                'channel'  : label,
                'ch_index' : ch_idx,
                'CF'       : peak[0],
                'PW'       : peak[1],
                'BW'       : peak[2],
            })

    print(f"done  ({len(labels)} ch,  mean R²={r2.mean():.3f},  "
          f"n_peaks={sum(len(r.peak_converted) for r in group_results)})")

    # ── per-subject figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(sub, fontsize=12, fontweight='bold')

    # left: example channel fit (channel with median R²)
    mid_ch = int(np.argsort(r2)[len(r2) // 2])
    fm     = fg.get_model(mid_ch, regenerate=True)
    fm.plot(ax=axes[0])
    axes[0].set_title(f'Example fit — {labels[mid_ch]} (R²={r2[mid_ch]:.3f})')

    # right: R² distribution across channels
    axes[1].hist(r2, bins=30, color='steelblue', edgecolor='white')
    axes[1].axvline(r2.mean(), color='red', linestyle='--',
                    label=f'mean = {r2.mean():.3f}')
    axes[1].set_xlabel('R²')
    axes[1].set_ylabel('Channels')
    axes[1].set_title('Goodness of fit across channels')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / f'{sub}_summary.png', dpi=120)
    plt.close()

# ── save CSVs ─────────────────────────────────────────────────────────────────
df       = pd.DataFrame(all_rows)
df_peaks = pd.DataFrame(all_peaks)

df.to_csv(OUT_DIR / 'aperiodic_results.csv', index=False)
df_peaks.to_csv(OUT_DIR / 'peak_results.csv', index=False)
print(f"\nSaved aperiodic_results.csv  ({len(df)} rows)")
print(f"Saved peak_results.csv        ({len(df_peaks)} peaks total)")

# ── group-level figures ───────────────────────────────────────────────────────

# 1. Distribution of exponent and offset across all channels × subjects
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['exponent'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Aperiodic Exponent')
axes[0].set_ylabel('Count  (channels × subjects)')
axes[0].set_title('Distribution of Aperiodic Exponents')

axes[1].hist(df['offset'], bins=50, color='salmon', edgecolor='white')
axes[1].set_xlabel('Aperiodic Offset')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Aperiodic Offsets')

plt.tight_layout()
plt.savefig(FIG_DIR / 'group_aperiodic_distributions.png', dpi=120)
plt.close()

# 2. Mean exponent per subject (bar chart)
sub_mean = df.groupby('subject')[['exponent', 'r_squared']].mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(range(len(sub_mean)), sub_mean['exponent'], color='steelblue')
axes[0].set_xticks(range(len(sub_mean)))
axes[0].set_xticklabels(sub_mean['subject'], rotation=45, ha='right', fontsize=7)
axes[0].set_ylabel('Mean Aperiodic Exponent')
axes[0].set_title('Mean Aperiodic Exponent per Subject')

axes[1].bar(range(len(sub_mean)), sub_mean['r_squared'], color='mediumseagreen')
axes[1].set_xticks(range(len(sub_mean)))
axes[1].set_xticklabels(sub_mean['subject'], rotation=45, ha='right', fontsize=7)
axes[1].set_ylabel('Mean R²')
axes[1].set_title('Mean Goodness of Fit per Subject')
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig(FIG_DIR / 'group_per_subject.png', dpi=120)
plt.close()

# 3. Peak frequency distribution
if len(df_peaks) > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df_peaks['CF'], bins=88, range=(1, 45),
            color='mediumpurple', edgecolor='white')
    ax.set_xlabel('Center Frequency (Hz)')
    ax.set_ylabel('Peak count  (channels × subjects)')
    ax.set_title('Distribution of Peak Center Frequencies')
    for band, (lo, hi) in [('delta',(1,4)),('theta',(4,8)),
                            ('alpha',(8,13)),('beta',(13,30)),('gamma',(30,45))]:
        ax.axvspan(lo, hi, alpha=0.07)
        ax.text((lo+hi)/2, ax.get_ylim()[1]*0.92, band,
                ha='center', fontsize=8, color='grey')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'group_peak_frequencies.png', dpi=120)
    plt.close()

print("\nGroup figures saved to:", FIG_DIR)
print("\nAll done!")
