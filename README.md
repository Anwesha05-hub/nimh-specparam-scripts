# NIMH specparam scripts

Spectral parameterisation (specparam/FOOOF) analysis of the NIMH MEG resting-state
dataset — aperiodic (offset, exponent) and periodic (band peak) fits per subject,
plus downstream analyses relating those fits to age, sex, and cognitive measures.

## Data needed (not included in this repo)

- `subject PSD/*_psd.mat` — per-subject PSD, exported from MATLAB (HDF5 format)
- `NIMH participants.xlsx` — participant metadata (age, sex)
- `kbit2_vas.csv`, `nih_toolbox.csv` — cognitive measures (tab-separated)

## Before running

Every script hardcodes `DATA_DIR` / `OUT_DIR` / `META_FILE` near the top as
absolute paths. Update these to match your own machine before running.

## Order to run

**1. Run the main pipeline first** — everything else depends on its output CSVs.

- `nimh_specparam.py` — fits specparam to every subject, prints a methods report
  and goodness-of-fit summary, and writes to `results/`:
  - `nimh_aperiodic.csv` — per-channel offset/exponent/R²/MAE
  - `nimh_peaks.csv` — per-channel band peaks (delta–gamma)
  - `nimh_subject_summary.csv` — per-subject means, merged with age/sex
  - `failed_fits.csv` — only if any channels failed to fit
  - per-subject and group summary figures in `results/figures/`

  `nimh_specparam_walkthrough.py` is an annotated, tutorial-style version of
  the same pipeline (mirrors the specparam tutorial steps 1:1). It's optional —
  useful for understanding *how* the fitting works, but produces the same
  outputs, so you only need to run one of the two.

**2. Then run any of these downstream analyses, in any order** (each only needs
   the CSVs from step 1):

- `addplots.py` — raw vs. aperiodic-corrected band power comparisons
- `robustness.py` — checks whether sex confounds the age–exponent relationship
- `cogmeasures.py` — correlates aperiodic exponent with KBIT-2 and NIH Toolbox
  cognitive scores (controlling for age + sex)

**Not part of this flow:** `run_pipeline.py` / `run_pipeline.ipynb` are an
earlier, standalone version of the fitting pipeline (different output file
names: `aperiodic_results.csv`, `peak_results.csv`). Kept for reference only —
nothing else in this repo reads their output.

## Dependencies

`specparam`, `h5py`, `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`,
`openpyxl` (for reading the `.xlsx` metadata file).
