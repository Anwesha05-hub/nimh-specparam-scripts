"""
Sex as a confound in the age–exponent relationship
==================================================================
1. Multiple regression: exponent ~ age + sex
2. If sex is significant, report age correlation separately by sex

"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, t as t_dist

OUT_DIR = Path('/Users/a.chakraborty.2@bham.ac.uk/Documents/specparam/results')
FIG_DIR = OUT_DIR / 'figures'

summary = pd.read_csv(OUT_DIR / 'nimh_subject_summary.csv').dropna(subset=['age', 'sex'])
n_subs  = len(summary)

ages      = summary['age'].values
exponents = summary['mean_exponent'].values
sex_bin   = (summary['sex'] == 'female').astype(float).values  # female=1, male=0

# =============================================================================
# Exponent ~ intercept + age + sex
# =============================================================================
def ols(X, y):
    """Return beta, se, t, p for each predictor."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None) #finding beta coefficients minimising sum of squared residuals.
    y_hat     = X @ beta   #fitting the model to get predicted values of y
    residuals = y - y_hat  #calculating residuals by subtracting predicted values from actual values
    n, k      = X.shape    #n = number of subjects (rows); k = number of predictors (columns)
    mse       = residuals @ residuals / (n - k) #mean squared error 
    cov       = mse * np.linalg.inv(X.T @ X)    #covariance matrix of the beta coefficients
    se        = np.sqrt(np.diag(cov))           #standard errors of the beta coefficients (sqroot of diagonal of cov matrix)
    t_stat    = beta / se                       #tstat: each coefficient/standard error
    p         = 2 * t_dist.sf(np.abs(t_stat), df=n - k) 
    return beta, se, t_stat, p

X_full = np.column_stack([np.ones(n_subs), ages, sex_bin])   #Build full matrix
beta, se, t_stat, p_ols = ols(X_full, exponents)             #Run full regression: exponent ~ intercept + age + sex

print(f"{'='*65}")
print(f"Multiple regression: exponent ~ age + sex  (n={n_subs})")
print(f"{'='*65}")
print(f"{'Predictor':<20} {'β':>8} {'SE':>8} {'t':>8} {'p':>9} {'sig':>5}")
print(f"{'-'*65}")
labels = ['Intercept', 'Age', 'Sex (female=1)']
for lab, b, s, tv, pv in zip(labels, beta, se, t_stat, p_ols):
    sig = '**' if pv < 0.01 else '*' if pv < 0.05 else 'ns'
    print(f"{lab:<20} {b:>8.4f} {s:>8.4f} {tv:>8.3f} {pv:>9.4f} {sig:>5}")

r_all, p_all = pearsonr(ages, exponents)

# =============================================================================
# Sex-stratified correlations
# =============================================================================
print(f"\n{'='*65}")
print("Sex-stratified age–exponent correlations")
print(f"{'='*65}")

for sex_label in ['female', 'male']:        #itirating twice for each sex
    mask = summary['sex'] == sex_label      #boolean mask 
    n    = mask.sum()                       #counts subjects in each group
    if n < 3:
        print(f"{sex_label.capitalize():<8} (n={n}) — too few for correlation")
        continue
    r, p = pearsonr(ages[mask], exponents[mask])    #Filters ages and exponents using mask, then computes Pearson r between age and exponent
    sig  = '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    pstr = f'{p:.4f}' if p >= 0.001 else '<0.001'
    print(f"{sex_label.capitalize():<8} (n={n}):  r={r:.3f},  r²={r**2:.3f},  p={pstr}  {sig}")

# =============================================================================
# Summary
# =============================================================================
sex_sig = p_ols[2] < 0.05 #pull p-value and cheeck if significant predictor
print(f"\n{'='*65}")
if sex_sig:
    print("Sex IS a significant predictor")
else:
    print("Sex is NOT a significant predictor (p={:.3f}).".format(p_ols[2]))
    print("The age–exponent effect holds independently of sex.")
print(f"{'='*65}")

# =============================================================================
# FIGURE: age–exponent scatter, coloured by sex + subplots
# =============================================================================
SEX_COLORS = {'female': '#E06C75', 'male': '#61AFEF'}

age_min = ages.min()
age_max = ages.max()
xline_full = np.linspace(age_min, age_max, 100)

def regression_line(x, y, xline):
    m, b = np.polyfit(x, y, 1)
    return xline, m * xline + b

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── Panel 1: all subjects coloured by sex, single regression line ─────────────
ax = axes[0]
for sex_label, color in SEX_COLORS.items():
    mask = summary['sex'] == sex_label
    ax.scatter(ages[mask], exponents[mask], color=color, s=60, alpha=0.85,
               edgecolors='white', linewidths=0.5, label=sex_label)
xline, yline = regression_line(ages, exponents, xline_full)        
ax.plot(xline, yline, 'k-', linewidth=1.8)
pstr = f'p={p_all:.3f}' if p_all >= 0.001 else 'p<0.001'
ax.set_title(f'All participants (n={n_subs})\nr={r_all:.3f}, {pstr}', fontsize=11)
ax.set_xlabel('Age (years)')
ax.set_ylabel('Aperiodic exponent (χ)')
ax.set_xlim(age_min - 1, age_max + 1)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Panels 2 & 3: sex-stratified ──────────────────────────────────────────────
for ax, sex_label in zip(axes[1:], ['female', 'male']):
    mask  = (summary['sex'] == sex_label).values
    x, y  = ages[mask], exponents[mask]
    color = SEX_COLORS[sex_label]
    r, p  = pearsonr(x, y)
    pstr  = f'p={p:.3f}' if p >= 0.001 else 'p<0.001'

    ax.scatter(x, y, color=color, s=60, alpha=0.85,
               edgecolors='white', linewidths=0.5)
    xline, yline = regression_line(x, y, xline_full)
    ax.plot(xline, yline, '-', color=color, linewidth=1.8)
    ax.set_title(f'{sex_label.capitalize()} (n={mask.sum()})\nr={r:.3f}, {pstr}', fontsize=11)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Aperiodic exponent (χ)')
    ax.set_xlim(age_min - 1, age_max + 1)
    ax.grid(alpha=0.3)

y_min = min(ax.get_ylim()[0] for ax in axes)
y_max = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(y_min, y_max)

plt.suptitle(f'Age–exponent relationship by sex  (n={n_subs})',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'robustness_sex.png', dpi=150)
plt.close()
print(f"\nSaved: robustness_sex.png")
