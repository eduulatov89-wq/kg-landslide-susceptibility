import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
MC = dict(zip(['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN', 'Ensemble'], ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#222222']))

# ---- combined main figure: ROC (a) + random vs spatial CV (b)
a = Image.open('out/ml_01_roc_all_models.png').convert('RGB'); b = Image.open('out/ml_02b_cv_random_vs_spatial.png').convert('RGB')
h = 1250; b = b.resize((round(b.width * h / b.height), h), Image.LANCZOS); a = a.resize((round(a.width * h / a.height), h), Image.LANCZOS)
gap = 60; top = 70
c = Image.new('RGB', (a.width + b.width + gap, h + top), 'white'); c.paste(a, (0, top)); c.paste(b, (a.width + gap, top))
from PIL import ImageDraw, ImageFont
d = ImageDraw.Draw(c)
try: f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 52)
except Exception: f = ImageFont.load_default()
d.text((10, 5), '(a)', fill='black', font=f); d.text((a.width + gap + 10, 5), '(b)', fill='black', font=f)
c.save('out/fig_main_roc_cv.png', dpi=(220, 220)); print('combined', c.size)

# ---- success / prediction rate curves
Z = np.load('data/rate_curves.npz'); R = pd.read_csv('out/table_success_prediction_rates.csv').set_index('model')
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, k, t in [(axs[0], 'tr', '(a) Success rate: training landslide cells (n = 3,600)'), (axs[1], 'te', '(b) Prediction rate: test landslide cells (n = 1,543)')]:
    for m in ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN', 'Ensemble']:
        x = np.r_[0, Z[f'{m}_{k}a'], 1]; y = np.r_[0, Z[f'{m}_{k}y'], 1]
        auc = R.loc[m, 'success_rate_AUC' if k == 'tr' else 'prediction_rate_AUC']
        ax.plot(x * 100, y * 100, color=MC[m], lw=2.2 if m == 'Ensemble' else 1.5, ls='--' if m == 'Ensemble' else '-', label=f'{m}  AUC = {auc:.3f}')
    ax.plot([0, 100], [0, 100], ':', color='#999', lw=1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_xlabel('Cumulative % of land area (ranked from highest probability)')
    ax.set_ylabel('Cumulative % of landslide cells'); ax.set_title(t, loc='left', fontsize=10); ax.legend(fontsize=7.5, loc='lower right', frameon=False)
    ax.grid(color='#eee', lw=0.6)
    ia = ax.inset_axes([0.42, 0.38, 0.3, 0.3])
    for m in ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN']:
        ia.plot(np.r_[0, Z[f'{m}_{k}a']] * 100, np.r_[0, Z[f'{m}_{k}y']] * 100, color=MC[m], lw=1.2)
    ia.set_xlim(0, 15); ia.set_ylim(0, 100); ia.tick_params(labelsize=6.5); ia.set_title('first 15% of area', fontsize=7)
fig.tight_layout(); fig.savefig('out/chart_16_success_prediction_rates.png', dpi=220, bbox_inches='tight'); plt.close(fig); print('rates')

# ---- classification schemes
C = pd.read_csv('out/table_classification_schemes.csv')
O = ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN', 'Ensemble']; sch = [('equal', 'Equal interval', ''), ('natural', 'Natural breaks (Jenks)', '////'), ('quantile', 'Quantile', '....')]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2)); x = np.arange(len(O)); w = 0.27
for ax, col, lab in [(axs[0], 'VH_area_pct', 'Very high class: % of land area'), (axs[1], 'VH_LS_pct', 'Very high class: % of landslide cells')]:
    for j, (s, n, h) in enumerate(sch):
        v = C[C.scheme == s].set_index('model').loc[O, col]
        ax.bar(x + (j - 1) * w, v, w, color=[MC[m] for m in O], edgecolor='#333', lw=0.5, hatch=h, alpha=0.85)
        for xi, val in zip(x, v): ax.text(xi + (j - 1) * w, val + 1, f'{val:.0f}', ha='center', fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(O); ax.set_ylabel(lab); ax.set_ylim(0, 105 if 'LS' in col else 25)
from matplotlib.patches import Patch
axs[0].legend(handles=[Patch(fc='white', ec='#333', hatch=h, label=n) for _, n, h in sch], frameon=False, fontsize=8, loc='upper left')
axs[0].set_title('(a) Area of the Very high class', loc='left', fontsize=10); axs[1].set_title('(b) Landslide cells in the Very high class', loc='left', fontsize=10)
fig.tight_layout(); fig.savefig('out/chart_17_classification_schemes.png', dpi=220, bbox_inches='tight'); plt.close(fig); print('schemes')
