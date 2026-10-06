import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from ml import PRED
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
LAB = {'elevation': 'Elevation', 'slope': 'Slope', 'northness': 'Northness', 'eastness': 'Eastness', 'plan_curv': 'Plan curvature',
       'profile_curv': 'Profile curvature', 'twi': 'TWI', 'rain_annual': 'Annual rainfall', 'lulc': 'Land cover', 'lithology': 'Lithology',
       'dist_roads': 'Dist. to roads', 'dist_rivers': 'Dist. to rivers', 'dist_faults': 'Dist. to active faults', 'ndvi': 'NDVI', 'clay': 'Clay', 'sand': 'Sand'}
FULL = {'RF': 'Random Forest', 'GBT': 'Gradient Boosting', 'XGBoost': 'XGBoost', 'SVM': 'Support Vector Machine', 'DNN': 'Deep Neural Network', 'KNN': 'k-Nearest Neighbours'}
Z = np.load('data/shap_all6_400.npz'); X = Z['X']; O = ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN']
AUC = pd.read_csv('data/ml_metrics.csv').set_index('model').AUC
TOP = 12; rng = np.random.default_rng(0)
# feature colour = within-feature percentile rank (categorical factors in grey)
pr = np.column_stack([pd.Series(X[:, j]).rank(pct=True).values for j in range(X.shape[1])])
cmap = plt.get_cmap('RdYlBu_r')
fig, axs = plt.subplots(2, 3, figsize=(15, 10.5)); fig.subplots_adjust(wspace=0.55, hspace=0.22, left=0.08, right=0.92, top=0.93, bottom=0.07)
xl = max(np.percentile(np.abs(Z[m]), 99.8) for m in O)
for k, (ax, m) in enumerate(zip(axs.flat, O)):
    sv = Z[m]; imp = np.abs(sv).mean(0); order = np.argsort(imp)[::-1][:TOP][::-1]
    for i, j in enumerate(order):
        s = sv[:, j]; cat = PRED[j] in ('lulc', 'lithology')
        # beeswarm-like vertical jitter proportional to local density
        bins = np.round(s / (2 * xl / 120)); _, inv, cnt = np.unique(bins, return_inverse=True, return_counts=True)
        jit = (rng.random(len(s)) - 0.5) * np.minimum(cnt[inv] / cnt.max(), 1) * 0.8
        ax.scatter(s, i + jit, s=3.2, c='#9a9a9a' if cat else cmap(pr[:, j]), lw=0, alpha=0.85, rasterized=True)
        ax.text(xl * 1.02, i, f'{imp[j] / imp.sum() * 100:.1f}%', va='center', fontsize=7, color='#555')
    ax.set_yticks(range(len(order))); ax.set_yticklabels([LAB[PRED[j]] for j in order])
    ax.axvline(0, color='#777', lw=0.7); ax.set_xlim(-xl, xl); ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel('SHAP value (change in landslide probability)')
    ax.set_title(f'({"abcdef"[k]}) {FULL[m]} — test AUC {AUC[m]:.3f}', loc='left', fontsize=10, fontweight='bold')
    ax.grid(axis='x', color='#eee')
cax = fig.add_axes([0.945, 0.3, 0.012, 0.4]); cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax, ticks=[0, 1])
cb.ax.set_yticklabels(['Low', 'High']); cb.set_label('Factor value (percentile)', fontsize=8.5)
fig.text(0.08, 0.015, f'Same {len(X)} balanced test cells for every model; SHAP on the probability scale (tree models: interventional TreeSHAP; SVM, DNN, KNN: permutation SHAP on the full pipeline), '
         'background = 100 training cells. Twelve leading factors per model; right labels give each factor\'s share of mean |SHAP|. Grey: categorical factors (land cover, lithology).',
         fontsize=7.5, color='#555')
fig.savefig('out/shap_04_beeswarm_6models.png', dpi=300, bbox_inches='tight'); plt.close(fig); print('ok')
