"""Fig. 6 without the overall title and in-panel notes; panels lettered (a)-(f) to match the caption."""
import sys; sys.path.insert(0, '.')
from mapkit import *
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
CLASS_NAMES = ['Very low', 'Low', 'Moderate', 'High', 'Very high']
CLASS_COLORS = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
ORDER = ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN']
FULL = {'RF': 'Random Forest', 'GBT': 'Gradient Boosting', 'XGBoost': 'XGBoost', 'SVM': 'Support Vector Machine', 'DNN': 'Deep Neural Network (MLP)', 'KNN': 'k-Nearest Neighbours'}
M = pd.read_csv('data/ml_metrics.csv').set_index('model'); GR = np.load('data/ml_prob_grids.npz')
cc = ListedColormap(CLASS_COLORS); cn = BoundaryNorm(np.arange(-0.5, 5.5), 5)
def classes(p): return np.where(np.isnan(p), np.nan, np.clip(np.floor(p * 5), 0, 4))
fig, axs = plt.subplots(3, 2, figsize=(13, 9.8)); fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.07, hspace=0.16, wspace=0.04)
for k, (ax, m) in enumerate(zip(axs.flat, ORDER)):
    ax.imshow(compose(classes(GR[m]), cc, cn, 0.22), extent=EXTENT, origin='upper', interpolation='nearest')
    draw_lakes(ax, 0.3)
    for p_ in getattr(KG, 'geoms', [KG]): ax.plot(*p_.exterior.xy, color='#111', lw=0.7)
    for n_, o_, g_ in ADM:
        for p_ in getattr(g_, 'geoms', [g_]): ax.plot(*p_.exterior.xy, color='#555', lw=0.3)
    ax.set_xlim(X0, X1); ax.set_ylim(Y0 + 15000, Y1 - 15000); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'({"abcdef"[k]}) {FULL[m]}  —  AUC {M.loc[m, "AUC"]:.3f}', loc='left', fontsize=10.5)
fig.legend(handles=[Patch(fc=c, ec='#666', lw=0.5, label=n) for n, c in zip(CLASS_NAMES, CLASS_COLORS)], loc='lower center', ncol=5, fontsize=10,
           title='Landslide susceptibility class', title_fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.savefig('out/nocap/ml_06_susceptibility_maps_6_models.png', dpi=300, bbox_inches='tight', pad_inches=0.12); print('ok')
