"""Fig. 3a ROC curves: Okabe-Ito colours plus distinct line styles so models stay separable without colour."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6})
ORDER = ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN']
FULL = {'RF': 'Random Forest', 'GBT': 'Gradient Boosting', 'XGBoost': 'XGBoost', 'SVM': 'Support Vector Machine', 'DNN': 'Deep Neural Network (MLP)', 'KNN': 'k-Nearest Neighbours'}
MC = dict(zip(ORDER, ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']))
LS = dict(zip(ORDER, ['-', (0, (6, 2)), (0, (1, 1.2)), (0, (6, 2, 1, 2)), (0, (3, 1.5)), (0, (6, 2, 1, 2, 1, 2))]))
P = pd.read_csv('data/ml_test_probs.csv'); y = P.cls.values
fig, ax = plt.subplots(figsize=(6.6, 6.2))
for m in ORDER:
    f, t, _ = roc_curve(y, P[m]); ax.plot(f, t, color=MC[m], ls=LS[m], lw=1.9, label=f'{FULL[m]}  AUC = {auc(f, t):.3f}')
ax.plot([0, 1], [0, 1], color='#9e9e9e', ls=(0, (2, 3)), lw=1, label='No skill  AUC = 0.500')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect('equal'); ax.set_xlabel('False positive rate (1 − specificity)'); ax.set_ylabel('True positive rate (sensitivity)')
ax.legend(loc='lower right', fontsize=8, frameon=False, handlelength=3.2)
axi = ax.inset_axes([0.42, 0.42, 0.33, 0.3])
for m in ORDER:
    f, t, _ = roc_curve(y, P[m]); axi.plot(f, t, color=MC[m], ls=LS[m], lw=1.4)
axi.set_xlim(0, 0.25); axi.set_ylim(0.75, 1.0); axi.tick_params(labelsize=6.5); axi.set_title('zoom', fontsize=7)
fig.savefig('out/cvd/ml_01_roc_all_models.png', dpi=220, bbox_inches='tight', pad_inches=0.15); print('roc ok')
