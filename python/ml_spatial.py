import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, scipy.stats as st, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
O = ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN']
MC = dict(zip(O, ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']))
R = pd.read_csv('data/ml_cv_auc.csv')[O]; S = pd.read_csv('data/ml_cv_auc_spatial.csv')[O]
fr_r = st.friedmanchisquare(*[R[c] for c in O]); fr_s = st.friedmanchisquare(*[S[c] for c in O])
rk_r = R.rank(axis=1, ascending=False).mean(); rk_s = S.rank(axis=1, ascending=False).mean()
wins = (S.values.argmax(1)[:, None] == np.arange(6)).sum(0)
T = pd.DataFrame({'random_mean': R.mean(), 'random_sd': R.std(ddof=0), 'spatial_mean': S.mean(), 'spatial_sd': S.std(ddof=0),
                  'drop': R.mean() - S.mean(), 'rank_random': rk_r, 'rank_spatial': rk_s, 'spatial_wins': wins})
T.to_csv('out/table_cv_random_vs_spatial.csv'); S.to_csv('out/table_ml_cv_auc_spatial_25folds.csv', index=False)
# Wilcoxon RF vs others on spatial folds
W = {m: st.wilcoxon(S['RF'], S[m]).pvalue for m in O if m != 'RF'}
W.update({'XGBoost_vs_GBT': st.wilcoxon(S['XGBoost'], S['GBT']).pvalue})
print(T.round(4)); print('Friedman random', fr_r, 'spatial', fr_s); print('Wilcoxon (spatial) RF vs', {k: f'{v:.2e}' for k, v in W.items()})
pd.Series(W).to_csv('out/table_spatial_wilcoxon.csv')
fig, ax = plt.subplots(figsize=(9, 4.6))
for i, m in enumerate(O):
    for j, (D, off, alpha) in enumerate([(R, -0.18, 0.35), (S, 0.18, 0.9)]):
        b = ax.boxplot(D[m], positions=[i + off], widths=0.3, patch_artist=True, showfliers=False)
        for p in b['boxes']: p.set(facecolor=MC[m], alpha=alpha, edgecolor='#333')
        for k in ('medians',): [l.set(color='#111') for l in b[k]]
        ax.scatter(np.full(len(D), i + off) + np.random.default_rng(i).uniform(-0.07, 0.07, len(D)), D[m], s=6, color='#333', alpha=0.5, zorder=3)
ax.set_xticks(range(6)); ax.set_xticklabels(O); ax.set_ylabel('AUC'); ax.grid(axis='y', color='#e6e6e6')
from matplotlib.patches import Patch
ax.legend([Patch(fc='#999', alpha=0.35, ec='#333'), Patch(fc='#999', alpha=0.9, ec='#333')], ['Random 5×5 CV', 'Spatial-block 5×5 CV (50 km blocks)'], loc='lower left', frameon=False)
ax.set_title('Cross-validated AUC: random vs spatial-block folds (25 folds each)', loc='left', fontweight='bold')
fig.text(0.01, -0.04, f'Friedman: random χ² = {fr_r.statistic:.1f}, p = {fr_r.pvalue:.1e}; spatial χ² = {fr_s.statistic:.1f}, p = {fr_s.pvalue:.1e}.', fontsize=8, color='#555')
fig.savefig('out/ml_02b_cv_random_vs_spatial.png', dpi=220, bbox_inches='tight'); print('saved')
