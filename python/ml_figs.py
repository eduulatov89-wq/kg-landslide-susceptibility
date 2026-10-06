import sys; sys.path.insert(0, '.')
from mapkit import *
from make_maps import CLASS_NAMES, CLASS_COLORS
import pandas as pd, scipy.stats as st
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False})
ORDER = ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN']
FULL = {'RF': 'Random Forest', 'GBT': 'Gradient Boosting', 'XGBoost': 'XGBoost', 'SVM': 'Support Vector Machine', 'DNN': 'Deep Neural Network (MLP)', 'KNN': 'k-Nearest Neighbours'}
MC = dict(zip(ORDER, ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']))
M = pd.read_csv('data/ml_metrics.csv').set_index('model').loc[ORDER]
P = pd.read_csv('data/ml_test_probs.csv'); CV = pd.read_csv('data/ml_cv_auc.csv')[ORDER]
IMP = pd.read_csv('data/ml_perm_importance.csv', index_col=0)[ORDER]
GR = np.load('data/ml_prob_grids.npz')
S = pd.read_csv('data/samples.csv')
J = np.clip(((S.x - X0) / (X1 - X0) * W).astype(int), 0, W - 1); I = np.clip(((Y1 - S.y) / (Y1 - Y0) * H).astype(int), 0, H - 1)
PI, PJ = S.row[S.cls == 1].values, S.col[S.cls == 1].values
def save(fig, name): fig.savefig(f'out/{name}.png', dpi=220, bbox_inches='tight', pad_inches=0.15); plt.close(fig); print(name)
y = P.cls.values

# DeLong
def delong(y, a, b):
    def comp(pos, neg):
        V = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
        return V.mean(1), V.mean(0), V.mean()
    pa, na, A = comp(a[y == 1], a[y == 0]); pb, nb, B = comp(b[y == 1], b[y == 0])
    m, n = len(pa), len(na)
    s10 = np.cov(np.vstack([pa, pb])); s01 = np.cov(np.vstack([na, nb]))
    S_ = s10 / m + s01 / n; var = S_[0, 0] + S_[1, 1] - 2 * S_[0, 1]
    z = (A - B) / np.sqrt(var) if var > 0 else 0
    return A - B, 2 * st.norm.sf(abs(z))
DL = pd.DataFrame(index=ORDER, columns=ORDER, dtype=float)
for a in ORDER:
    for b in ORDER: DL.loc[a, b] = np.nan if a == b else delong(y, P[a].values, P[b].values)[1]
DL.to_csv('out/table_ml_delong_pvalues.csv')
fr = st.friedmanchisquare(*[CV[c] for c in ORDER]); print('Friedman', fr)

# ML01 ROC
fig, ax = plt.subplots(figsize=(6.6, 6.2))
for m in ORDER:
    f, t, _ = roc_curve(y, P[m]); ax.plot(f, t, color=MC[m], lw=1.9, label=f'{FULL[m]}  AUC = {auc(f, t):.3f}')
ax.plot([0, 1], [0, 1], '--', color='#9e9e9e', lw=1, label='No skill  AUC = 0.500')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect('equal'); ax.set_xlabel('False positive rate (1 − specificity)'); ax.set_ylabel('True positive rate (sensitivity)')
ax.legend(loc='lower right', fontsize=8); ax.set_title(f'ROC curves of six models — test set (n = {len(y)})', loc='left')
axi = ax.inset_axes([0.42, 0.42, 0.33, 0.3])
for m in ORDER:
    f, t, _ = roc_curve(y, P[m]); axi.plot(f, t, color=MC[m], lw=1.4)
axi.set_xlim(0, 0.25); axi.set_ylim(0.75, 1.0); axi.tick_params(labelsize=6.5); axi.set_title('zoom', fontsize=7)
save(fig, 'ml_01_roc_all_models')

# ML02 CV AUC distributions
fig, ax = plt.subplots(figsize=(8.5, 4.6))
bp = ax.boxplot([CV[m] for m in ORDER], widths=0.55, patch_artist=True, showfliers=True, medianprops=dict(color='#222', lw=1.3), flierprops=dict(ms=3))
for p_, m in zip(bp['boxes'], ORDER): p_.set_facecolor(MC[m]); p_.set_alpha(0.6); p_.set_edgecolor(MC[m])
for k, m in enumerate(ORDER): ax.scatter(np.random.default_rng(1).normal(k + 1, 0.05, len(CV)), CV[m], s=6, color=MC[m], alpha=0.6, zorder=3)
ax.set_xticks(range(1, 7)); ax.set_xticklabels([FULL[m].replace(' (MLP)', '\n(MLP)').replace('Support Vector Machine', 'SVM').replace('k-Nearest Neighbours', 'KNN') for m in ORDER], fontsize=8.5)
for k, m in enumerate(ORDER): ax.text(k + 1, CV[m].max() + 0.004, f'{CV[m].mean():.3f}\n± {CV[m].std():.3f}', ha='center', fontsize=7.5)
ax.set_ylabel('AUC'); ax.grid(axis='x', visible=False); ax.set_ylim(CV.values.min() - 0.01, CV.values.max() + 0.025)
ax.set_title(f'Robustness: AUC over 5 × 5-fold cross-validation ({len(CV)} folds, all {len(S):,} cells)', loc='left')
fig.text(0.01, -0.03, f'Friedman test across models: χ² = {fr.statistic:.1f}, p = {fr.pvalue:.1e}.', fontsize=7.5, color='#555')
save(fig, 'ml_02_cv_auc_boxplots')

# ML03 metrics heatmap
cols = ['AUC', 'CV_AUC_mean', 'Accuracy', 'Kappa', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'MCC', 'Brier']
lab = ['AUC\n(test)', 'AUC\n(5×5 CV)', 'Accuracy', 'Kappa', 'Sensi-\ntivity', 'Speci-\nficity', 'Precision', 'F1', 'MCC', 'Brier\n(lower = better)']
V = M[cols].copy(); Rk = V.rank(ascending=False, method='min'); Rk['Brier'] = V['Brier'].rank(method='min')
V['mean rank'] = Rk.mean(1); lab2 = lab + ['Mean\nrank']
fig, ax = plt.subplots(figsize=(11.5, 4.2))
norm_cols = (V.iloc[:, :-1] - V.iloc[:, :-1].min()) / (V.iloc[:, :-1].max() - V.iloc[:, :-1].min() + 1e-12)
norm_cols['Brier'] = 1 - norm_cols['Brier']; nr = norm_cols.copy(); nr['mean rank'] = 1 - (V['mean rank'] - 1) / 5
ax.imshow(nr.values, cmap='Greens', vmin=-0.2, vmax=1.3, aspect='auto'); ax.grid(False)
for i in range(len(V)):
    for j, c in enumerate(V.columns):
        v = V.iloc[i, j]; ax.text(j, i, f'{v:.2f}' if c == 'mean rank' else f'{v:.3f}', ha='center', va='center', fontsize=8.5,
                                  fontweight='bold' if (c != 'mean rank' and Rk.iloc[i, j] == 1) or (c == 'mean rank' and v == V['mean rank'].min()) else 'normal')
ax.set_xticks(range(len(lab2))); ax.set_xticklabels(lab2, fontsize=8); ax.set_yticks(range(len(V))); ax.set_yticklabels([FULL[m] for m in ORDER], fontsize=9)
ax.xaxis.tick_top(); [s_.set_visible(False) for s_ in ax.spines.values()]
ax.set_title('Model comparison on the test set (bold = best; darker = better within each column)', loc='left', pad=34)
save(fig, 'ml_03_metrics_comparison')
V.round(4).assign(best_params=M.best_params, train_AUC=M.train_AUC.round(3)).to_csv('out/table_ml_metrics.csv')

# ML04 confusion matrices
fig, axs = plt.subplots(2, 3, figsize=(11, 7.2)); fig.subplots_adjust(hspace=0.45, wspace=0.35)
for ax, m in zip(axs.flat, ORDER):
    r = M.loc[m]; cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
    ax.imshow(cm, cmap=ListedColormap(['#ffffff', MC[m]]), vmin=0, vmax=cm.max() * 1.6, alpha=0.9); ax.grid(False)
    for i in range(2):
        for j in range(2): ax.text(j, i, f'{cm[i, j]}\n({cm[i, j] / cm[i].sum() * 100:.0f}%)', ha='center', va='center', fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Non-LS', 'Landslide']); ax.set_yticks([0, 1]); ax.set_yticklabels(['Non-LS', 'Landslide'], rotation=90, va='center')
    ax.set_xlabel('Predicted', fontsize=8); ax.set_ylabel('Observed', fontsize=8); ax.set_title(f'{FULL[m]}\nOA {r.Accuracy:.3f} · κ {r.Kappa:.3f}', loc='left', fontsize=9.5)
    [s_.set_visible(False) for s_ in ax.spines.values()]
fig.suptitle('Confusion matrices on the test set (threshold 0.5)', x=0.02, ha='left', fontweight='bold', fontsize=12)
save(fig, 'ml_04_confusion_matrices')

# ML05 permutation importance
SH = {'elevation': 'Elevation', 'slope': 'Slope', 'northness': 'Northness', 'eastness': 'Eastness', 'plan_curv': 'Plan curvature', 'profile_curv': 'Profile curvature',
      'twi': 'TWI', 'rain_annual': 'Annual rainfall', 'lulc': 'Land cover', 'lithology': 'Lithology', 'dist_roads': 'Distance to roads', 'dist_rivers': 'Distance to rivers', 'dist_faults': 'Distance to faults', 'ndvi': 'NDVI', 'clay': 'Clay', 'sand': 'Sand'}
imp = IMP.clip(lower=0); impn = imp / imp.sum() * 100
order = impn.mean(1).sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(8.8, 6.6))
ax.imshow(impn.loc[order].values, cmap='Blues', aspect='auto', vmin=0, vmax=impn.values.max()); ax.grid(False)
for i, f in enumerate(order):
    for j, m in enumerate(ORDER):
        v = impn.loc[f, m]; ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8, color='white' if v > impn.values.max() * 0.55 else '#222')
ax.set_xticks(range(6)); ax.set_xticklabels(ORDER); ax.set_yticks(range(len(order))); ax.set_yticklabels([SH[f] for f in order]); ax.xaxis.tick_top()
[s_.set_visible(False) for s_ in ax.spines.values()]
ax.set_title('Permutation importance (AUC drop, % of each model\'s total)', loc='left', pad=24)
fig.text(0.01, 0.0, 'Test set, 20 permutations per factor. Factors sorted by mean importance across models.', fontsize=7.5, color='#555')
save(fig, 'ml_05_permutation_importance')

# class statistics per model
cc = ListedColormap(CLASS_COLORS); cn = BoundaryNorm(np.arange(-0.5, 5.5), 5)
def classes(p): return np.where(np.isnan(p), np.nan, np.clip(np.floor(p * 5), 0, 4))
rows = []
for m in ORDER:
    c = classes(GR[m]); valid = ~np.isnan(c); lsc = c[PI, PJ]; lsc = lsc[~np.isnan(lsc)]
    for k in range(5):
        a = (c[valid] == k).mean() * 100; l = (lsc == k).mean() * 100
        rows.append(dict(model=m, cls=CLASS_NAMES[k], area_pct=a, landslide_pct=l, FR=l / a if a > 0 else np.nan))
CS = pd.DataFrame(rows); CS.to_csv('out/table_ml_class_stats.csv', index=False)

# ML06 panel of 6 class maps
fig, axs = plt.subplots(3, 2, figsize=(13, 10.2)); fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07, hspace=0.18, wspace=0.04)
for ax, m in zip(axs.flat, ORDER):
    ax.imshow(compose(classes(GR[m]), cc, cn, 0.22), extent=EXTENT, origin='upper', interpolation='nearest')
    draw_lakes(ax, 0.3)
    for p_ in getattr(KG, 'geoms', [KG]): ax.plot(*p_.exterior.xy, color='#111', lw=0.7)
    for n_, o_, g_ in ADM:
        for p_ in getattr(g_, 'geoms', [g_]): ax.plot(*p_.exterior.xy, color='#555', lw=0.3)
    ax.set_xlim(X0, X1); ax.set_ylim(Y0 + 15000, Y1 - 15000); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    vh = CS[(CS.model == m) & (CS.cls == 'Very high')].iloc[0]
    ax.set_title(f'{FULL[m]}  —  AUC {M.loc[m, "AUC"]:.3f}', loc='left', fontsize=10.5)
    ax.text(0.99, 0.03, f'Very high: {vh.area_pct:.1f}% of area, {vh.landslide_pct:.0f}% of landslides', transform=ax.transAxes, ha='right', fontsize=8,
            bbox=dict(fc='white', ec='none', alpha=0.85))
fig.legend(handles=[Patch(fc=c, ec='#666', lw=0.5, label=n) for n, c in zip(CLASS_NAMES, CLASS_COLORS)], loc='lower center', ncol=5, fontsize=10, title='Landslide susceptibility class', title_fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.suptitle('Landslide susceptibility of Kyrgyzstan predicted by six machine-learning models', x=0.01, ha='left', fontweight='bold', fontsize=13)
save(fig, 'ml_06_susceptibility_maps_6_models')

# ML07 individual full maps
for k, m in enumerate(ORDER):
    fig, ax = new_map()
    ax.imshow(compose(classes(GR[m]), cc, cn, 0.22), extent=EXTENT, origin='upper', interpolation='nearest')
    boundaries(ax, lw_obl=0.7, labels=True); graticule(ax); furniture(ax)
    ax.scatter(LS[:, 0], LS[:, 1], s=1.2, c='#111', lw=0, zorder=6, alpha=0.7)
    cs = CS[CS.model == m]
    legend_boxes(ax, [(f'{r.cls}  ({r.area_pct:.1f}% of area)', CLASS_COLORS[i]) for i, (_, r) in enumerate(cs.iterrows())], 'Susceptibility class',
                 extra=[Line2D([], [], marker='o', ls='', mfc='#111', mec='#111', ms=3.5, label='Mapped landslide')])
    ax.set_title(f'Landslide susceptibility — {FULL[m]} (test AUC {M.loc[m, "AUC"]:.3f})', loc='left')
    import textwrap
    credit(fig, '\n'.join(textwrap.wrap(f'Best hyper-parameters (5-fold CV grid search): {M.loc[m, "best_params"].replace(chr(34), "")}. Equal-interval probability classes (0.2 steps). {CRS_NOTE}.', 190)))
    save(fig, f'ml_07{chr(97 + k)}_susceptibility_{m}')

# ML08 class shares per model
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={'wspace': 0.25})
for ax, col, ttl in [(axs[0], 'area_pct', 'Share of country area (%)'), (axs[1], 'landslide_pct', 'Share of mapped landslides (%)')]:
    left = np.zeros(6)
    for k, c in enumerate(CLASS_NAMES):
        v = np.array([CS[(CS.model == m) & (CS.cls == c)][col].iloc[0] for m in ORDER])
        ax.barh(ORDER, v, left=left, color=CLASS_COLORS[k], edgecolor='#666', lw=0.4, height=0.65, label=c)
        for i, (l, vv) in enumerate(zip(left, v)):
            if vv > 6: ax.text(l + vv / 2, i, f'{vv:.0f}', ha='center', va='center', fontsize=7.5, color='white' if k in (0, 4) else '#222')
        left += v
    ax.set_xlim(0, 100); ax.invert_yaxis(); ax.set_title(ttl, loc='left'); ax.grid(False)
axs[0].legend(ncol=5, loc='lower left', bbox_to_anchor=(0, 1.08), fontsize=8.5)
fig.suptitle('Susceptibility class composition by model', x=0.01, ha='left', fontweight='bold', fontsize=12, y=1.08)
save(fig, 'ml_08_class_shares_by_model')

# ML09 frequency ratio per class per model
fig, ax = plt.subplots(figsize=(9, 4.6)); x = np.arange(5); w = 0.13
for k, m in enumerate(ORDER):
    v = [CS[(CS.model == m) & (CS.cls == c)].FR.iloc[0] for c in CLASS_NAMES]
    ax.bar(x + (k - 2.5) * w, v, w, color=MC[m], label=m)
ax.set_yscale('log'); ax.axhline(1, color='#555', lw=0.8, ls=':'); ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES); ax.set_ylabel('Frequency ratio (log scale)')
ax.legend(ncol=6, loc='upper left', fontsize=8); ax.grid(axis='x', visible=False)
ax.set_title('Frequency ratio of landslides in each susceptibility class', loc='left')
fig.text(0.01, -0.03, f'FR = landslide share / area share; FR > 1 means more landslides than expected by area. All {len(PI):,} landslide cells.', fontsize=7.5, color='#555')
save(fig, 'ml_09_frequency_ratio_by_class')

# ML10 ensemble & agreement maps
stack = np.stack([GR[m] for m in ORDER]); mean = np.nanmean(stack, 0); sd = np.nanstd(stack, 0)
agree = np.stack([classes(GR[m]) for m in ORDER]); vh = (agree >= 3).sum(0).astype(float); vh[np.isnan(mean)] = np.nan
for name, arr, cmap_, norm_, lab_, ttl in [
    ('ml_10a_ensemble_mean_classes', classes(mean), cc, cn, None, 'Ensemble susceptibility (mean probability of six models)'),
    ('ml_10b_model_disagreement', sd, plt.get_cmap('PuBu'), Normalize(0, 0.3), 'Standard deviation of probability across models', 'Model disagreement (uncertainty)'),
    ('ml_10c_high_class_agreement', vh, ListedColormap(['#f7f7f7', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d7191c', '#a50026']), BoundaryNorm(np.arange(-0.5, 7.5), 7),
     'Number of models classing the cell High or Very high', 'Consensus: models agreeing on High / Very high susceptibility')]:
    fig, ax = new_map()
    ax.imshow(compose(arr, cmap_, norm_, 0.22), extent=EXTENT, origin='upper', interpolation='nearest')
    boundaries(ax, lw_obl=0.6, labels=name.endswith('classes')); graticule(ax); furniture(ax)
    if lab_ is None:
        legend_boxes(ax, [(n, c) for n, c in zip(CLASS_NAMES, CLASS_COLORS)], 'Susceptibility class')
    else:
        cb = cbar(fig, cmap_, norm_, lab_, ticks=list(range(7)) if 'agree' in name else None, extend='max' if 'disagree' in name else 'neither')
    ax.set_title(ttl, loc='left'); credit(fig, f'Models: RF, GBT, XGBoost, SVM, DNN, KNN. {CRS_NOTE}.')
    save(fig, name)
np.savez_compressed('data/ml_ensemble.npz', mean=mean, sd=sd)

# ML11 DeLong matrix
fig, ax = plt.subplots(figsize=(6.6, 5.4))
L = -np.log10(DL.values.astype(float))
ax.imshow(np.where(np.isnan(L), 0, np.clip(L, 0, 3)), cmap='Purples', vmin=0, vmax=3.5); ax.grid(False)
for i, a in enumerate(ORDER):
    for j, b in enumerate(ORDER):
        if i != j:
            d, p = delong(y, P[a].values, P[b].values)
            ax.text(j, i, f'ΔAUC {d:+.3f}\np = {p:.3f}', ha='center', va='center', fontsize=7, color='white' if p < 0.01 else '#222')
ax.set_xticks(range(6)); ax.set_xticklabels(ORDER); ax.set_yticks(range(6)); ax.set_yticklabels(ORDER); ax.xaxis.tick_top()
[s_.set_visible(False) for s_ in ax.spines.values()]
ax.set_title('Pairwise DeLong tests of test-set AUC (row − column)', loc='left', pad=22)
fig.text(0.01, 0.0, 'Darker = smaller p. p < 0.05 means the AUC difference is statistically significant.', fontsize=7.5, color='#555')
save(fig, 'ml_11_delong_tests')
print(M[['AUC', 'CV_AUC_mean', 'Accuracy', 'Kappa']].round(3)); print(DL.round(3))
