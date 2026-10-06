import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, joblib, shap, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from ml import PRED, NUM, CAT
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
S = pd.read_csv('data/samples.csv'); te = S[S.rand >= 0.7].reset_index(drop=True)
best = joblib.load('data/ml_models.joblib')
X = te[PRED]; cols = NUM + CAT       # passthrough order used by the tree pipelines
Xp = best['RF'][:-1].transform(X); Xp = pd.DataFrame(Xp, columns=cols)
LAB = {'elevation': 'Elevation (m)', 'slope': 'Slope (°)', 'northness': 'Northness', 'eastness': 'Eastness', 'plan_curv': 'Plan curvature',
       'profile_curv': 'Profile curvature', 'twi': 'TWI', 'rain_annual': 'Annual rainfall (mm)', 'lulc': 'Land cover (code)', 'lithology': 'Lithology (group code)',
       'dist_roads': 'Distance to roads (km)', 'dist_rivers': 'Distance to rivers (km)', 'dist_faults': 'Distance to active faults (km)', 'ndvi': 'NDVI', 'clay': 'Clay (%)', 'sand': 'Sand (%)'}
out = {}
for name in ['RF', 'XGBoost']:
    mdl = best[name][-1]
    sub = Xp.sample(800, random_state=42) if name == 'RF' else Xp
    ex = shap.TreeExplainer(mdl)
    sv = ex.shap_values(sub)
    sv = sv[..., 1] if sv.ndim == 3 else (sv[1] if isinstance(sv, list) else sv)
    out[name] = (sub, sv)
    print(name, 'shap', sv.shape, flush=True)
# mean |SHAP|
imp = pd.DataFrame({n: np.abs(sv).mean(0) for n, (sub, sv) in out.items()}, index=cols)
imp_pct = imp / imp.sum() * 100; imp_pct.round(2).to_csv('out/table_shap_mean_abs.csv'); print(imp_pct.sort_values('RF', ascending=False).round(1))
# 1 beeswarm RF + XGB
for name, (sub, sv) in out.items():
    plt.figure()
    shap.summary_plot(sv, sub.rename(columns=LAB), show=False, max_display=14, plot_size=(7.5, 6.2), cmap='RdYlBu_r')
    ax = plt.gca(); ax.set_title(f'SHAP summary — {"Random Forest" if name=="RF" else "XGBoost"} (test set, n = {len(sub):,})', loc='left', fontsize=11, fontweight='bold')
    ax.set_xlabel('SHAP value (impact on landslide probability; RF) ' if name == 'RF' else 'SHAP value (impact on log-odds of landslide; XGBoost)')
    plt.savefig(f'out/shap_01{"a" if name=="RF" else "b"}_beeswarm_{name}.png', dpi=220, bbox_inches='tight'); plt.close()
# 2 importance comparison bar
o = imp_pct.sort_values('RF').index
fig, ax = plt.subplots(figsize=(7.5, 5.2)); y = np.arange(len(o))
ax.barh(y + 0.2, imp_pct.loc[o, 'RF'], 0.38, color='#0072B2', label='Random Forest')
ax.barh(y - 0.2, imp_pct.loc[o, 'XGBoost'], 0.38, color='#009E73', label='XGBoost')
ax.set_yticks(y); ax.set_yticklabels([LAB[c] for c in o]); ax.set_xlabel('Mean |SHAP| (% of model total)'); ax.legend(frameon=False, loc='lower right')
ax.grid(axis='x', color='#e6e6e6'); ax.set_axisbelow(True)
ax.set_title('Global factor contribution by SHAP', loc='left', fontweight='bold')
fig.savefig('out/shap_02_mean_abs_RF_vs_XGBoost.png', dpi=220, bbox_inches='tight'); plt.close(fig)
# 3 dependence panel (RF)
sub, sv = out['RF']; keys = ['elevation', 'slope', 'clay', 'ndvi', 'dist_roads', 'rain_annual']
fig, axs = plt.subplots(2, 3, figsize=(12, 7)); fig.subplots_adjust(hspace=0.38, wspace=0.28)
for ax, k in zip(axs.flat, keys):
    j = cols.index(k); x = sub[k].values; s = sv[:, j]
    ax.scatter(x, s, s=6, c=sub['slope'].values if k != 'slope' else sub['elevation'].values, cmap='RdYlBu_r', alpha=0.7, lw=0)
    order = np.argsort(x); xs, ss = x[order], s[order]
    w = max(15, len(x) // 25); sm = np.convolve(ss, np.ones(w) / w, mode='valid'); ax.plot(xs[w // 2: w // 2 + len(sm)], sm, color='#222', lw=1.6)
    ax.axhline(0, color='#888', lw=0.7, ls=':'); ax.set_xlabel(LAB[k]); ax.set_ylabel('SHAP value'); ax.grid(color='#eee')
    ax.set_title(LAB[k].split(' (')[0], loc='left', fontsize=10, fontweight='bold')
fig.suptitle('SHAP dependence of landslide probability on key factors (Random Forest; points coloured by slope, or elevation for the slope panel; line = running mean)',
             x=0.01, ha='left', fontsize=10.5, fontweight='bold')
fig.savefig('out/shap_03_dependence_RF.png', dpi=220, bbox_inches='tight'); plt.close(fig)
# thresholds where running-mean SHAP crosses zero
res = {}
for k in keys:
    j = cols.index(k); x = sub[k].values; s = sv[:, j]; order = np.argsort(x); xs, ss = x[order], s[order]
    w = max(15, len(x) // 25); sm = np.convolve(ss, np.ones(w) / w, mode='valid'); xm = xs[w // 2: w // 2 + len(sm)]
    pos = xm[sm > 0]; res[k] = (round(float(pos.min()), 2) if len(pos) else None, round(float(pos.max()), 2) if len(pos) else None, round(float(xm[np.argmax(sm)]), 2))
print('positive-SHAP range and peak (running mean):', res)
