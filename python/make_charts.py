import sys; sys.path.insert(0, '.')
from make_maps import F, layer, snap, WC_CODES, WC_NAMES, WC_COLORS, CLASS_NAMES, CLASS_COLORS
from common import *
import pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#555', 'axes.grid': True, 'grid.color': '#e6e6e6',
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'xtick.color': '#444', 'ytick.color': '#444', 'legend.frameon': False})
RED, BLUE, ORANGE, GREY, INK = '#c0392b', '#2c7fb8', '#d94801', '#9e9e9e', '#222'
def save(fig, name): fig.savefig(f'out/{name}.png', dpi=220, bbox_inches='tight', pad_inches=0.15); plt.close(fig); print(name)

S = pd.read_csv('data/samples.csv')
J = np.clip(((S.x - X0) / (X1 - X0) * W).astype(int), 0, W - 1); I = np.clip(((Y1 - S.y) / (Y1 - Y0) * H).astype(int), 0, H - 1)
LABEL = {f[0]: f[6].split('\n')[0] for f in F}
SHORT = {'elevation': 'Elevation', 'slope': 'Slope', 'aspect': 'Aspect', 'plan_curv': 'Plan curv.', 'profile_curv': 'Profile curv.',
         'relief': 'Relief', 'twi': 'TWI', 'rain_annual': 'Rain annual', 'rain_wetq': 'Rain wet qtr', 'dist_roads': 'Dist. roads',
         'dist_rivers': 'Dist. rivers', 'dist_faults': 'Dist. faults', 'ndvi': 'NDVI', 'clay': 'Clay', 'sand': 'Sand', 'lulc': 'Land cover', 'lithology': 'Lithology', 'northness': 'Northness', 'eastness': 'Eastness'}
L = {}
for f in F:
    L[f[0]] = layer(f[0]); S[f[0]] = L[f[0]][I, J]
lc = snap('lulc', WC_CODES + [95]); S['lulc'] = lc[I, J]; S['lithology'] = decode('lithology')[I, J]
pres, absn = S[S.cls == 1], S[S.cls == 0]

# C01 landslides per oblast
names = {name: oid for name, oid, g in ADM}
from shapely.geometry import Point
cnt = {n: 0 for n in names}
for x, y in LS:
    for n, oid, g in ADM:
        if g.contains(Point(x, y)): cnt[n] += 1; break
area = {n: g.area / 1e6 for n, oid, g in ADM}
d = pd.DataFrame({'n': cnt, 'area': area}); d['dens'] = d.n / d.area * 1000
d = d.sort_values('n')
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={'wspace': 0.55})
for ax, col, title, xl in [(axs[0], 'n', 'Mapped landslides per unit', 'number of landslides'), (axs[1], 'dens', 'Landslide density', 'landslides per 1000 km²')]:
    dd = d.sort_values(col)
    ax.barh(dd.index, dd[col], color=RED, height=0.62)
    for k, v in enumerate(dd[col]): ax.text(v, k, f' {v:.0f}' if col == 'n' else f' {v:.2f}', va='center', fontsize=8, color=INK)
    ax.set_title(title, loc='left'); ax.set_xlabel(xl); ax.grid(axis='y', visible=False); ax.margins(x=0.15)
fig.suptitle(f'Landslide inventory by administrative unit (compiled inventory, n = {len(LS):,})', x=0.02, ha='left', fontweight='bold', fontsize=11, y=1.02)
save(fig, 'chart_01_inventory_by_oblast')

# C02 factor distributions inside KG
fig, axs = plt.subplots(4, 4, figsize=(13, 10.5)); fig.subplots_adjust(hspace=0.55, wspace=0.3)
keys = [f for f in F]
for ax, f in zip(axs.flat, keys):
    v = L[f[0]][MASK]; v = v[~np.isnan(v)]
    lo, hi = np.percentile(v, [0.5, 99.5])
    bins = np.linspace(lo, hi, 40)
    ax.hist(v, bins=bins, color='#9ecae1', weights=np.full(v.size, 100 / v.size), label='All cells')
    pv = pres[f[0]].dropna()
    ax.hist(pv, bins=bins, histtype='step', color=RED, lw=1.4, weights=np.full(pv.size, 100 / max(pv.size, 1)), label='Landslide cells')
    ax.set_title(f[7].split(' — ')[0].split(',')[0].replace(' (max − min within 250 m)', '').replace(' 1991–2020', ''), loc='left', fontsize=9)
    ax.set_xlabel(f[6].split('\n')[0], fontsize=8); ax.set_ylabel('% of cells', fontsize=8); ax.tick_params(labelsize=7)
ax = axs.flat[len(keys)]
lcv = lc[MASK]; lcv = lcv[~np.isnan(lcv)]
share = [(WC_NAMES[k], (lcv == c).mean() * 100, (pres.lulc == c).mean() * 100, WC_COLORS[k]) for k, c in enumerate(WC_CODES)]
share = [s for s in share if s[1] > 0.1]
yy = np.arange(len(share))
ax.barh(yy + 0.2, [s[1] for s in share], height=0.38, color='#9ecae1'); ax.barh(yy - 0.2, [s[2] for s in share], height=0.38, color=RED)
ax.set_yticks(yy); ax.set_yticklabels([s[0] for s in share], fontsize=7); ax.invert_yaxis(); ax.set_title('Land cover', loc='left', fontsize=9); ax.set_xlabel('% of cells', fontsize=8)
for a in axs.flat[len(keys) + 1:]: a.axis('off')
h1 = plt.Rectangle((0, 0), 1, 1, fc='#9ecae1'); h2 = plt.Line2D([], [], color=RED, lw=1.6)
fig.legend([h1, h2], ['All 250 m cells in Kyrgyzstan', f'Landslide cells (n = {len(pres)})'], loc='upper right', ncol=2, bbox_to_anchor=(0.98, 0.955), fontsize=9)
fig.suptitle('Distribution of conditioning factors: country vs landslide cells', x=0.02, ha='left', fontweight='bold', fontsize=12, y=0.96)
save(fig, 'chart_02_factor_distributions')

# C03 presence vs absence boxplots
fig, axs = plt.subplots(3, 5, figsize=(13, 8)); fig.subplots_adjust(hspace=0.45, wspace=0.45)
for ax, f in zip(axs.flat, keys):
    a, b = absn[f[0]].dropna(), pres[f[0]].dropna()
    bp = ax.boxplot([a, b], widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.2))
    for p_, c in zip(bp['boxes'], [BLUE, RED]): p_.set_facecolor(c); p_.set_alpha(0.55); p_.set_edgecolor(c)
    for w_ in bp['whiskers'] + bp['caps']: w_.set_color('#666')
    ax.set_xticks([1, 2]); ax.set_xticklabels(['Non-\nlandslide', 'Landslide'], fontsize=7.5)
    ax.set_title(SHORT[f[0]], loc='left', fontsize=9.5); ax.set_ylabel(f[6].split('\n')[0], fontsize=7); ax.tick_params(axis='y', labelsize=7); ax.grid(axis='x', visible=False)
from scipy.stats import mannwhitneyu
for ax, f in zip(axs.flat, keys):
    a, b = absn[f[0]].dropna(), pres[f[0]].dropna(); p = mannwhitneyu(a, b).pvalue
    ax.text(0.98, 0.97, 'p < 0.001' if p < 0.001 else f'p = {p:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=7, color='#555')
axs.flat[-1].axis('off')
fig.suptitle('Factor values at landslide vs non-landslide cells (boxes: IQR, whiskers: 1.5×IQR; Mann–Whitney U test)', x=0.02, ha='left', fontweight='bold', fontsize=11.5, y=0.97)
save(fig, 'chart_03_presence_absence_boxplots')

# C04 correlation heatmap
C = ['elevation', 'slope', 'aspect', 'plan_curv', 'profile_curv', 'relief', 'twi', 'rain_annual', 'rain_wetq', 'lulc', 'dist_roads', 'dist_rivers', 'dist_faults', 'ndvi', 'clay', 'sand']
M = S[C].copy(); M['aspect'] = np.cos(np.radians(M['aspect'])); M = M.rename(columns={'aspect': 'northness'})
R = M.corr(method='pearson').values; labels = [SHORT[c] for c in M.columns]
fig, ax = plt.subplots(figsize=(8.6, 7.4))
im = ax.imshow(R, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=8); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
ax.grid(False)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f'{R[i, j]:.2f}'.replace('0.', '.').replace('-.', '−.') if i != j else '', ha='center', va='center', fontsize=6.3,
                color='white' if abs(R[i, j]) > 0.6 else '#333', fontweight='bold' if abs(R[i, j]) >= 0.7 and i != j else 'normal')
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('Pearson r'); cb.outline.set_linewidth(0.4)
ax.set_title(f'Correlation between conditioning factors ({len(S):,} model cells)', loc='left')
fig.text(0.01, -0.01, f'Bold: |r| ≥ 0.7. Relief (r = {R[1,5]:.2f} with slope) and wettest-quarter rain (r = {R[7,8]:.2f} with annual rain) were dropped before modelling.', fontsize=7.5, color='#555')
save(fig, 'chart_04_correlation_matrix')

# C05 VIF
v = pd.read_csv('data/kg_ls_vif.csv'); v['set'] = np.where(v['system:index'].str.startswith('1_'), 'all', 'model')
a = v[v.set == 'all'].set_index('factor').VIF; m = v[v.set == 'model'].set_index('factor').VIF
order = a.sort_values().index
fig, ax = plt.subplots(figsize=(8.5, 5.2)); y = np.arange(len(order))
ax.barh(y + 0.2, a[order], height=0.38, color=GREY, label='All 18 candidate factors')
ax.barh(y - 0.2, [m.get(k, np.nan) for k in order], height=0.38, color=BLUE, label='16 factors kept in the model')
ax.axvline(10, color=RED, lw=1, ls='--'); ax.text(10.2, len(order) - 0.6, 'VIF = 10', color=RED, fontsize=8)
ax.axvline(5, color=ORANGE, lw=0.8, ls=':'); ax.text(5.2, len(order) - 0.6, '5', color=ORANGE, fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([SHORT.get(k, k) for k in order]); ax.set_xlabel('Variance inflation factor'); ax.grid(axis='y', visible=False)
for k, o in enumerate(order):
    ax.text(a[o], k + 0.2, f' {a[o]:.1f}', va='center', fontsize=7, color='#555')
    if o in m: ax.text(m[o], k - 0.2, f' {m[o]:.1f}', va='center', fontsize=7, color=BLUE)
ax.legend(loc='lower right'); ax.set_title('Multicollinearity check (variance inflation factor)', loc='left')
save(fig, 'chart_05_vif')

# C06 frequency ratio panel
fr = pd.read_csv('data/kg_ls_frequency_ratio.csv')
titles = {'slope': 'Slope (°)', 'elevation': 'Elevation (m)', 'relief': 'Local relief (m)', 'rain_annual': 'Annual precipitation (mm)', 'dist_roads': 'Distance to roads (km)',
          'dist_rivers': 'Distance to rivers (km)', 'dist_faults': 'Distance to active faults (km)', 'ndvi': 'NDVI', 'twi': 'TWI', 'lulc': 'Land cover', 'lithology': 'Lithology (formation group)'}
fig, axs = plt.subplots(3, 4, figsize=(17, 11)); fig.subplots_adjust(hspace=0.95, wspace=0.3)
for ax, (k, t) in zip(axs.flat, titles.items()):
    d = fr[fr.factor == k].reset_index(drop=True); x = np.arange(len(d))
    ax.bar(x - 0.2, d.area_pct, 0.38, color='#c6dbef', label='% of country area')
    ax.bar(x + 0.2, d.landslide_pct, 0.38, color='#fcbba1', label='% of landslides')
    ax2 = ax.twinx(); ax2.plot(x, d.FR, 'o-', color=RED, lw=1.6, ms=4, label='Frequency ratio')
    ax2.axhline(1, color=RED, lw=0.7, ls=':'); ax2.set_ylim(0, max(2, d.FR.max() * 1.15)); ax2.spines['right'].set_visible(True); ax2.grid(False)
    ax2.tick_params(labelsize=7, colors=RED); ax2.set_ylabel('FR', color=RED, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(d.bin, rotation=40 if k not in ('lulc', 'lithology') else 50, ha='right', fontsize=7); ax.tick_params(axis='y', labelsize=7)
    ax.set_ylabel('%', fontsize=8); ax.set_title(t, loc='left', fontsize=9.5); ax.grid(axis='x', visible=False)
h = [plt.Rectangle((0, 0), 1, 1, fc='#c6dbef'), plt.Rectangle((0, 0), 1, 1, fc='#fcbba1'), plt.Line2D([], [], color=RED, marker='o', lw=1.6)]
fig.legend(h, ['% of country area (bars)', '% of landslides (bars)', 'Frequency ratio (line, right axis; FR > 1 = over-represented)'], loc='upper left', ncol=3, bbox_to_anchor=(0.02, 0.955), fontsize=8.5)
fig.suptitle('Bivariate frequency-ratio analysis of landslide occurrence', x=0.02, ha='left', fontweight='bold', fontsize=12, y=0.99)
for a_ in axs.flat[len(titles):]: a_.axis('off')
save(fig, 'chart_06_frequency_ratio')

# C07 ROC + C08 metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
te = S[S.rand >= 0.7]; tr = S[S.rand < 0.7]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.9), gridspec_kw={'wspace': 0.3})
for ax, data, ttl in [(axs[0], te, f'Test set (success-independent, n = {len(te)})'), (axs[1], tr, f'Training set (n = {len(tr)})')]:
    for col, c, nm in [('p_rf', BLUE, 'Random Forest'), ('p_gbt', ORANGE, 'Gradient Tree Boosting')]:
        f_, t_, _ = roc_curve(data.cls, data[col]); ax.plot(f_, t_, color=c, lw=2, label=f'{nm}  AUC = {auc(f_, t_):.3f}')
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls='--', label='No skill  AUC = 0.500')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect('equal'); ax.set_xlabel('False positive rate (1 − specificity)'); ax.set_ylabel('True positive rate (sensitivity)')
    ax.set_title(ttl.replace('success-independent, ', ''), loc='left'); ax.legend(loc='lower right', fontsize=8)
fig.suptitle('Receiver operating characteristic curves', x=0.02, ha='left', fontweight='bold', fontsize=12, y=1.0)
save(fig, 'chart_07_roc_curves')

fig, axs = plt.subplots(1, 3, figsize=(13, 4.1), gridspec_kw={'width_ratios': [1.5, 1, 1], 'wspace': 0.35})
rows = []
for col, nm in [('p_rf', 'Random Forest'), ('p_gbt', 'Gradient Tree Boosting')]:
    pr = (te[col] >= 0.5).astype(int); tn, fp, fn, tp = confusion_matrix(te.cls, pr).ravel(); n = tn + fp + fn + tp
    oa = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    f_, t_, _ = roc_curve(te.cls, te[col])
    rows.append([nm, auc(f_, t_), oa, (oa - pe) / (1 - pe), tp / (tp + fn), tn / (tn + fp), tp / (tp + fp), 2 * tp / (2 * tp + fp + fn), (tn, fp, fn, tp)])
mt = pd.DataFrame(rows, columns=['model', 'AUC', 'Accuracy', 'Kappa', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'cm'])
ax = axs[0]; ms = ['AUC', 'Accuracy', 'Kappa', 'Sensitivity', 'Specificity', 'Precision', 'F1']; x = np.arange(len(ms))
for k, (c, (_, r)) in enumerate(zip([BLUE, ORANGE], mt.iterrows())):
    ax.bar(x + (k - 0.5) * 0.38, r[ms].astype(float), 0.36, color=c, label=r.model)
    for xi, v in zip(x, r[ms]): ax.text(xi + (k - 0.5) * 0.38, v + 0.01, f'{v:.2f}', ha='center', fontsize=6.5, rotation=90, color=INK)
ax.set_xticks(x); ax.set_xticklabels(ms, fontsize=8, rotation=30, ha='right'); ax.set_ylim(0.6, 1.08); ax.set_ylabel('score'); ax.grid(axis='x', visible=False)
ax.legend(loc='upper right', fontsize=7.5, ncol=2); ax.set_title('Test-set performance (threshold 0.5)', loc='left')
for ax, (_, r), c in zip(axs[1:], mt.iterrows(), ['Blues', 'Oranges']):
    tn, fp, fn, tp = r.cm; cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap=c, vmin=0, vmax=cm.max() * 1.2); ax.grid(False)
    for i in range(2):
        for j in range(2): ax.text(j, i, f'{cm[i, j]}\n({cm[i, j] / cm[i].sum() * 100:.0f}%)', ha='center', va='center', fontsize=10, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Non-landslide', 'Landslide']); ax.set_yticks([0, 1]); ax.set_yticklabels(['Non-landslide', 'Landslide'], rotation=90, va='center')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Observed'); ax.set_title(f'Confusion matrix — {r.model.replace("Gradient Tree Boosting", "GTB")}', loc='left', fontsize=9.5)
    for s_ in ax.spines.values(): s_.set_visible(False)
save(fig, 'chart_08_model_metrics')
mt.drop(columns='cm').round(3).to_csv('out/table_model_metrics.csv', index=False)

# C09 importance
im = pd.read_csv('data/kg_ls_rf_importance.csv').sort_values('importance_pct')
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.barh([SHORT.get(k, k) for k in im.factor], im.importance_pct, color=BLUE, height=0.62)
for k, v in enumerate(im.importance_pct): ax.text(v, k, f' {v:.1f}', va='center', fontsize=8)
ax.set_xlabel('Share of total Gini importance (%)'); ax.grid(axis='y', visible=False); ax.margins(x=0.12)
ax.set_title('Random Forest variable importance', loc='left')
save(fig, 'chart_09_rf_importance')

# C10 class stats
cs = pd.read_csv('data/kg_ls_class_stats.csv'); x = np.arange(5)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.bar(x - 0.2, cs.area_pct, 0.38, color=CLASS_COLORS, edgecolor='#777', lw=0.5, label='% of country area')
ax.bar(x + 0.2, cs.landslide_pct, 0.38, color=CLASS_COLORS, edgecolor='#222', lw=0.5, hatch='////', label='% of mapped landslides')
for xi, a, l in zip(x, cs.area_pct, cs.landslide_pct):
    ax.text(xi - 0.2, a + 1, f'{a:.1f}', ha='center', fontsize=8); ax.text(xi + 0.2, l + 1, f'{l:.1f}', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f'{c}\n{a:,.0f} km²' for c, a in zip(cs['class'], cs.area_km2)], fontsize=8.5)
ax.set_ylabel('%'); ax.grid(axis='x', visible=False); ax.set_ylim(0, 85)
for xi, v in zip(x, cs.FR): ax.text(xi, 80, f'FR {v:.2f}', ha='center', fontsize=8, color=RED)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc='white', ec='#777', label='% of country area'), Patch(fc='white', ec='#222', hatch='////', label='% of mapped landslides')], loc='upper center', bbox_to_anchor=(0.45, 0.93))
ax.set_title('Susceptibility classes: area share vs landslide share', loc='left')
fig.text(0.01, -0.02, 'FR = landslide share / area share. Landslide share uses all landslide cells (training and test).', fontsize=7.5, color='#555')
save(fig, 'chart_10_class_area_vs_landslides')

# C11 class by oblast
ob = pd.read_csv('data/kg_ls_class_by_oblast.csv').sort_values('high_plus_very_high_pct')
fig, ax = plt.subplots(figsize=(9, 4.8)); left = np.zeros(len(ob))
for c, col in zip(CLASS_NAMES, CLASS_COLORS):
    ax.barh(ob.oblast, ob[c], left=left, color=col, edgecolor='#8a8a8a', lw=0.5, label=c, height=0.7); left += ob[c].values
for k, (v, a) in enumerate(zip(ob.high_plus_very_high_pct, ob.area_km2)): ax.text(101, k, f'{v:.1f}%  ({a:,.0f} km²)', va='center', fontsize=8)
ax.set_xlim(0, 100); ax.set_xlabel('% of unit area'); ax.grid(False)
ax.legend(ncol=5, loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=8)
ax.text(101, len(ob) - 0.35, 'High + very high\n(unit area)', fontsize=7.5, color='#555', va='bottom')
ax.set_title('Susceptibility class shares by oblast', loc='left', pad=26)
save(fig, 'chart_11_classes_by_oblast')

# C12 probability distributions
fig, ax = plt.subplots(figsize=(8, 4.2)); bins = np.linspace(0, 1, 26)
for d, c, nm in [(te[te.cls == 0], BLUE, 'Non-landslide (test)'), (te[te.cls == 1], RED, 'Landslide (test)')]:
    ax.hist(d.p_rf, bins=bins, color=c, alpha=0.55, label=f'{nm}, n = {len(d)}')
for t in [0.2, 0.4, 0.6, 0.8]: ax.axvline(t, color='#888', lw=0.6, ls=':')
ax.set_xlabel('Random Forest probability'); ax.set_ylabel('cells'); ax.legend(loc='upper center'); ax.grid(axis='x', visible=False)
ax.set_title('Separation of test cells by predicted probability (dotted: class breaks)', loc='left')
save(fig, 'chart_12_probability_separation')
