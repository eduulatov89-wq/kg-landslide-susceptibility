"""Recompute every statistic that previously came from Earth Engine CSVs, using the new inventory and local grids."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, joblib
from common import *
from ml import G, PRED, NUM, CAT
from make_maps import WC_CODES, WC_NAMES, CLASS_NAMES
S = pd.read_csv('data/samples_with_probs.csv')
S['p_rf'] = S['p_RF']; S['p_gbt'] = S['p_GBT']
S.to_csv('data/samples.csv', index=False)

CAND = ['elevation', 'slope', 'northness', 'eastness', 'plan_curv', 'profile_curv', 'relief', 'twi', 'rain_annual', 'rain_wetq',
        'lulc', 'lithology', 'dist_roads', 'dist_rivers', 'dist_faults', 'ndvi', 'clay', 'sand']
for c in CAND: S[c] = G[c][S.row, S.col]
# VIF
def vif(cols):
    R = S[cols].corr().values; return np.diag(np.linalg.inv(R))
rows = [dict(**{'system:index': f'1_{i}'}, VIF=v, factor=c) for i, (c, v) in enumerate(zip(CAND, vif(CAND)))]
rows += [dict(**{'system:index': f'2_{i}'}, VIF=v, factor=c) for i, (c, v) in enumerate(zip(PRED, vif(PRED)))]
pd.DataFrame(rows).to_csv('data/kg_ls_vif.csv', index=False)
print('max VIF model set', vif(PRED).max().round(2), '| slope-relief r', S[['slope', 'relief']].corr().iloc[0, 1].round(3))
# frequency ratio
valid = MASK & np.all([~np.isnan(G[p]) for p in PRED], axis=0)
pres = S[S.cls == 1]
SPECS = [('slope', [0, 5, 10, 15, 20, 25, 30, 35, 90]), ('elevation', [0, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 8000]),
         ('relief', [0, 100, 200, 300, 400, 600, 800, 5000]), ('rain_annual', [0, 200, 300, 400, 500, 600, 800, 3000]),
         ('dist_roads', [0, 0.5, 1, 2, 5, 10, 50]), ('dist_rivers', [0, 0.25, 0.5, 1, 2, 5, 50]), ('dist_faults', [0, 1, 2, 5, 10, 20, 200]),
         ('ndvi', [-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1]), ('twi', [0, 5, 6, 7, 8, 9, 10, 30])]
fr = []
for k, e in SPECS:
    a = G[k][valid]; l = pres[k].values; n = len(e) - 1
    for i in range(n):
        lo, hi = e[i], e[i + 1]; last = i == n - 1
        ap = ((a >= lo) & ((a <= hi) if last else (a < hi))).mean() * 100
        lp = ((l >= lo) & ((l <= hi) if last else (l < hi))).mean() * 100
        fr.append(dict(FR=lp / ap if ap > 0 else 0, area_pct=ap, bin=(f'≥ {lo}' if last else f'{lo}–{hi}'), factor=k, landslide_pct=lp))
a = G['lulc'][valid]
for c, nm in zip(WC_CODES, WC_NAMES):
    ap = (a == c).mean() * 100; lp = (pres.lulc == c).mean() * 100
    if ap > 0.05: fr.append(dict(FR=lp / ap, area_pct=ap, bin=nm, factor='lulc', landslide_pct=lp))
a = G['lithology'][valid]
for c, nm in zip(LITH_CODES, LITH_NAMES):
    ap = (a == c).mean() * 100; lp = (pres.lithology == c).mean() * 100
    if ap > 0.05: fr.append(dict(FR=lp / ap, area_pct=ap, bin=nm, factor='lithology', landslide_pct=lp))
pd.DataFrame(fr).to_csv('data/kg_ls_frequency_ratio.csv', index=False)
# RF gini importance
best = joblib.load('data/ml_models.joblib'); rf = best['RF'][-1]
imp = pd.DataFrame({'factor': NUM + CAT, 'importance_pct': rf.feature_importances_ / rf.feature_importances_.sum() * 100}).sort_values('importance_pct', ascending=False)
imp.to_csv('data/kg_ls_rf_importance.csv', index=False)
# class stats from local RF probability grid
GR = np.load('data/ml_prob_grids.npz'); p = GR['RF']; OB = np.load('data/oblast_grid.npy')
cls = np.where(np.isnan(p), np.nan, np.clip(np.floor(p * 5), 0, 4)); v = ~np.isnan(cls)
pixkm2 = (X1 - X0) / W * (Y1 - Y0) / H / 1e6
lc = cls[pres.row, pres.col]
cs = []
for k, nm in enumerate(CLASS_NAMES):
    ap = (cls[v] == k).mean() * 100; lp = (lc == k).mean() * 100
    cs.append(dict(FR=lp / ap, area_km2=(cls[v] == k).sum() * pixkm2, area_pct=ap, **{'class': nm}, landslide_pct=lp))
pd.DataFrame(cs).to_csv('data/kg_ls_class_stats.csv', index=False)
NAMES = {2: 'Issyk-Kul', 3: 'Jalal-Abad', 4: 'Naryn', 5: 'Batken', 6: 'Osh', 7: 'Talas', 8: 'Chui', 11: 'Bishkek city', 21: 'Osh city'}
ob = []
for oid, nm in NAMES.items():
    m = v & (OB == oid)
    if m.sum() == 0: continue
    d = {n: (cls[m] == k).mean() * 100 for k, n in enumerate(CLASS_NAMES)}
    d.update(area_km2=m.sum() * pixkm2, high_plus_very_high_pct=d['High'] + d['Very high'], oblast=nm); ob.append(d)
pd.DataFrame(ob).to_csv('data/kg_ls_class_by_oblast.csv', index=False)
print(pd.DataFrame(cs).round(2).to_string()); print(pd.DataFrame(ob)[['oblast', 'high_plus_very_high_pct']].round(1).to_string())
