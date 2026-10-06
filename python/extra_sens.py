"""Sensitivity analyses at 250 m (RF and XGBoost, hyperparameters fixed at the grid-search optimum):
  1. absence buffer (0.5, 1, 2, 5 km) and presence:absence ratio (1:1, 1:2, 1:5 at 1 km)
  2. predictor ablation (without satellite NDVI + WorldCover, without roads, without both)
  3. leave-one-oblast-out validation for all six models"""
import sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd, joblib
from ml import G, PRED, CAT, NUM
from common import *
from scipy import ndimage
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
SEED = 42
S0 = pd.read_csv('data/samples.csv')
J = np.clip(((S0.x - X0) / (X1 - X0) * W).astype(int), 0, W - 1); I = np.clip(((Y1 - S0.y) / (Y1 - Y0) * H).astype(int), 0, H - 1)
for p in PRED: S0[p] = G[p][I, J]
S0 = S0.dropna(subset=PRED).reset_index(drop=True)
valid = MASK & np.all([~np.isnan(G[p]) for p in PRED], axis=0)
L = np.zeros(valid.shape, bool); pr = S0[S0.cls == 1]; L[pr.row, pr.col] = True
D = ndimage.distance_transform_edt(~L, sampling=RES)
# landslide points (not only sample cells) define the exclusion distance, as in the original sampling
LP = np.zeros(valid.shape, bool)
ii = ((Y1 - LS[:, 1]) / RES).astype(int); jj = ((LS[:, 0] - X0) / RES).astype(int); k = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
LP[ii[k], jj[k]] = True
DP = ndimage.distance_transform_edt(~LP, sampling=RES)
rng = np.random.default_rng(7)
# neutral test set: test presences vs 20,000 random land cells (any distance, not landslide cells)
cand = np.flatnonzero((valid & ~LP).ravel()); nz = rng.choice(cand, 20000, replace=False)
NR, NC = np.unravel_index(nz, valid.shape)
NEUT = pd.DataFrame({p: G[p][NR, NC] for p in PRED}); NEUT['cls'] = 0
TP = S0[(S0.cls == 1) & (S0.rand >= 0.7)]
NEUT = pd.concat([TP[PRED + ['cls']], NEUT], ignore_index=True)

def rf(): return RandomForestClassifier(n_estimators=500, max_features=4, min_samples_leaf=1, random_state=SEED, n_jobs=2)
def xgb(): return XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=5, min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                                random_state=SEED, n_jobs=2, eval_metric='logloss', tree_method='hist')
def splits(S):
    bid = (S.x // 50000).astype(int) * 1000 + (S.y // 50000).astype(int); ub = np.unique(bid); out = []
    for rep in range(5):
        f = bid.map(dict(zip(ub, np.random.default_rng(SEED + rep).integers(0, 5, len(ub))))).values
        for kk in range(5):
            te = np.where(f == kk)[0]; tr = np.where(f != kk)[0]
            if len(np.unique(S.cls.values[te])) == 2: out.append((tr, te))
    return out
def scv(make, S, cols):
    a = []
    for tr, te in splits(S):
        m = make().fit(S[cols].iloc[tr], S.cls.values[tr]); a.append(roc_auc_score(S.cls.values[te], m.predict_proba(S[cols].iloc[te])[:, 1]))
    return np.mean(a), np.std(a)
Xall = pd.DataFrame({p: G[p][valid] for p in PRED})
base_rf = np.load('data/ml_prob_grids.npz')['RF'][valid]
def national(m, cols):
    return np.concatenate([m.predict_proba(Xall[cols].iloc[i:i + 200000])[:, 1] for i in range(0, len(Xall), 200000)])
Lv = L[valid]

# ---------------- 1. absence buffer and ratio
rows = []
pres = S0[S0.cls == 1].copy()
for buf, ratio in [(1000, 1), (500, 1), (2000, 1), (5000, 1), (1000, 2), (1000, 5)]:
    t0 = time.time()
    if (buf, ratio) == (1000, 1):
        S = S0.copy()
    else:
        c = np.flatnonzero((valid & (DP >= buf)).ravel()); pick = np.random.default_rng(SEED + buf + ratio).choice(c, len(pres) * ratio, replace=False)
        r_, c_ = np.unravel_index(pick, valid.shape)
        ab = pd.DataFrame({p: G[p][r_, c_] for p in PRED}); ab['cls'] = 0; ab['row'] = r_; ab['col'] = c_
        ab['x'] = X0 + (c_ + 0.5) * RES; ab['y'] = Y1 - (r_ + 0.5) * RES
        ab['rand'] = (np.random.default_rng(SEED).permutation(len(ab)) + 0.5) / len(ab)
        S = pd.concat([pres[ab.columns], ab], ignore_index=True)
    tr, te = S[S.rand < 0.7], S[S.rand >= 0.7]
    row = dict(buffer_km=buf / 1000, ratio=f'1:{ratio}', n=len(S))
    for nm, make in [('RF', rf), ('XGBoost', xgb)]:
        m = make().fit(tr[PRED], tr.cls)
        row[f'{nm}_test_AUC'] = roc_auc_score(te.cls, m.predict_proba(te[PRED])[:, 1])
        row[f'{nm}_neutral_AUC'] = roc_auc_score(NEUT.cls, m.predict_proba(NEUT[PRED])[:, 1])
        mu, sd = scv(make, S, PRED); row[f'{nm}_spatialCV_AUC'] = mu; row[f'{nm}_spatialCV_SD'] = sd
        if nm == 'RF':
            q = national(m, PRED); cl = np.clip(np.floor(q * 5), 0, 4)
            row['RF_VH_area_pct'] = (cl == 4).mean() * 100; row['RF_VH_LS_pct'] = (cl[Lv] == 4).mean() * 100
            row['RF_HVH_area_pct'] = (cl >= 3).mean() * 100; row['RF_r_baseline'] = np.corrcoef(q[::10], base_rf[::10])[0, 1]
            top = q >= np.quantile(q, 0.95); row['RF_LS_in_top5pct_area'] = top[Lv].mean() * 100
    rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, round(time.time() - t0), "s", flush=True)
pd.DataFrame(rows).round(4).to_csv('out/table_absence_sensitivity.csv', index=False)

# ---------------- 2. ablation
rows = []
tr, te = S0[S0.rand < 0.7], S0[S0.rand >= 0.7]
for lab, drop in [('all 16 factors', []), ('without NDVI and land cover', ['ndvi', 'lulc']), ('without distance to roads', ['dist_roads']),
                  ('without NDVI, land cover and roads', ['ndvi', 'lulc', 'dist_roads'])]:
    cols = [p for p in PRED if p not in drop]; row = dict(set=lab, n_factors=len(cols))
    for nm, make in [('RF', rf), ('XGBoost', xgb)]:
        m = make().fit(tr[cols], tr.cls); row[f'{nm}_test_AUC'] = roc_auc_score(te.cls, m.predict_proba(te[cols])[:, 1])
        mu, sd = scv(make, S0, cols); row[f'{nm}_spatialCV_AUC'] = mu; row[f'{nm}_spatialCV_SD'] = sd
    rows.append(row); print(row, flush=True)
pd.DataFrame(rows).round(4).to_csv('out/table_predictor_ablation.csv', index=False)

# ---------------- 3. leave-one-oblast-out (six models, tuned pipelines)
best = joblib.load('data/ml_models.joblib')
NAMES = {2: 'Issyk-Kul', 3: 'Jalal-Abad', 4: 'Naryn', 5: 'Batken', 6: 'Osh', 7: 'Talas', 8: 'Chui'}
rows = []
for oid, nm in NAMES.items():
    te = S0[S0.oblast == oid]; tr = S0[S0.oblast != oid]
    row = dict(oblast=nm, n_landslide=int(te.cls.sum()), n_absence=int((te.cls == 0).sum()))
    for mn, pipe in best.items():
        m = clone(pipe).fit(tr[PRED], tr.cls); row[mn] = roc_auc_score(te.cls, m.predict_proba(te[PRED])[:, 1])
    rows.append(row); print(row, flush=True)
T = pd.DataFrame(rows)
w = T.n_landslide / T.n_landslide.sum()
T.loc[len(T)] = dict(oblast='Mean (unweighted)', **{m: T[m].mean() for m in best}); T.round(4).to_csv('out/table_leave_one_oblast_out.csv', index=False)
print(T.round(3).to_string())
