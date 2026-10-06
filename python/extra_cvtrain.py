"""Repeated CV on the TRAINING set only (7,200 cells), so that tuning, CV and the held-out test set are fully separated.
Same 5 x 5 random and 5 x 5 spatial-block designs and tuned pipelines as the main analysis."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, joblib, scipy.stats as st
from ml import G, PRED
from common import *
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
SEED = 42
S = pd.read_csv('data/samples.csv')
J = np.clip(((S.x - X0) / (X1 - X0) * W).astype(int), 0, W - 1); I = np.clip(((Y1 - S.y) / (Y1 - Y0) * H).astype(int), 0, H - 1)
for p in PRED: S[p] = G[p][I, J]
S = S.dropna(subset=PRED); S = S[S.rand < 0.7].reset_index(drop=True)
bid = (S.x // 50000).astype(int) * 1000 + (S.y // 50000).astype(int); ub = np.unique(bid); SPL = []
for rep in range(5):
    f = bid.map(dict(zip(ub, np.random.default_rng(SEED + rep).integers(0, 5, len(ub))))).values
    for k in range(5):
        te = np.where(f == k)[0]; tr = np.where(f != k)[0]
        if len(np.unique(S.cls.values[te])) == 2: SPL.append((tr, te))
print('train cells', len(S), 'blocks', len(ub), 'folds', len(SPL), flush=True)
best = joblib.load('data/ml_models.joblib'); R, P = {}, {}
for n, m in best.items():
    R[n] = cross_val_score(m, S[PRED], S.cls.values, scoring='roc_auc', cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED), n_jobs=2)
    P[n] = cross_val_score(m, S[PRED], S.cls.values, scoring='roc_auc', cv=SPL, n_jobs=2)
    print(n, R[n].mean().round(4), P[n].mean().round(4), flush=True)
R = pd.DataFrame(R); P = pd.DataFrame(P)
R.to_csv('data/ml_cv_auc_trainonly.csv', index=False); P.to_csv('data/ml_cv_auc_spatial_trainonly.csv', index=False)
O = list(best)
out = pd.DataFrame(dict(model=O, random_mean=R.mean().values, random_sd=R.std(ddof=0).values, spatial_mean=P.mean().values, spatial_sd=P.std(ddof=0).values,
                        random_rank=R.rank(axis=1, ascending=False).mean().values, spatial_rank=P.rank(axis=1, ascending=False).mean().values))
out.round(4).to_csv('out/table_cv_trainonly.csv', index=False); print(out.round(3).to_string())
print('Friedman random', st.friedmanchisquare(*[R[c] for c in O])); print('Friedman spatial', st.friedmanchisquare(*[P[c] for c in O]))
print('Wilcoxon spatial XGB vs RF', st.wilcoxon(P['XGBoost'], P['RF']))
