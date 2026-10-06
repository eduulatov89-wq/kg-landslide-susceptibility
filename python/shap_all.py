"""SHAP for all six models on a common, balanced subset of test cells, on the probability scale.
Trees (RF, GBT, XGBoost): TreeExplainer, interventional, model_output='probability'.
SVM, DNN, KNN: model-agnostic permutation explainer on the full pipeline (raw factors in, probability out)."""
import sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd, joblib, shap
from ml import PRED, NUM, CAT
SEED = 42; N = 400; NB = 100
S = pd.read_csv('data/samples.csv'); tr = S[S.rand < 0.7]; te = S[S.rand >= 0.7]
sub = pd.concat([te[te.cls == 1].sample(N // 2, random_state=SEED), te[te.cls == 0].sample(N // 2, random_state=SEED)])
bg = tr.sample(NB, random_state=SEED)
best = joblib.load('data/ml_models.joblib'); cols = NUM + CAT
out = {}
for name in ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN']:
    t0 = time.time(); m = best[name]
    if name in ('RF', 'GBT', 'XGBoost'):
        Xs = pd.DataFrame(m[:-1].transform(sub[PRED]), columns=cols); Xb = pd.DataFrame(m[:-1].transform(bg[PRED]), columns=cols)
        ex = shap.TreeExplainer(m[-1], data=Xb, feature_perturbation='interventional', model_output='probability')
        sv = ex.shap_values(Xs, check_additivity=False)
        sv = sv[..., 1] if sv.ndim == 3 else sv
        sv = pd.DataFrame(sv, columns=cols)[PRED].values; base = float(np.ravel(ex.expected_value)[-1])
    else:
        f = lambda X: m.predict_proba(pd.DataFrame(X, columns=PRED))[:, 1]
        ex = shap.PermutationExplainer(f, shap.maskers.Independent(bg[PRED].values, max_samples=NB), seed=SEED)
        e = ex(sub[PRED].values, max_evals=2 * len(PRED) * 10 + 1, silent=True)
        sv = e.values; base = float(np.mean(e.base_values))
    out[name] = sv
    pred = m.predict_proba(sub[PRED])[:, 1]
    print(name, sv.shape, 'base %.3f' % base, 'additivity err %.4f' % np.abs(sv.sum(1) + base - pred).mean(), round(time.time() - t0), 's', flush=True)
np.savez_compressed('data/shap_all6_400.npz', X=sub[PRED].values, cls=sub.cls.values, **out)
imp = pd.DataFrame({n: np.abs(v).mean(0) for n, v in out.items()}, index=PRED)
pct = imp / imp.sum() * 100; pct.round(2).to_csv('out/table_shap_mean_abs_6models.csv')
rk = pct.rank(ascending=False).astype(int); print(pct.round(1).assign(mean=pct.mean(1).round(1)).sort_values('mean', ascending=False)); print(rk.loc[pct.mean(1).sort_values(ascending=False).index])
