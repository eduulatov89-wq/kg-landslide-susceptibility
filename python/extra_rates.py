"""Success- and prediction-rate curves (training vs test landslide cells) and classification-scheme comparison
(equal interval, natural breaks / Jenks, quantile) for the six 250 m probability maps and the ensemble mean."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, jenkspy
from common import *
O = ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN']
GR = np.load('data/ml_prob_grids.npz'); E = np.load('data/ml_ensemble.npz')
P = {m: GR[m] for m in O}; P['Ensemble'] = E['mean']
S = pd.read_csv('data/samples.csv')
pres = S[S.cls == 1]
TR = (pres.rand < 0.7).values
v = ~np.isnan(P['RF'])
rows, curves = [], {}
for m, p in P.items():
    pv = p[v]; order = np.sort(pv)[::-1]
    def rate(rr, cc):
        q = p[rr, cc]; q = q[~np.isnan(q)]
        # fraction of area with probability >= each landslide's probability
        area = 1 - np.searchsorted(np.sort(pv), q, side='left') / pv.size
        a = np.sort(area); y = np.arange(1, a.size + 1) / a.size
        auc = np.trapezoid(np.r_[0, y, 1], np.r_[0, a, 1])
        return a, y, auc
    a1, y1, s_auc = rate(pres.row.values[TR], pres.col.values[TR])
    a2, y2, p_auc = rate(pres.row.values[~TR], pres.col.values[~TR])
    curves[m] = (a1, y1, a2, y2)
    top = lambda a, f: (a <= f).mean() * 100
    rows.append(dict(model=m, success_rate_AUC=s_auc, prediction_rate_AUC=p_auc, n_train=int(TR.sum()), n_test=int((~TR).sum()),
                     train_LS_in_top10pct=top(a1, .10), test_LS_in_top10pct=top(a2, .10), train_LS_in_top5pct=top(a1, .05), test_LS_in_top5pct=top(a2, .05)))
R = pd.DataFrame(rows); R.round(4).to_csv('out/table_success_prediction_rates.csv', index=False); print(R.round(3).to_string())
np.savez_compressed('data/rate_curves.npz', **{f'{m}_{k}': c for m, cs in curves.items() for k, c in zip(['tra', 'try', 'tea', 'tey'], cs)})

# ---- classification schemes
L = np.zeros(v.shape, bool); L[pres.row.values, pres.col.values] = True; L &= v
rng = np.random.default_rng(42)
rows = []; BRK = {}
for m, p in P.items():
    pv = p[v]
    ei = np.array([0.2, 0.4, 0.6, 0.8])
    qb = np.quantile(pv, [0.2, 0.4, 0.6, 0.8])
    nb = np.array(jenkspy.jenks_breaks(rng.choice(pv, 20000, replace=False).astype(float), n_classes=5)[1:5])
    BRK[m] = dict(equal=ei, natural=nb, quantile=qb)
    for sch, b in BRK[m].items():
        c = np.digitize(p, b)
        vh = v & (c == 4); hvh = v & (c >= 3)
        ap = vh.sum() / v.sum() * 100; lp = (vh & L).sum() / L.sum() * 100
        rows.append(dict(model=m, scheme=sch, breaks=' / '.join(f'{x:.3f}' for x in b), VH_area_pct=ap, VH_LS_pct=lp, VH_FR=lp / ap,
                         HVH_area_pct=hvh.sum() / v.sum() * 100, HVH_LS_pct=(hvh & L).sum() / L.sum() * 100))
C = pd.DataFrame(rows); C.round(3).to_csv('out/table_classification_schemes.csv', index=False)
print(C.round(2).to_string())
pv = P['RF'][v]
for sch in ('natural', 'quantile'):
    a = np.digitize(pv, BRK['RF']['equal']); b = np.digitize(pv, BRK['RF'][sch])
    print('RF equal vs', sch, 'identical class', (a == b).mean())
