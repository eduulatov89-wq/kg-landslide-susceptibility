import sys, time, json; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from common import *
from make_maps import snap, WC_CODES
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score, recall_score, precision_score, f1_score, confusion_matrix, brier_score_loss, matthews_corrcoef
from sklearn.inspection import permutation_importance
import joblib
SEED = 42

# ---- predictor grids at 250 m, full precision (GeoTIFFs exported from Earth Engine)
G = {k: decode(k) for k in ['elevation', 'slope', 'aspect', 'plan_curv', 'profile_curv', 'twi', 'rain_annual',
                            'dist_roads', 'dist_rivers', 'ndvi', 'clay', 'sand', 'relief', 'rain_wetq']}
flat = G['aspect'] < 0
a = np.radians(G['aspect'])
G['northness'] = np.where(flat, 0, np.cos(a)); G['eastness'] = np.where(flat, 0, np.sin(a))
G['northness'][~MASK] = np.nan; G['eastness'][~MASK] = np.nan
for s_ in ('clay', 'sand'):                       # SoilGrids gaps (water, rock, ice) -> 0, as in the 500 m version
    G[s_] = np.where(MASK & np.isnan(G[s_]), 0, G[s_])
G['lulc'] = snap('lulc', WC_CODES + [95])
# distance to active faults (Tien Shan active fault database, user-supplied; UTM 43N polylines), km
import shapefile as _shp
from rasterio import features as _feat
from shapely.geometry import shape as _shape, mapping as _mapping
_fl = _shp.Reader(SCR + '/faults/TS_Active_Flts_update')
_fr = _feat.rasterize([(_mapping(_shape(s.__geo_interface__)), 1) for s in _fl.shapes()], out_shape=(H, W), transform=AFF,
                      fill=0, dtype='uint8', all_touched=True).astype(bool)
G['dist_faults'] = ndimage.distance_transform_edt(~_fr, sampling=RES) / 1000.0
G['dist_faults'][~MASK] = np.nan
G['lithology'] = decode('lithology')
PRED = ['elevation', 'slope', 'northness', 'eastness', 'plan_curv', 'profile_curv', 'twi', 'rain_annual', 'lulc', 'lithology',
        'dist_roads', 'dist_rivers', 'dist_faults', 'ndvi', 'clay', 'sand']
CAT = ['lulc', 'lithology']
NUM = [p for p in PRED if p not in CAT]

if __name__ == '__main__':
    S = pd.read_csv('data/samples.csv')
    J = np.clip(((S.x - X0) / (X1 - X0) * W).astype(int), 0, W - 1); I = np.clip(((Y1 - S.y) / (Y1 - Y0) * H).astype(int), 0, H - 1)
    for p in PRED: S[p] = G[p][I, J]
    S = S.dropna(subset=PRED).reset_index(drop=True)
    tr, te = S[S.rand < 0.7], S[S.rand >= 0.7]
    Xtr, ytr, Xte, yte = tr[PRED], tr.cls.values, te[PRED], te.cls.values
    print('samples', len(S), 'train', len(tr), 'test', len(te))

    def prep(scale):
        t = [('num', StandardScaler() if scale else 'passthrough', NUM),
             ('lc', OneHotEncoder(handle_unknown='ignore', sparse_output=False) if scale else 'passthrough', CAT)]
        return ColumnTransformer(t)

    MODELS = {
     'RF':  (make_pipeline(prep(False), RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=2)),
             {'randomforestclassifier__max_features': [2, 4, 'sqrt', 0.5], 'randomforestclassifier__min_samples_leaf': [1, 3, 5]}),
     'GBT': (make_pipeline(prep(False), GradientBoostingClassifier(random_state=SEED)),
             {'gradientboostingclassifier__n_estimators': [200, 400], 'gradientboostingclassifier__learning_rate': [0.03, 0.1],
              'gradientboostingclassifier__max_depth': [2, 3, 4], 'gradientboostingclassifier__subsample': [0.7]}),
     'XGBoost': (make_pipeline(prep(False), XGBClassifier(random_state=SEED, n_jobs=2, eval_metric='logloss', tree_method='hist')),
             {'xgbclassifier__n_estimators': [300, 600], 'xgbclassifier__learning_rate': [0.03, 0.1], 'xgbclassifier__max_depth': [3, 5],
              'xgbclassifier__subsample': [0.8], 'xgbclassifier__colsample_bytree': [0.8], 'xgbclassifier__min_child_weight': [1, 3]}),
     'SVM': (make_pipeline(prep(True), SVC(kernel='rbf', probability=True, random_state=SEED)),
             {'svc__C': [1, 10, 30], 'svc__gamma': ['scale', 0.03, 0.1]}),
     'DNN': (make_pipeline(prep(True), MLPClassifier(max_iter=2000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED)),
             {'mlpclassifier__hidden_layer_sizes': [(64, 32), (128, 64, 32), (128, 64, 32, 16)], 'mlpclassifier__alpha': [1e-4, 1e-3, 1e-2],
              'mlpclassifier__learning_rate_init': [1e-3]}),
     'KNN': (make_pipeline(prep(True), KNeighborsClassifier()),
             {'kneighborsclassifier__n_neighbors': [5, 9, 15, 21, 31], 'kneighborsclassifier__weights': ['uniform', 'distance']}),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best, rows, probs, cvauc, cvsp = {}, [], {}, {}, {}
    # 5 x 5 repeated spatial-block CV: 50 km blocks randomly assigned to 5 folds, 5 different assignments
    bid = (S.x // 50000).astype(int) * 1000 + (S.y // 50000).astype(int)
    ub = np.unique(bid); SPLITS = []
    for rep in range(5):
        fold_of = dict(zip(ub, np.random.default_rng(SEED + rep).integers(0, 5, len(ub))))
        f = bid.map(fold_of).values
        for k in range(5):
            te_i = np.where(f == k)[0]; tr_i = np.where(f != k)[0]
            if len(np.unique(S.cls.values[te_i])) == 2: SPLITS.append((tr_i, te_i))
    print('spatial blocks', len(ub), 'folds', len(SPLITS), flush=True)
    for name, (pipe, grid) in MODELS.items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, scoring='roc_auc', cv=cv, n_jobs=2).fit(Xtr, ytr)
        m = gs.best_estimator_; best[name] = m
        p = m.predict_proba(Xte)[:, 1]; probs[name] = p; yh = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
        rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
        cvauc[name] = cross_val_score(m, S[PRED], S.cls.values, scoring='roc_auc', cv=rcv, n_jobs=2)
        cvsp[name] = cross_val_score(m, S[PRED], S.cls.values, scoring='roc_auc', cv=SPLITS, n_jobs=2)
        rows.append(dict(model=name, AUC=roc_auc_score(yte, p), train_AUC=roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1]),
                         Accuracy=accuracy_score(yte, yh), Kappa=cohen_kappa_score(yte, yh), Sensitivity=recall_score(yte, yh),
                         Specificity=tn / (tn + fp), Precision=precision_score(yte, yh), F1=f1_score(yte, yh), MCC=matthews_corrcoef(yte, yh),
                         Brier=brier_score_loss(yte, p), CV_AUC_mean=cvauc[name].mean(), CV_AUC_sd=cvauc[name].std(), SCV_AUC_mean=cvsp[name].mean(), SCV_AUC_sd=cvsp[name].std(),
                         TP=tp, FN=fn, FP=fp, TN=tn, best_params=json.dumps({k.split('__')[1]: v for k, v in gs.best_params_.items()}),
                         fit_s=round(time.time() - t0, 1)))
        print(name, round(rows[-1]['AUC'], 4), round(cvauc[name].mean(), 4), 'spatial', round(cvsp[name].mean(), 4), gs.best_params_, rows[-1]['fit_s'], 's', flush=True)
    M = pd.DataFrame(rows); M.to_csv('data/ml_metrics.csv', index=False)
    for name, m in best.items(): S['p_' + name] = m.predict_proba(S[PRED])[:, 1]
    S.to_csv('data/samples_with_probs.csv', index=False)
    pd.DataFrame(probs).assign(cls=yte).to_csv('data/ml_test_probs.csv', index=False)
    pd.DataFrame(cvauc).to_csv('data/ml_cv_auc.csv', index=False)
    pd.DataFrame(cvsp).to_csv('data/ml_cv_auc_spatial.csv', index=False)
    joblib.dump(best, 'data/ml_models.joblib')

    # permutation importance on the test set
    imp = {}
    for name, m in best.items():
        r = permutation_importance(m, Xte, yte, scoring='roc_auc', n_repeats=20, random_state=SEED, n_jobs=2)
        imp[name] = r.importances_mean
    pd.DataFrame(imp, index=PRED).to_csv('data/ml_perm_importance.csv')

    # prediction over the whole grid
    mask = MASK & np.all([~np.isnan(G[p]) for p in PRED], axis=0)
    Xall = pd.DataFrame({p: G[p][mask] for p in PRED})
    out = {}
    for name, m in best.items():
        pr = np.full(MASK.shape, np.nan, dtype=np.float32)
        pr[mask] = np.concatenate([m.predict_proba(Xall.iloc[i:i + 100000])[:, 1] for i in range(0, len(Xall), 100000)])
        out[name] = pr; print('predicted', name, flush=True)
    np.savez_compressed('data/ml_prob_grids.npz', **out)
    joblib.dump(best, 'data/ml_models.joblib')
    print(M[['model', 'AUC', 'CV_AUC_mean', 'CV_AUC_sd', 'Accuracy', 'Kappa', 'F1', 'train_AUC']].round(3).to_string())
