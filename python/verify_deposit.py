"""Check a downloaded copy of the Zenodo data record against the published results.

Usage:  python verify_deposit.py <folder with the files downloaded from Zenodo>

The script applies the six trained models to the predictor values in the sample table. It then checks:
  1. the predicted probabilities against the stored test-set probabilities, and
  2. the resulting test AUCs against table_ml_metrics.csv.
It needs scikit-learn 1.8.0 and XGBoost 3.2.0 (see requirements.txt).
"""
import sys, os, io, zipfile
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score

root = sys.argv[1] if len(sys.argv) > 1 else '.'
def find(name):
    for d, _, fs in os.walk(root):
        if name in fs: return os.path.join(d, name)
    return None
models = joblib.load(find('kg250_trained_models_sklearn1.8_xgboost3.2.joblib'))
S = pd.read_csv(find('kg250_samples_no_coordinates.csv'))
t = find('table_ml_metrics.csv')
if t: M = pd.read_csv(t)
else:
    with zipfile.ZipFile(find('kg250_result_tables.zip')) as z: M = pd.read_csv(io.BytesIO(z.read('table_ml_metrics.csv')))
M = M.set_index('model')
PRED = ['elevation', 'slope', 'northness', 'eastness', 'plan_curv', 'profile_curv', 'twi', 'rain_annual', 'lulc', 'lithology',
        'dist_roads', 'dist_rivers', 'dist_faults', 'ndvi', 'clay', 'sand']
te = S[S.split == 'test']
print(f'{len(S):,} samples ({(S.cls == 1).sum():,} landslide), {len(te):,} in the test set')
ok = True
for m, pipe in models.items():
    p = pipe.predict_proba(te[PRED])[:, 1]
    d = float(np.abs(p - te[f'p_{m}'].values).max()); auc = roc_auc_score(te.cls, p)
    good = d < 1e-6 and abs(auc - M.loc[m, 'AUC']) < 5e-5
    ok &= good
    print(f'{m:8s} max |p - stored| = {d:.2e}   test AUC = {auc:.4f} (published {M.loc[m, "AUC"]:.4f})   {"OK" if good else "MISMATCH"}')
print('All checks passed.' if ok else 'Some checks failed: check package versions.')
sys.exit(0 if ok else 1)
