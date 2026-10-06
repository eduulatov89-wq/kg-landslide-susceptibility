import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from common import *
from ml import G, PRED  # decoded predictor grids
from scipy import ndimage
from matplotlib.path import Path
rng = np.random.default_rng(42)
pts = np.load('data/ls_new_xy.npy')
px = (X1 - X0) / W; py = (Y1 - Y0) / H
J = ((pts[:, 0] - X0) / px).astype(int); I = ((Y1 - pts[:, 1]) / py).astype(int)
valid = MASK & np.all([~np.isnan(G[p]) for p in PRED], axis=0)
ok = (I >= 0) & (I < H) & (J >= 0) & (J < W)
I, J = I[ok], J[ok]; keep = valid[I, J]; I, J = I[keep], J[keep]
cells = np.unique(np.stack([I, J], 1), axis=0)
# oblast raster
yy, xx = np.mgrid[0:H, 0:W]; cx = X0 + (xx + 0.5) * px; cy = Y1 - (yy + 0.5) * py
OB = np.zeros((H, W), np.int16)
for name, oid, g in ADM:
    for p_ in getattr(g, 'geoms', [g]):
        inside = Path(np.array(p_.exterior.coords)).contains_points(np.c_[cx.ravel(), cy.ravel()]).reshape(H, W)
        for hole in p_.interiors: inside &= ~Path(np.array(hole.coords)).contains_points(np.c_[cx.ravel(), cy.ravel()]).reshape(H, W)
        OB[inside] = oid
np.save('data/oblast_grid.npy', OB)
# absence: >= 1 km from any landslide pixel
lsmask = np.zeros((H, W), bool); lsmask[I, J] = True
allI = ((Y1 - pts[:, 1]) / py).astype(int); allJ = ((pts[:, 0] - X0) / px).astype(int)
k = (allI >= 0) & (allI < H) & (allJ >= 0) & (allJ < W); lsmask[allI[k], allJ[k]] = True
dist = ndimage.distance_transform_edt(~lsmask, sampling=(py, px))
cand = np.argwhere(valid & (dist >= 1000))
ab = cand[rng.choice(len(cand), len(cells), replace=False)]
S = pd.DataFrame(np.vstack([cells, ab]), columns=['row', 'col'])
S['cls'] = [1] * len(cells) + [0] * len(ab)
S['x'] = X0 + (S.col + 0.5) * px; S['y'] = Y1 - (S.row + 0.5) * py
S['oblast'] = OB[S.row, S.col]
# stratified 70/30 split
S['rand'] = rng.random(len(S))
for c in (0, 1):
    idx = S.index[S.cls == c]; r = rng.permutation(len(idx)); S.loc[idx, 'rand'] = (r + 0.5) / len(idx)
S.to_csv('data/samples.csv', index=False)
print('landslide pixels', len(cells), 'absence', len(ab), 'train', (S.rand < 0.7).sum(), 'test', (S.rand >= 0.7).sum())
print(S[S.cls == 1].oblast.value_counts().to_dict())
