"""Population exposure (2025) to landslide susceptibility.
WorldPop Global2 R2025A constrained 100 m (primary) and GHS-POP R2023A 2025 epoch (check), both summed to the 250 m grid in GEE
(density x cell area, area-weighted), exported as int32 x10 in raw250/kg250_pop2025b.tif."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, rasterio, shapefile, matplotlib
from shapely.geometry import shape, mapping
from rasterio import features
from scipy import ndimage
from common import *
from make_maps import CLASS_NAMES, CLASS_COLORS
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

with rasterio.open('raw250/kg250_pop2025b.tif') as r:
    assert r.width == W and r.height == H, (r.width, r.height)
    WP = r.read(1).astype(float) / 10; GH = r.read(2).astype(float) / 10
O = ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN']
GR = np.load('data/ml_prob_grids.npz'); E = np.load('data/ml_ensemble.npz')
p = GR['RF']; v = ~np.isnan(p)
cls = np.where(v, np.clip(np.floor(np.nan_to_num(p) * 5), 0, 4), -1)
ecl = np.where(v, np.clip(np.floor(np.nan_to_num(E['mean']) * 5), 0, 4), -1)
agree = sum((np.clip(np.floor(np.nan_to_num(GR[m]) * 5), 0, 4) >= 3).astype(int) for m in O)
OB = np.load('data/oblast_grid.npy')
tot_wp_all, tot_gh_all = WP[KGMASK].sum(), GH[KGMASK].sum()
tot_wp, tot_gh = WP[v].sum(), GH[v].sum()
print(f'WorldPop 2025: {tot_wp_all:,.0f} in Kyrgyzstan, {tot_wp:,.0f} in analysed cells; GHS-POP: {tot_gh_all:,.0f} / {tot_gh:,.0f}')

rows = []
for k, n in enumerate(CLASS_NAMES):
    m = v & (cls == k); me = v & (ecl == k)
    rows.append(dict(**{'class': n}, area_pct=m.sum() / v.sum() * 100, pop_worldpop=WP[m].sum(), pop_worldpop_pct=WP[m].sum() / tot_wp * 100,
                     pop_ghs=GH[m].sum(), pop_ghs_pct=GH[m].sum() / tot_gh * 100, pop_worldpop_ensemble=WP[me].sum(),
                     pop_worldpop_ensemble_pct=WP[me].sum() / tot_wp * 100))
C = pd.DataFrame(rows); C.round(2).to_csv('out/table_pop_by_class.csv', index=False); print(C.round(1).to_string())
hv = v & (cls >= 3)
cons = {'none (0/6)': v & (agree == 0), 'split (1-5/6)': v & (agree > 0) & (agree < 6), 'unanimous (6/6)': v & (agree == 6)}
CS = pd.DataFrame([dict(zone=k, area_pct=m.sum() / v.sum() * 100, pop_worldpop=WP[m].sum(), pop_worldpop_pct=WP[m].sum() / tot_wp * 100,
                        pop_ghs=GH[m].sum(), pop_ghs_pct=GH[m].sum() / tot_gh * 100) for k, m in cons.items()])
CS.round(2).to_csv('out/table_pop_by_consensus.csv', index=False); print(CS.round(1).to_string())

# proximity to mapped landslides
I = ((Y1 - LS[:, 1]) / RES).astype(int); J = ((LS[:, 0] - X0) / RES).astype(int); k = (I >= 0) & (I < H) & (J >= 0) & (J < W)
L = np.zeros((H, W), bool); L[I[k], J[k]] = True
dls = ndimage.distance_transform_edt(~L, sampling=RES)
PX = []
for d in (0, 500, 1000, 2000, 5000):
    m = v & (dls <= d)
    PX.append(dict(distance_m=d, label='landslide cell' if d == 0 else f'within {d / 1000:g} km', area_km2=m.sum() * RES * RES / 1e6,
                   pop_worldpop=WP[m].sum(), pop_worldpop_pct=WP[m].sum() / tot_wp * 100, pop_ghs=GH[m].sum(), pop_ghs_pct=GH[m].sum() / tot_gh * 100))
PX = pd.DataFrame(PX); PX.round(2).to_csv('out/table_pop_near_landslides.csv', index=False); print(PX.round(1).to_string())

# oblast
NAMES = {2: 'Issyk-Kul', 3: 'Jalal-Abad', 4: 'Naryn', 5: 'Batken', 6: 'Osh', 7: 'Talas', 8: 'Chui', 11: 'Bishkek city', 21: 'Osh city'}
OBT = []
for oid, nm in NAMES.items():
    m = v & (OB == oid)
    if not m.any(): continue
    OBT.append(dict(oblast=nm, pop_worldpop=WP[m].sum(), pop_high_vhigh=WP[m & (cls >= 3)].sum(), pop_very_high=WP[m & (cls == 4)].sum(),
                    pop_unanimous=WP[m & (agree == 6)].sum(), pop_within_1km_landslide=WP[m & (dls <= 1000)].sum(), pop_ghs_high_vhigh=GH[m & (cls >= 3)].sum()))
OBT = pd.DataFrame(OBT); OBT['pct_high_vhigh'] = OBT.pop_high_vhigh / OBT.pop_worldpop * 100
OBT['share_of_national_exposed'] = OBT.pop_high_vhigh / OBT.pop_high_vhigh.sum() * 100
OBT = OBT.sort_values('pop_high_vhigh', ascending=False); OBT.round(1).to_csv('out/table_pop_by_oblast.csv', index=False); print(OBT.round(1).to_string())

# raion (statistics only)
d2 = f'{SCR}/adm2/adm2'
rr = shapefile.Reader(shp=open(d2 + '.shp', 'rb'), dbf=open(d2 + '.dbf', 'rb'), encoding='utf-8')
U = [(rec['ADM2_EN'], rec['ADM1_EN'], proj_geom(shape(s.__geo_interface__))) for rec, s in zip(rr.records(), rr.iterShapes())]
RID = features.rasterize([(mapping(g), i + 1) for i, (_, _, g) in enumerate(U)], out_shape=(H, W), transform=AFF, fill=0, dtype='int16')
RT = []
for i, (nm, ob, g) in enumerate(U, start=1):
    m = v & (RID == i)
    if not m.any(): continue
    RT.append(dict(raion=nm, oblast=ob, pop_worldpop=WP[m].sum(), pop_high_vhigh=WP[m & (cls >= 3)].sum(), pop_very_high=WP[m & (cls == 4)].sum(),
                   pop_unanimous=WP[m & (agree == 6)].sum(), pop_within_1km_landslide=WP[m & (dls <= 1000)].sum(), pop_ghs_high_vhigh=GH[m & (cls >= 3)].sum()))
RT = pd.DataFrame(RT); RT['pct_high_vhigh'] = RT.pop_high_vhigh / RT.pop_worldpop * 100
RT = RT.sort_values('pop_high_vhigh', ascending=False); RT.round(1).to_csv('out/table_pop_by_raion.csv', index=False)
print(RT.head(12).round(1).to_string()); print('raion pop covered', RT.pop_worldpop.sum(), 'H+VH', RT.pop_high_vhigh.sum())
np.savez_compressed('data/pop2025_250m.npz', wp=WP.astype(np.float32), ghs=GH.astype(np.float32))
