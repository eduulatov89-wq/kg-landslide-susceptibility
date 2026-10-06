"""Susceptibility statistics by raion (district) and city — ADM2 boundaries (MoES 2018, kgz_admbnda_adm2_moes_20181119).
Statistics only; the raion boundaries are not drawn on any map."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, shapefile, matplotlib
from shapely.geometry import shape, mapping, Point
from rasterio import features
from common import *
from make_maps import CLASS_NAMES, CLASS_COLORS
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

d = f'{SCR}/adm2/adm2'
r = shapefile.Reader(shp=open(d + '.shp', 'rb'), dbf=open(d + '.dbf', 'rb'), encoding='utf-8')
U = [(rec['ADM2_EN'], rec['ADM1_EN'], rec['ADM2TYP_EN'], proj_geom(shape(s.__geo_interface__))) for rec, s in zip(r.records(), r.iterShapes())]
RID = features.rasterize([(mapping(g), i + 1) for i, (_, _, _, g) in enumerate(U)], out_shape=(H, W), transform=AFF, fill=0, dtype='int16')
GR = np.load('data/ml_prob_grids.npz'); ORDER = ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN']
p = GR['RF']; valid = ~np.isnan(p)
cls = np.clip(np.floor(p * 5), 0, 4)
agree = sum((np.clip(np.floor(GR[m] * 5), 0, 4) >= 3).astype(int) for m in ORDER)
E = np.load('data/ml_ensemble.npz'); sd = E['sd']
pix = RES * RES / 1e6
# landslide points and cells
LSI = ((Y1 - LS[:, 1]) / RES).astype(int); LSJ = ((LS[:, 0] - X0) / RES).astype(int)
k = (LSI >= 0) & (LSI < H) & (LSJ >= 0) & (LSJ < W)
pt_unit = np.zeros(len(LS), int); pt_unit[k] = RID[LSI[k], LSJ[k]]
lsc = np.zeros((H, W), bool); lsc[LSI[k], LSJ[k]] = True; lsc &= valid
rows = []
for i, (nm, ob, typ, g) in enumerate(U, start=1):
    m = (RID == i) & valid
    if m.sum() == 0: continue
    row = dict(raion=nm, oblast=ob, type=typ, area_km2=m.sum() * pix, landslides=int((pt_unit == i).sum()))
    row['landslide_density_per_1000km2'] = row['landslides'] / row['area_km2'] * 1000
    for c, n in enumerate(CLASS_NAMES): row[f'{n}_pct'] = (cls[m] == c).mean() * 100
    row['high_plus_very_high_pct'] = row['High_pct'] + row['Very high_pct']
    row['high_plus_very_high_km2'] = ((cls[m] >= 3).sum()) * pix
    row['mean_RF_probability'] = float(np.nanmean(p[m]))
    row['consensus_6of6_pct'] = (agree[m] == 6).mean() * 100
    row['mean_between_model_sd'] = float(np.nanmean(sd[m]))
    lm = lsc & (RID == i)
    row['landslide_cells_in_high_plus_very_high_pct'] = (cls[lm] >= 3).mean() * 100 if lm.sum() else np.nan
    rows.append(row)
T = pd.DataFrame(rows).sort_values('high_plus_very_high_pct', ascending=False)
T.round(3).to_csv('out/table_raion_statistics.csv', index=False)
print('units', len(T), 'points assigned', (pt_unit > 0).sum(), 'of', len(LS))
print(T[['raion', 'oblast', 'area_km2', 'landslides', 'high_plus_very_high_pct', 'consensus_6of6_pct']].round(1).head(12).to_string())

# chart: class shares by raion (sorted), landslide count annotated
T2 = T.sort_values('high_plus_very_high_pct'); T2 = T2[T2.area_km2 > 50]
fig, ax = plt.subplots(figsize=(9.5, 11)); left = np.zeros(len(T2))
lab = [f'{a} ({o})' for a, o in zip(T2.raion, T2.oblast)]
for n, col in zip(CLASS_NAMES, CLASS_COLORS):
    ax.barh(lab, T2[f'{n}_pct'], left=left, color=col, edgecolor='#8a8a8a', lw=0.4, label=n, height=0.75); left += T2[f'{n}_pct'].values
for j, (v, nls) in enumerate(zip(T2.high_plus_very_high_pct, T2.landslides)): ax.text(101, j, f'{v:.1f}%   n = {nls}', va='center', fontsize=7.2)
ax.set_xlim(0, 100); ax.set_xlabel('% of unit area (RF susceptibility classes)'); ax.tick_params(axis='y', labelsize=7.5); ax.grid(False)
ax.legend(ncol=5, loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False)
ax.text(101, len(T2) - 0.2, 'High + very high,\nmapped landslides', fontsize=7, color='#555', va='bottom')
ax.set_title('Susceptibility class shares by raion', loc='left', pad=24, fontweight='bold')
fig.text(0.01, 0.0, 'Units < 50 km² omitted from the chart (listed in the table). Boundaries: MoES (2018) ADM2.', fontsize=7.5, color='#555')
fig.savefig('out/chart_14_classes_by_raion.png', dpi=220, bbox_inches='tight', pad_inches=0.15); plt.close(fig); print('saved')
