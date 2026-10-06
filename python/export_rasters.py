"""GeoTIFF exports of the susceptibility products (EPSG:32643, 250 m)."""
import sys; sys.path.insert(0, '.')
import numpy as np, rasterio
from common import *
O = ['RF', 'GBT', 'XGBoost', 'SVM', 'DNN', 'KNN']
GR = np.load('data/ml_prob_grids.npz'); E = np.load('data/ml_ensemble.npz')
prof = dict(driver='GTiff', width=W, height=H, crs='EPSG:32643', transform=AFF, compress='deflate', tiled=True, blockxsize=512, blockysize=512)
def u16(a): return np.where(np.isnan(a), 65535, np.round(a * 10000)).astype('uint16')
def write(fn, bands, names, dtype, nodata):
    with rasterio.open(fn, 'w', count=len(bands), dtype=dtype, nodata=nodata, **prof) as d:
        for i, (b, n) in enumerate(zip(bands, names), 1): d.write(b, i); d.set_band_description(i, n)
write('rasters/kg_susceptibility_models_prob_x10000_250m.tif', [u16(GR[m]) for m in O], O, 'uint16', 65535)
write('rasters/kg_susceptibility_ensemble_prob_x10000_250m.tif', [u16(E['mean']), u16(E['sd'])], ['ensemble_mean', 'ensemble_sd'], 'uint16', 65535)
cl = lambda p: np.where(np.isnan(p), 255, np.clip(np.floor(p * 5), 0, 4)).astype('uint8')
ag = sum((np.clip(np.floor(np.nan_to_num(GR[m], nan=0) * 5), 0, 4) >= 3).astype('uint8') for m in O); ag[np.isnan(E['mean'])] = 255
write('rasters/kg_susceptibility_classes_250m.tif', [cl(GR['RF']), cl(E['mean']), ag], ['RF_class_0-4', 'ensemble_class_0-4', 'consensus_models_high'], 'uint8', 255)
L = decode('lithology'); write('rasters/kg_lithology_groups_250m.tif', [np.where(np.isnan(L), 255, L).astype('uint8')], ['lithology_group_1-11'], 'uint8', 255)
print('ok')
