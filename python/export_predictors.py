"""Export the 16 predictor layers used by the models as GeoTIFFs (EPSG:32643, 250 m), grouped by source licence."""
import sys; sys.path.insert(0, '.')
import numpy as np, rasterio
from ml import G, PRED
from common import W, H, AFF, MASK
D = 'deposit/'
prof = dict(driver='GTiff', width=W, height=H, crs='EPSG:32643', transform=AFF, compress='deflate', predictor=3, zlevel=9,
            tiled=True, blockxsize=512, blockysize=512, dtype='float32', nodata=-9999.0)
GROUPS = {
  'kg250_predictors_open_CC-BY-4.0.tif': ['elevation', 'slope', 'northness', 'eastness', 'plan_curv', 'profile_curv',
                                          'rain_annual', 'lulc', 'dist_roads', 'ndvi', 'clay', 'sand'],
  'kg250_predictors_MERIT-Hydro_CC-BY-NC-4.0.tif': ['twi', 'dist_rivers'],
  'kg250_predictors_HOLD_faults_lithology.tif': ['dist_faults', 'lithology'],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(PRED)
for fn, bands in GROUPS.items():
    with rasterio.open(D + fn, 'w', count=len(bands), **prof) as d:
        for i, b in enumerate(bands, 1):
            a = np.where(MASK & ~np.isnan(G[b]), G[b], -9999.0).astype('float32'); d.write(a, i); d.set_band_description(i, b)
    print(fn, bands)
