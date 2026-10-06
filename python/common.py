"""250 m full-precision grid (EPSG:32643). Same public API as the 650 m thumbnail version:
X0..Y1, W, H, EXTENT, T, TI, MASK, LAKE, decode(), snap(), load(), ADM, KG, LS, LAKES."""
import os, numpy as np, shapefile, rasterio
from rasterio import features
from rasterio.transform import from_origin
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as stransform, unary_union
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
X0, X1, Y0, Y1 = 3500, 937500, 4335000, 4796000
RES = 250
W, H = 3736, 1844
EXTENT = (X0, X1, Y0, Y1)
AFF = from_origin(X0, Y1, RES, RES)
T = Transformer.from_crs(4326, 32643, always_xy=True)
TI = Transformer.from_crs(32643, 4326, always_xy=True)
NODATA = -32768
SCR = os.environ.get('KG_AUX', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aux'))

def proj_geom(g):
    return stransform(lambda x, y, z=None: T.transform(x, y), g)

sf = shapefile.Reader(f'{SCR}/kg_adm1/kg_adm1_9', encoding='utf-8')
ADM = [(r['ADM1_EN'], r['oblast_id'], proj_geom(shape(s.__geo_interface__))) for r, s in zip(sf.records(), sf.shapes())]
KG = unary_union([g for _, _, g in ADM])
LS = np.load(f'{HERE}/data/ls_new_xy.npy')
_lr = shapefile.Reader(f'{SCR}/lakes/lakes_reserv_main', encoding='utf-8')
LAKES = [(r['Name_ru'], proj_geom(shape(s_.__geo_interface__))) for r, s_ in zip(_lr.records(), _lr.shapes())]

def _rast(geoms):
    return features.rasterize([(mapping(g), 1) for g in geoms], out_shape=(H, W), transform=AFF, fill=0, dtype='uint8').astype(bool)

_cache = f'{HERE}/data/masks250.npz'
if os.path.exists(_cache):
    _m = np.load(_cache); LAKE, KGMASK = _m['lake'], _m['kg']
else:
    LAKE = _rast([g for _, g in LAKES]); KGMASK = _rast([KG])
    np.savez_compressed(_cache, lake=LAKE, kg=KGMASK)
MASK = KGMASK & ~LAKE

# band -> (file, band index 1-based, scale to physical units)
BANDS = {'elevation': ('kg250_terrainA', 1, 1), 'slope': ('kg250_terrainA', 2, 0.1), 'aspect': ('kg250_terrainA', 3, 1),
         'plan_curv': ('kg250_terrainB', 1, 1), 'profile_curv': ('kg250_terrainB', 2, 1),      # 1e-5 1/m units
         'relief': ('kg250_terrainC', 1, 1), 'twi': ('kg250_terrainC', 2, 0.01),
         'rain_annual': ('kg250_envA', 1, 1), 'rain_wetq': ('kg250_envA', 2, 1), 'lulc': ('kg250_envA', 3, 1), 'oblast': ('kg250_envA', 4, 1),
         'dist_roads': ('kg250_envB', 1, 0.001), 'dist_rivers': ('kg250_envB', 2, 0.001),            # km
         'ndvi': ('kg250_envC', 1, 1e-4), 'clay': ('kg250_envC', 2, 0.1), 'sand': ('kg250_envC', 3, 0.1)}  # %
_raw = {}
def raw(name):
    if name not in _raw:
        f, b, _ = BANDS[name]
        with rasterio.open(f'{HERE}/raw250/{f}.tif') as r:
            assert r.width == W and r.height == H and abs(r.transform.a - RES) < 1e-6, (r.width, r.height, r.transform)
            _raw[name] = r.read(b)
    return _raw[name]

def fault_distance():
    """Distance (km) to the nearest active fault (Tien Shan active fault database, user-supplied)."""
    p = f'{HERE}/data/dist_faults.npy'
    if os.path.exists(p): return np.load(p)
    fl = shapefile.Reader(f'{SCR}/faults/TS_Active_Flts_update')
    fr = features.rasterize([(mapping(shape(s.__geo_interface__)), 1) for s in fl.shapes()], out_shape=(H, W), transform=AFF,
                            fill=0, dtype='uint8', all_touched=True).astype(bool)
    d = ndimage.distance_transform_edt(~fr, sampling=RES) / 1000.0
    np.save(p, d); return d

def decode(name, lo=None, hi=None):
    """Physical values; NaN outside the analysis mask or where no data."""
    if name == 'lithology': return lithology()
    if name == 'dist_faults':
        v = fault_distance().astype(float); v[~MASK] = np.nan; return v
    a = raw(name); v = a.astype(float) * BANDS[name][2]
    v[(a == NODATA) | ~MASK] = np.nan
    return v

def snap(name, codes):
    """Categorical layer; no-data cells inside the mask filled from the nearest valid cell."""
    a = raw(name); valid = np.isin(a, np.array(codes))
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    out = a[idx[0], idx[1]].astype(float); out[~MASK] = np.nan
    return out

def _hillshade():
    z = raw('elevation').astype(float); z[z == NODATA] = np.nan
    z = np.where(np.isnan(z), np.nanmean(z), z)
    gy, gx = np.gradient(z, RES)
    slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)

def load(name):
    """Compatibility shim for the old 8-bit loader: only 'hillshade' and 'mask' are used."""
    if name == 'hillshade': return (_hillshade() * 255).astype(np.int16)
    if name == 'mask': return (MASK * 255).astype(np.int16)
    raise KeyError(name)

# ---- lithology (geological formation map, geonode.water.gov.kg layer geonode:geology; user-supplied), grouped classes
LITH_GROUPS = [(1, 'Glaciers', [1]), (2, 'Geosynclinal terrigenous', [2]), (3, 'Geosynclinal carbonate', [3]),
               (4, 'Geosynclinal carbonate–terrigenous', [11]), (5, 'Geosynclinal volcanic and sed.–volcanic', [8, 9]),
               (6, 'Orogenic continental molasse', [4]), (7, 'Orogenic flysch–molasse', [7]), (8, 'Orogenic volcanic', [10, 16]),
               (9, 'Platform sedimentary', [5, 15]), (10, 'Platform intrusive', [14]), (11, 'Pre-geosynclinal and ultrabasic', [6, 13, 17, 18])]
LITH_CODES = [g[0] for g in LITH_GROUPS]; LITH_NAMES = [g[1] for g in LITH_GROUPS]
LITH_COLORS = ['#dff3fb', '#c9a26b', '#7fb3d5', '#a3c4a8', '#8e6c8a', '#f2d48f', '#e3a96b', '#c0504d', '#d9d2a6', '#e58fb3', '#5e7a54']
def lithology():
    p = f'{HERE}/data/lithology_grid.npy'
    if os.path.exists(p): return np.load(p)
    fl = shapefile.Reader(f'{SCR}/geology/geology', encoding='utf-8')
    lut = {o: g for g, _, os_ in LITH_GROUPS for o in os_}
    shp = [(mapping(proj_geom(shape(s.__geo_interface__))), lut.get(r['OBJECTID'], 0)) for r, s in zip(fl.records(), fl.shapes())]
    a = features.rasterize([(g, v) for g, v in shp if v > 0], out_shape=(H, W), transform=AFF, fill=0, dtype='int16')
    idx = ndimage.distance_transform_edt(a == 0, return_distances=False, return_indices=True)   # gaps / lake polygon -> nearest unit
    out = a[idx[0], idx[1]].astype(float); out[~MASK] = np.nan
    np.save(p, out); return out
