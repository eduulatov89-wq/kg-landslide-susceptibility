import sys; sys.path.insert(0, '.')
from mapkit import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm

hyps = LinearSegmentedColormap.from_list('hyps', ['#4a7c59', '#9fbf7a', '#e8e0a8', '#c9a26b', '#9c6b3f', '#7a5a4a', '#f2f2f2'])
F = [  # key, enc lo, enc hi, vmin, vmax, cmap, cbar label, title, source, extend, scale
 ('elevation', 0, 7500, 500, 6500, hyps, 'Elevation (m a.s.l.)', 'Elevation', 'Copernicus GLO-30 DEM, mean of 30 m cells', 'both'),
 ('slope', 0, 60, 0, 40, plt.get_cmap('YlOrBr'), 'Slope (°)', 'Slope', 'Copernicus GLO-30, mean of 50 m slopes', 'max'),
 ('aspect', -1, 360, 0, 360, plt.get_cmap('twilight'), 'Aspect (° from north)', 'Aspect', 'Copernicus GLO-30, circular mean; grey = flat', 'neither'),
 ('plan_curv', -60, 60, -200, 200, plt.get_cmap('RdBu_r'), 'Plan curvature (10⁻⁵ m⁻¹)\n+ converging, − diverging', 'Plan curvature', 'Copernicus GLO-30, Zevenbergen–Thorne, 50 m', 'both'),
 ('profile_curv', -60, 60, -200, 200, plt.get_cmap('RdBu_r'), 'Profile curvature (10⁻⁵ m⁻¹)\n+ concave, − convex', 'Profile curvature', 'Copernicus GLO-30, Zevenbergen–Thorne, 50 m', 'both'),
 ('relief', 0, 1500, 0, 400, plt.get_cmap('PuRd'), 'Local relief (m)', 'Local relief (max − min within 250 m)', 'Copernicus GLO-30, 50 m cells', 'max'),
 ('twi', 0, 25, 4, 14, plt.get_cmap('Blues'), 'TWI  ln(a / tan β)', 'Topographic wetness index', 'MERIT Hydro upstream area and elevation', 'both'),
 ('rain_annual', 0, 1200, 100, 900, plt.get_cmap('GnBu'), 'Precipitation (mm yr⁻¹)', 'Mean annual precipitation 1991–2020', 'CHIRPS v2.0 pentad', 'both'),
 ('rain_wetq', 0, 600, 50, 400, plt.get_cmap('GnBu'), 'Precipitation (mm)', 'Precipitation of the wettest quarter 1991–2020', 'CHIRPS v2.0 pentad (CHELSA bio16 analogue)', 'both'),
 ('dist_roads', 0, 32, 0, 20, plt.get_cmap('magma_r'), 'Distance (km)', 'Distance to roads', 'GRIP4 global roads database', 'max'),
 ('dist_rivers', 0, 26, 0, 4, plt.get_cmap('Blues_r'), 'Distance (km)', 'Distance to rivers', 'MERIT Hydro channels, upstream area ≥ 10 km²', 'max'),
 ('dist_faults', 0, 200, 0, 30, plt.get_cmap('YlOrRd_r'), 'Distance (km)', 'Distance to active faults', 'Tien Shan active fault database', 'max'),
 ('ndvi', -0.2, 0.9, 0, 0.8, plt.get_cmap('YlGn'), 'NDVI', 'Vegetation index (NDVI), June–August median 2019–2023', 'MODIS MOD13Q1 v6.1, 250 m', 'both'),
 ('clay', 0, 60, 5, 35, plt.get_cmap('Oranges'), 'Clay (%)', 'Topsoil clay content (0–30 cm)', 'SoilGrids 2.0; grey = no soil data (water, rock, ice)', 'both'),
 ('sand', 0, 90, 10, 60, plt.get_cmap('YlOrBr'), 'Sand (%)', 'Topsoil sand content (0–30 cm)', 'SoilGrids 2.0; grey = no soil data (water, rock, ice)', 'both'),
]
WC_CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
WC_NAMES = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare / sparse',
            'Snow and ice', 'Water', 'Wetland', 'Moss and lichen']
WC_COLORS = ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000', '#b4b4b4', '#f0f0f0', '#0064c8', '#0096a0', '#fae6a0']
CLASS_NAMES = ['Very low', 'Low', 'Moderate', 'High', 'Very high']
CLASS_COLORS = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']

def layer(key):
    for f in F:
        if f[0] == key:
            v = decode(key, f[1], f[2])
            if key == 'aspect': v[v < 0] = np.nan   # flat
            if key in ('clay', 'sand'): v[v <= 0.3] = np.nan
            return v

def factor_map(i, f, num):
    key, lo, hi, vmin, vmax, cmap, lab, title, src, ext = f
    v = layer(key)
    norm = TwoSlopeNorm(0, vmin, vmax) if key.endswith('curv') else Normalize(vmin, vmax)
    fig, ax = new_map()
    ax.imshow(compose(v, cmap, norm, 0.3 if key not in ('aspect',) else 0.15), extent=EXTENT, origin='upper', interpolation='nearest', zorder=1)
    boundaries(ax); graticule(ax); furniture(ax)
    ax.set_title(f'{title} — Kyrgyzstan', loc='left')
    cbar(fig, cmap, norm, lab, extend=ext)
    credit(fig, f'Source: {src}. {CRS_NOTE}.')
    fn = f'out/map_{num:02d}_{key}.png'; fig.savefig(fn, dpi=220, bbox_inches='tight', pad_inches=0.12); plt.close(fig); return fn

if __name__ == '__main__':
    n = 3
    for f in F:
        print(factor_map(0, f, n)); n += 1
    # land cover
    lc = snap('lulc', WC_CODES + [95])
    idx = np.full(lc.shape, np.nan)
    for k, c in enumerate(WC_CODES): idx[lc == c] = k
    cmap = ListedColormap(WC_COLORS); norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), 10)
    fig, ax = new_map()
    ax.imshow(compose(idx, cmap, norm, 0.2), extent=EXTENT, origin='upper', interpolation='nearest')
    boundaries(ax); graticule(ax); furniture(ax)
    ax.set_title('Land cover (ESA WorldCover 2021) — Kyrgyzstan', loc='left')
    present = [k for k in range(10) if np.sum(idx == k) > 50]
    legend_boxes(ax, [(WC_NAMES[k], WC_COLORS[k]) for k in present], 'Land cover (2021)')
    credit(fig, f'Source: ESA WorldCover v200 (10 m), majority class per 250 m cell. {CRS_NOTE}.')
    fig.savefig(f'out/map_{n:02d}_land_cover.png', dpi=220, bbox_inches='tight', pad_inches=0.12); plt.close(fig); print('lulc done')

def lithology_map(fn='out/map_18b_lithology.png'):
    li = decode('lithology'); idx = li - 1
    cmap = ListedColormap(LITH_COLORS); norm = BoundaryNorm(np.arange(-0.5, len(LITH_CODES) + 0.5, 1), len(LITH_CODES))
    fig, ax = new_map()
    ax.imshow(compose(idx, cmap, norm, 0.2), extent=EXTENT, origin='upper', interpolation='nearest')
    boundaries(ax); graticule(ax); furniture(ax)
    ax.set_title('Lithology (geological formation groups) — Kyrgyzstan', loc='left')
    legend_boxes(ax, [(n, c) for n, c in zip(LITH_NAMES, LITH_COLORS)], 'Formation group')
    credit(fig, f'Source: geological formation map, geonode.water.gov.kg (layer geonode:geology), 17 units merged into 11 groups. {CRS_NOTE}.')
    fig.savefig(fn, dpi=220, bbox_inches='tight', pad_inches=0.12); plt.close(fig); return fn
