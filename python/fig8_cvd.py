import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, matplotlib
from mapkit import *
from make_maps import CLASS_NAMES, CLASS_COLORS
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Patch

P = np.load('data/pop2025_250m.npz'); WP = P['wp']
GR = np.load('data/ml_prob_grids.npz'); p = GR['RF']; v = ~np.isnan(p)
cls = np.where(v, np.clip(np.floor(np.nan_to_num(p) * 5), 0, 4), -1)
hv = v & (cls >= 3)

# ---- map: population living in High / Very high cells
cmap = plt.get_cmap('PuBu'); cmap = ListedColormap(cmap(np.linspace(0.5, 1, 256))); norm = LogNorm(1, 3000)
rgb = base_rgb()
s = (0.75 + 0.25 * np.clip(HS * 1.25, 0, 1))[..., None]
land = MASK & ~hv
rgb[land] = (np.array([0.97, 0.97, 0.95]) * s[land])
e = hv & (WP < 1); rgb[e] = (np.array([0.99, 0.83, 0.62]) * s[e])
q = hv & (WP >= 1); rgb[q] = cmap(norm(np.clip(WP[q], 1, 3000)))[:, :3] * s[q]
fig, ax = new_map()
ax.imshow(rgb, extent=EXTENT, origin='upper', interpolation='nearest')
boundaries(ax, lw_obl=0.6, labels=True); graticule(ax); furniture(ax)
legend_boxes(ax, [('High / Very high, no resident population', '#fdd49e'), ('Very low to Moderate susceptibility', '#f5f5f1')], 'RF susceptibility')
cbar(fig, cmap, norm, 'Residents per 250 m cell (log scale)', ticks=[1, 10, 100, 1000], extend='max')
fig.savefig('out/cvd/map_22_population_exposure.png', dpi=300, bbox_inches='tight', pad_inches=0.12); plt.close(fig); print('map ok')
