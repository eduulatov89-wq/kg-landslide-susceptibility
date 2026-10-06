import os
import sys; sys.path.insert(0, '.')
from make_maps import *
import numpy as np, shapefile
from shapely.geometry import shape
from shapely.ops import unary_union
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MPoly
SHP = os.path.join(os.environ.get('KG_AUX', 'aux'), 'cca', 'inform_cca_admin1_gadm.shp')
r = shapefile.Reader(SHP)
by = {}
for rec, s in zip(r.records(), r.shapes()): by.setdefault(rec[0], []).append(shape(s.__geo_interface__).buffer(0))
CTRY = {k: unary_union(v) for k, v in by.items()}
NAME = {'KZ': 'Kazakhstan', 'UZ': 'Uzbekistan', 'TM': 'Turkmenistan', 'TJ': 'Tajikistan', 'KG': 'Kyrgyzstan'}

f = F[0]; v = layer('elevation'); norm = Normalize(500, 6500)
fig, ax = new_map()
ax.imshow(compose(v, hyps, norm, 0.4), extent=EXTENT, origin='upper', interpolation='nearest')
boundaries(ax, lw_obl=0.8, labels=True); graticule(ax); furniture(ax)
ax.scatter(LS[:, 0], LS[:, 1], s=3.5, c='#ffd400', edgecolors='#111', linewidths=0.25, zorder=6)
ax.set_title('Study area: Kyrgyzstan, administrative units and mapped landslides', loc='left')
cbar(fig, hyps, norm, 'Elevation (m a.s.l.)', extend='both', pos=(0.875, 0.36, 0.018, 0.48))
legend_items(ax, [Line2D([], [], marker='o', ls='', mfc='#ffd400', mec='#111', ms=5, label=f'Mapped landslide (n = {len(LS):,})'),
                  Line2D([], [], color='#3a3a3a', lw=0.8, label='Oblast boundary'),
                  Line2D([], [], color='#111', lw=1.2, label='State border'),
                  Patch(fc=WATER, ec=WATER_EDGE, lw=0.5, hatch='////', label='Lake / reservoir')])
# ---- Central Asia locator inset (upper left, over Kazakhstan, outside the study area)
ia = ax.inset_axes([0.004, 0.755, 0.15, 0.24])
ia.set_facecolor('#dbe9f3')
def draw(g, fc, ec, lw, z):
    for p in getattr(g, 'geoms', [g]):
        ia.add_patch(MPoly(np.array(p.exterior.coords), closed=True, fc=fc, ec=ec, lw=lw, zorder=z))
for k, g in CTRY.items():
    if k in ('AM', 'AZ', 'GE'): continue
    elif k == 'KG': draw(g, '#d7191c', '#5a0000', 0.6, 3)
    else: draw(g, '#f4f1e8', '#6b6b6b', 0.4, 2)
lab = {'KZ': (68.0, 48.5), 'UZ': (63.5, 42.3), 'TM': (59.0, 38.8), 'TJ': (70.8, 37.0), 'KG': (75.2, 44.4)}
for k, (x, y) in lab.items():
    ia.text(x, y, NAME[k], fontsize=4.6, ha='center', va='center', color='#5a0000' if k == 'KG' else '#333',
            fontweight='bold' if k == 'KG' else 'normal', zorder=5)
ia.set_xlim(51, 88); ia.set_ylim(35, 56); ia.set_aspect(1 / np.cos(np.radians(45)))
ia.set_xticks([]); ia.set_yticks([])
for s in ia.spines.values(): s.set_edgecolor('#444'); s.set_linewidth(0.7)
ia.set_zorder(25)
credit(fig, f'Sources: Copernicus GLO-30 DEM; compiled Central Asian landslide inventory (incl. Behling et al. 2014, 2016; Strom and Abdrakhmatov 2018); boundaries MES 2018 (HDX); inset: OCHA ROCCA / INFORM Central Asia admin 1 (GADM). {CRS_NOTE}.')
fig.savefig('out/map_01_study_area.png', dpi=300, bbox_inches='tight', pad_inches=0.12); plt.close(fig); print('ok')
