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
cmap = plt.get_cmap('RdPu'); cmap = ListedColormap(cmap(np.linspace(0.3, 1, 256))); norm = LogNorm(1, 3000)
rgb = base_rgb()
s = (0.75 + 0.25 * np.clip(HS * 1.25, 0, 1))[..., None]
land = MASK & ~hv
rgb[land] = (np.array([0.97, 0.97, 0.95]) * s[land])
e = hv & (WP < 1); rgb[e] = (np.array([0.99, 0.83, 0.62]) * s[e])
q = hv & (WP >= 1); rgb[q] = cmap(norm(np.clip(WP[q], 1, 3000)))[:, :3] * s[q]
fig, ax = new_map()
ax.imshow(rgb, extent=EXTENT, origin='upper', interpolation='nearest')
boundaries(ax, lw_obl=0.6, labels=True); graticule(ax); furniture(ax)
ax.set_title('Population (2025) living in High and Very high landslide susceptibility cells — Kyrgyzstan', loc='left')
legend_boxes(ax, [('High / Very high, no resident population', '#fdd49e'), ('Very low to Moderate susceptibility', '#f5f5f1')], 'RF susceptibility')
cbar(fig, cmap, norm, 'Residents per 250 m cell (log scale)', ticks=[1, 10, 100, 1000], extend='max')
credit(fig, f'Population: WorldPop Global2 R2025A, 2025, constrained 100 m, summed to 250 m. Susceptibility: Random Forest, equal-interval classes. {CRS_NOTE}.')
fig.savefig('out/map_22_population_exposure.png', dpi=220, bbox_inches='tight', pad_inches=0.12); plt.close(fig); print('map ok')

# ---- chart: exposure by class, oblast and raion
C = pd.read_csv('out/table_pop_by_class.csv'); OB = pd.read_csv('out/table_pop_by_oblast.csv'); RT = pd.read_csv('out/table_pop_by_raion.csv')
fig = plt.figure(figsize=(14, 5.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.25], wspace=0.55)
ax = fig.add_subplot(gs[0]); x = np.arange(5)
ax.bar(x - 0.27, C.area_pct, 0.26, color=CLASS_COLORS, edgecolor='#777', lw=0.5)
ax.bar(x, C.pop_worldpop_pct, 0.26, color=CLASS_COLORS, edgecolor='#222', lw=0.5, hatch='////')
ax.bar(x + 0.27, C.pop_ghs_pct, 0.26, color=CLASS_COLORS, edgecolor='#222', lw=0.5, hatch='....')
for xi, a, b in zip(x, C.pop_worldpop_pct, C.pop_worldpop): ax.text(xi, a + 1.5, f'{b / 1000:,.0f}k', ha='center', fontsize=7, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right'); ax.set_ylabel('% of country total'); ax.set_ylim(0, 100)
ax.legend(handles=[Patch(fc='white', ec='#777', label='Area'), Patch(fc='white', ec='#222', hatch='////', label='Population, WorldPop'),
                   Patch(fc='white', ec='#222', hatch='....', label='Population, GHS-POP')], frameon=False, fontsize=7.5, loc='upper right')
ax.set_title('(a) Area and population by class', loc='left', fontweight='bold', fontsize=10)
ax = fig.add_subplot(gs[1]); o = OB.sort_values('pop_high_vhigh'); y = np.arange(len(o))
ax.barh(y, (o.pop_high_vhigh - o.pop_very_high) / 1000, color=CLASS_COLORS[3], edgecolor='#777', lw=0.4, label='High')
ax.barh(y, o.pop_very_high / 1000, left=(o.pop_high_vhigh - o.pop_very_high) / 1000, color=CLASS_COLORS[4], edgecolor='#777', lw=0.4, label='Very high')
for k, (a, pc) in enumerate(zip(o.pop_high_vhigh, o.pct_high_vhigh)): ax.text(a / 1000 + 3, k, f'{a / 1000:,.0f}k ({pc:.1f}%)', va='center', fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels(o.oblast); ax.set_xlabel('Residents (thousands)'); ax.margins(x=0.32); ax.legend(frameon=False, fontsize=8, loc='lower right')
ax.set_title('(b) Residents in High + Very high, by oblast', loc='left', fontweight='bold', fontsize=10)
ax = fig.add_subplot(gs[2]); r = RT.head(15).iloc[::-1]; y = np.arange(len(r))
ax.barh(y, (r.pop_high_vhigh - r.pop_very_high) / 1000, color=CLASS_COLORS[3], edgecolor='#777', lw=0.4)
ax.barh(y, r.pop_very_high / 1000, left=(r.pop_high_vhigh - r.pop_very_high) / 1000, color=CLASS_COLORS[4], edgecolor='#777', lw=0.4)
for k, (a, pc) in enumerate(zip(r.pop_high_vhigh, r.pct_high_vhigh)): ax.text(a / 1000 + 1.5, k, f'{a / 1000:,.1f}k ({pc:.0f}%)', va='center', fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels([f'{a} ({b})' for a, b in zip(r.raion, r.oblast)], fontsize=8); ax.set_xlabel('Residents (thousands)'); ax.margins(x=0.3)
ax.set_title('(c) The 15 raions with most exposed residents', loc='left', fontweight='bold', fontsize=10)
fig.text(0.01, -0.04, 'Population 2025: WorldPop Global2 R2025A (constrained, 100 m); GHS-POP R2023A 2025 epoch as a check in (a). Percentages in (b) and (c) are shares of each unit\'s population. RF equal-interval classes.', fontsize=7.5, color='#555')
fig.savefig('out/chart_15_population_exposure.png', dpi=220, bbox_inches='tight', pad_inches=0.15); plt.close(fig); print('chart ok')
