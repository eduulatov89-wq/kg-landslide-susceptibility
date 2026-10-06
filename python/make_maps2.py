import sys; sys.path.insert(0, '.')
from make_maps import *
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
S = pd.read_csv('data/samples.csv')
PRES = S[S.cls == 1]; ABS = S[S.cls == 0]

# 01 study area
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
credit(fig, f'Sources: Copernicus GLO-30 DEM; compiled Central Asian landslide inventory (landslides1; incl. Behling et al. 2014/2016, Strom); boundaries MES 2018 (HDX). {CRS_NOTE}.')
fig.savefig('out/map_01_study_area.png', dpi=220, bbox_inches='tight', pad_inches=0.12); plt.close(fig)

# 02 samples
fig, ax = new_map()
ax.imshow(base_rgb() * 0 + np.where(MASK[..., None], (0.93 + 0.07 * HS[..., None]) * np.array([1, 1, 1]), base_rgb()), extent=EXTENT, origin='upper')
boundaries(ax); graticule(ax); furniture(ax)
tr = S.rand < 0.7
for sub, mk, lab in [(ABS[ABS.rand < 0.7], 'o', None)]: pass
ax.scatter(ABS.x[ABS.rand < 0.7], ABS.y[ABS.rand < 0.7], s=1.6, marker='o', c='#2c7fb8', lw=0, zorder=6)
ax.scatter(ABS.x[ABS.rand >= 0.7], ABS.y[ABS.rand >= 0.7], s=2.5, marker='o', facecolors='none', edgecolors='#2c7fb8', lw=0.4, zorder=6)
ax.scatter(PRES.x[PRES.rand < 0.7], PRES.y[PRES.rand < 0.7], s=2, marker='^', c='#d7301f', lw=0, zorder=7)
ax.scatter(PRES.x[PRES.rand >= 0.7], PRES.y[PRES.rand >= 0.7], s=3, marker='^', facecolors='none', edgecolors='#d7301f', lw=0.4, zorder=7)
ax.set_title('Model sample: landslide and non-landslide cells, training and test split', loc='left')
n = lambda d, t: int(((d.rand < 0.7) if t else (d.rand >= 0.7)).sum())
legend_items(ax, [Line2D([], [], marker='^', ls='', mfc='#d7301f', mec='#d7301f', label=f'Landslide, training ({n(PRES,1)})'),
                    Line2D([], [], marker='^', ls='', mfc='none', mec='#d7301f', label=f'Landslide, test ({n(PRES,0)})'),
                    Line2D([], [], marker='o', ls='', mfc='#2c7fb8', mec='#2c7fb8', label=f'Non-landslide, training ({n(ABS,1)})'),
                    Line2D([], [], marker='o', ls='', mfc='none', mec='#2c7fb8', label=f'Non-landslide, test ({n(ABS,0)})')])
credit(fig, f'One point per 250 m model cell; non-landslide cells drawn at random ≥ 1 km from any mapped landslide; stratified 70/30 split. {CRS_NOTE}.')
fig.savefig('out/map_02_training_samples.png', dpi=220, bbox_inches='tight', pad_inches=0.12); plt.close(fig)

# 18 probability
GRd = np.load('data/ml_prob_grids.npz'); p = GRd['RF']
MT = pd.read_csv('data/ml_metrics.csv').set_index('model')
pc = plt.get_cmap('RdYlBu_r'); norm = Normalize(0, 1)
fig, ax = new_map()
ax.imshow(compose(p, pc, norm, 0.25), extent=EXTENT, origin='upper', interpolation='nearest')
boundaries(ax); graticule(ax); furniture(ax)
ax.set_title('Landslide susceptibility — Random Forest probability', loc='left')
cbar(fig, pc, norm, 'Probability of landslide occurrence')
credit(fig, f'Random Forest (500 trees, 15 factors), trained on {int((S.rand < 0.7).sum()):,} cells; test AUC = {MT.loc["RF", "AUC"]:.3f}. {CRS_NOTE}.')
fig.savefig('out/map_19_susceptibility_probability.png', dpi=220, bbox_inches='tight', pad_inches=0.12); plt.close(fig)

# 19 classes
cl = np.where(np.isnan(p), np.nan, np.clip(np.floor(p * 5), 0, 4))
cc = ListedColormap(CLASS_COLORS); cn = BoundaryNorm(np.arange(-0.5, 5.5), 5)
fig, ax = new_map()
ax.imshow(compose(cl, cc, cn, 0.22), extent=EXTENT, origin='upper', interpolation='nearest')
boundaries(ax, lw_obl=0.7, labels=True); graticule(ax); furniture(ax)
ax.scatter(LS[:, 0], LS[:, 1], s=1.2, c='#111', lw=0, zorder=6, alpha=0.7)
ax.set_title('Landslide susceptibility zonation (Random Forest) — Kyrgyzstan', loc='left')
cs = pd.read_csv('data/kg_ls_class_stats.csv')
items = [(f"{r['class']}  ({r.area_pct:.1f}% of area)", CLASS_COLORS[i]) for i, r in cs.iterrows()]
legend_boxes(ax, items, 'Susceptibility class', extra=[Line2D([], [], marker='o', ls='', mfc='#111', mec='#111', ms=3.5, label='Mapped landslide')])
credit(fig, f'Classes: equal-interval RF probability (0.2 steps). {CRS_NOTE}.')
fig.savefig('out/map_20_susceptibility_classes.png', dpi=220, bbox_inches='tight', pad_inches=0.12); plt.close(fig)

# 20 factor panel
keys = [f for f in F] 
fig, axs = plt.subplots(3, 6, figsize=(21, 7.6))
fig.subplots_adjust(left=0.005, right=0.995, top=0.955, bottom=0.01, hspace=0.30, wspace=0.06)
lc = snap('lulc', WC_CODES + [95]); lidx = np.full(lc.shape, np.nan)
for k, c in enumerate(WC_CODES): lidx[lc == c] = k
panels = [(f[0], f) for f in F] + [('lulc', None), ('lithology', None)]
lith_idx = decode('lithology') - 1
for i, (ax, (key, f)) in enumerate(zip(axs.flat, panels)):
    ax.set_xlim(X0, X1); ax.set_ylim(Y0 + 20000, Y1 - 20000); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for s_ in ax.spines.values(): s_.set_visible(False)
    if key == 'lithology':
        cmap, norm = ListedColormap(LITH_COLORS), BoundaryNorm(np.arange(-0.5, 11.5), 11); vv = lith_idx; title = 'Lithology (formation group)'
    elif f is None:
        cmap, norm = ListedColormap(WC_COLORS), BoundaryNorm(np.arange(-0.5, 10.5), 10); vv = lidx; title = 'Land cover (WorldCover)'
    else:
        vv = layer(key); cmap = f[5]; title = f[7].split(',')[0].split(' (')[0].replace(' 1991–2020', '')
        norm = TwoSlopeNorm(0, f[3], f[4]) if key.endswith('curv') else Normalize(f[3], f[4])
    ax.imshow(compose(vv, cmap, norm, 0.25), extent=EXTENT, origin='upper', interpolation='nearest')
    draw_lakes(ax, 0.3)
    for p_ in getattr(KG, 'geoms', [KG]): ax.plot(*p_.exterior.xy, color='#111', lw=0.6)
    ax.set_title(f'({chr(97+i)}) {title}', loc='left', fontsize=9.5)
    if f is not None:
        cax = ax.inset_axes([0.62, 0.06, 0.34, 0.035])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=6, length=2, pad=1); cb.outline.set_linewidth(0.3)
        cb.set_ticks([f[3], f[4]] if not key.endswith('curv') else [f[3], 0, f[4]])
        cb.ax.set_title(f[6].split('\n')[0], fontsize=6, pad=2)
ax = axs.flat[len(panels)]; ax.axis('off')
from matplotlib.patches import Patch
h1 = [Patch(fc=c, ec='#666', lw=0.4, label=n) for n, c in zip(WC_NAMES, WC_COLORS)]
h2 = [Patch(fc=c, ec='#666', lw=0.4, label=n) for n, c in zip(LITH_NAMES, LITH_COLORS)]
l1 = ax.legend(handles=h1, title='(p) Land cover', loc='upper left', bbox_to_anchor=(-0.02, 1.08), fontsize=6, title_fontsize=7, frameon=False, labelspacing=0.15, handlelength=1.2, handleheight=0.8)
ax.add_artist(l1); l1._legend_box.align = 'left'
l2 = ax.legend(handles=h2, title='(q) Lithology', loc='upper left', bbox_to_anchor=(0.40, 1.08), fontsize=6, title_fontsize=7, frameon=False, labelspacing=0.15, handlelength=1.2, handleheight=0.8)
l2._legend_box.align = 'left'
fig.savefig('out/map_21_factor_panel.png', dpi=200, bbox_inches='tight', pad_inches=0.12); plt.close(fig)
print('ok')
