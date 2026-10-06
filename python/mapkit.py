import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyArrow
from common import *

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11,
                     'axes.titleweight': 'bold', 'axes.edgecolor': '#333333'})
INK = '#222222'; MUTED = '#666666'
HS = load('hillshade').astype(float) / 255.0

def base_rgb():
    """Light-grey hillshade for the neighbouring countries."""
    g = 0.80 + 0.18 * HS
    rgb = np.dstack([g, g, g * 1.0])
    return rgb

def shaded(rgba, strength=0.35):
    s = (1 - strength) + strength * np.clip(HS * 1.25, 0, 1)
    out = rgba.copy(); out[..., :3] *= s[..., None]
    return out

def compose(values, cmap, norm, shade=0.35):
    rgb = base_rgb()
    rgba = cmap(norm(np.nan_to_num(values, nan=norm.vmin if hasattr(norm, 'vmin') and norm.vmin is not None else 0)))
    rgba = shaded(rgba, shade)
    m = MASK & ~np.isnan(values)
    rgb[m] = rgba[m, :3]
    nod = MASK & np.isnan(values)
    rgb[nod] = np.array([0.86, 0.86, 0.86])[None, :] * (0.8 + 0.2 * HS[nod, None])
    return rgb

def graticule(ax, lons=range(70, 81, 2), lats=range(39, 44, 1)):
    for lon in lons:
        la = np.linspace(38.5, 44, 200); x, y = T.transform(np.full_like(la, lon), la)
        ax.plot(x, y, color='#9a9a9a', lw=0.4, ls=(0, (3, 3)), zorder=3)
        k = np.argmin(np.abs(y - Y0))
        if X0 < x[k] < X1: ax.text(x[k], Y0 - 9000, f'{lon}°E', ha='center', va='top', fontsize=7.5, color=MUTED)
    for lat in lats:
        lo = np.linspace(68, 82, 300); x, y = T.transform(lo, np.full_like(lo, lat))
        ax.plot(x, y, color='#9a9a9a', lw=0.4, ls=(0, (3, 3)), zorder=3)
        k = np.argmin(np.abs(x - X0))
        if Y0 < y[k] < Y1: ax.text(X0 - 8000, y[k], f'{lat}°N', ha='right', va='center', fontsize=7.5, color=MUTED)

WATER, WATER_EDGE = '#e0e0e0', '#6b6b6b'
plt.rcParams['hatch.color'] = '#8a8a8a'; plt.rcParams['hatch.linewidth'] = 0.5
def draw_lakes(ax, lw=0.4):
    from matplotlib.patches import Polygon as MPoly
    for name, g in LAKES:
        for p in getattr(g, 'geoms', [g]):
            ax.add_patch(MPoly(np.array(p.exterior.coords), closed=True, fc=WATER, ec=WATER_EDGE, lw=lw, hatch='////', zorder=4.5))

def boundaries(ax, lw_obl=0.5, lw_kg=1.1, labels=False):
    draw_lakes(ax)
    for name, oid, g in ADM:
        for p in getattr(g, 'geoms', [g]):
            ax.plot(*p.exterior.xy, color='#3a3a3a', lw=lw_obl, zorder=4)
    for p in getattr(KG, 'geoms', [KG]):
        ax.plot(*p.exterior.xy, color='#111111', lw=lw_kg, zorder=5)
    if labels:
        for name, oid, g in ADM:
            if oid in (11, 21):
                c = g.centroid; ax.plot(c.x, c.y, 's', ms=4, mfc='white', mec='#111', zorder=7)
                ax.annotate(name.replace(' city', ''), (c.x, c.y), xytext=(6, 5), textcoords='offset points',
                            fontsize=7.5, fontweight='bold', color='#111', zorder=8,
                            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))
            else:
                c = g.representative_point()
                off = {'Chui': (0, -22000), 'Osh': (15000, -25000), 'Jalal-Abad': (-25000, 10000), 'Batken': (-10000, -5000), 'Talas': (0, -5000)}.get(name, (0, 0))
                ax.text(c.x + off[0], c.y + off[1], name, ha='center', va='center', fontsize=9, fontweight='bold',
                        color='#111', zorder=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

def furniture(ax):
    """Scale bar in the empty band south of Kyrgyzstan (74.6-77E) and north arrow in the NE corner,
    both placed outside the country so they never overlap mapped cells."""
    x0, y0 = X0 + 0.555 * (X1 - X0), Y0 + 20000          # ~75.3E: south border is >100 km north here
    for i, (a, b) in enumerate([(0, 50), (50, 100), (100, 150)]):
        ax.add_patch(Rectangle((x0 + a * 1000, y0), (b - a) * 1000, 5000, fc=INK if i % 2 == 0 else 'white', ec=INK, lw=0.7, zorder=9))
    for v in (0, 50, 100, 150):
        ax.text(x0 + v * 1000, y0 + 9000, str(v), ha='center', va='bottom', fontsize=7, color=INK, zorder=9)
    ax.text(x0 + 162000, y0 + 2500, 'km', ha='left', va='center', fontsize=7, color=INK, zorder=9)
    # north arrow: top-right corner, base ~70 km below the frame, clear of the eastern border (~42.4N)
    nx, ny = X1 - 30000, Y1 - 72000
    ax.add_patch(FancyArrow(nx, ny, 0, 40000, width=5000, head_width=18000, head_length=18000,
                            length_includes_head=True, fc=INK, ec=INK, zorder=9))
    ax.text(nx, ny + 42000, 'N', ha='center', va='bottom', fontsize=10, fontweight='bold', color=INK, zorder=9)

LEG_POS = dict(loc='lower right', bbox_to_anchor=(0.995, 0.015))   # SE corner (China), outside Kyrgyzstan

def new_map(figsize=(11, 5.55)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.06, 0.075, 0.80, 0.855])
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax

def credit(fig, text):
    fig.text(0.06, 0.035, text, fontsize=7, color=MUTED, ha='left', va='top')

def cbar(fig, cmap, norm, label, ticks=None, extend='neither', pos=(0.875, 0.16, 0.018, 0.68)):
    cax = fig.add_axes(pos)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, extend=extend, ticks=ticks)
    cb.set_label(label, fontsize=9); cb.outline.set_linewidth(0.5); cax.tick_params(labelsize=8)
    return cb

def legend_boxes(ax, items, title, extra=(), loc=None):
    from matplotlib.patches import Patch
    h = [Patch(fc=c, ec='#555', lw=0.4, label=l) for l, c in items] + list(extra) + [Patch(fc=WATER, ec=WATER_EDGE, lw=0.5, hatch='////', label='Lake / reservoir (excluded)')]
    pos = LEG_POS if loc is None else dict(loc=loc)
    lg = ax.legend(handles=h, title=title, frameon=True, fontsize=6.5, title_fontsize=7.5,
                   framealpha=0.95, edgecolor='#999999', borderpad=0.45, labelspacing=0.25, handlelength=1.5, handleheight=0.6, **pos)
    lg.get_title().set_fontweight('bold'); lg.set_zorder(20); return lg

def legend_items(ax, handles, loc=None):
    pos = LEG_POS if loc is None else dict(loc=loc)
    lg = ax.legend(handles=handles, frameon=True, fontsize=6.5, framealpha=0.95, edgecolor='#999999', borderpad=0.45,
                   labelspacing=0.25, **pos)
    lg.set_zorder(20); return lg

CRS_NOTE = 'WGS 84 / UTM zone 43N (EPSG:32643), 250 m grid'
