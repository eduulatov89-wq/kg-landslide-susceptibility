import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(figsize=(12, 7.6)); ax.set_xlim(0, 120); ax.set_ylim(0, 76); ax.axis('off')
def box(x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=1.1))
    ax.text(x + 1.2, y + h - 1.4, title, fontsize=9.5, fontweight='bold', va='top', color='#1b1b1b')
    ax.text(x + 1.2, y + h - 4.6, '\n'.join(lines), fontsize=7.8, va='top', color='#333', linespacing=1.45)
def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#555', lw=1.3, mutation_scale=13))
IN, PR, MO, OU = ('#eef4fb', '#6b9bd1'), ('#f3f3f3', '#8c8c8c'), ('#fdf0e6', '#e08a4a'), ('#fbeaea', '#c0392b')
ax.text(1, 74.5, 'Methodology: nationwide landslide susceptibility mapping of Kyrgyzstan (Google Earth Engine)', fontsize=12, fontweight='bold')
box(1, 38, 30, 33, '1  Input data', ['Copernicus GLO-30 DEM', 'MERIT Hydro (flow accumulation)', 'CHIRPS v2.0 rainfall 1991–2020',
    'ESA WorldCover 2021', 'GRIP4 roads', 'MODIS MOD13Q1 NDVI 2019–23', 'SoilGrids 2.0 clay / sand', 'Compiled landslide inventory (5,392 pts)',
    'Tien Shan active faults; geological map', 'Admin. boundaries; main lakes masked'], *IN)
box(36, 50, 38, 21, '2  Factor stack (250 m, UTM 43N)', ['Terrain: elevation, slope, aspect, plan & profile', '   curvature, local relief, TWI',
    'Climate: annual & wettest-quarter rain', 'Land: land cover, lithology, NDVI, clay, sand', 'Proximity: roads, rivers, active faults', '→ 18 candidate factors (full precision)'], *PR)
box(36, 30, 38, 15, '3  Sample design', ['5,143 landslide cells (one per 250 m cell)', '5,143 random non-landslide cells ≥ 1 km away', 'Stratified 70 % training (7,200) / 30 % test (3,086)'], *PR)
box(79, 50, 40, 21, '4  Factor screening', ['Pearson correlation matrix, VIF', 'Dropped: relief (collinear with slope),', '   wettest-quarter rain (with annual rain)',
    'Kept 16 factors, all VIF ≤ 3.92', 'Bivariate frequency ratio for 11 factors'], *PR)
box(79, 30, 40, 15, '5  Machine learning (6 models)', ['RF, Gradient Boosting, XGBoost, SVM (RBF),', 'Deep neural network (MLP 128-64-32-16), KNN', 'Grid-search tuning, 5-fold CV; ensemble mean'], *MO)
box(36, 4, 38, 21, '7  Susceptibility products', ['Probability & class maps for each model', '5 classes (equal interval, 0.2 steps)', 'Class area vs landslide share, FR per class',
    'Class shares per oblast and raion', 'Ensemble, disagreement & consensus maps'], *OU)
box(79, 4, 40, 21, '6  Validation (test set)', ['Test-set AUC 0.943–0.961 (XGBoost, RF best)', 'Random 5 × 5 CV AUC 0.944–0.959', 'Spatial-block 5 × 5 CV AUC 0.907–0.924', 'DeLong, Friedman & Wilcoxon tests; SHAP'], *MO)
arrow(31, 60, 36, 60); arrow(31, 42, 36, 38); arrow(74, 60, 79, 60); arrow(55, 50, 55, 45); arrow(74, 38, 79, 38)
arrow(99, 30, 99, 25); arrow(79, 14, 74, 14); arrow(99, 50, 99, 45)
fig.savefig('out/chart_00_methodology_flowchart.png', dpi=220, bbox_inches='tight', pad_inches=0.15)
