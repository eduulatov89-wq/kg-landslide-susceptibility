"""Assemble colour-blind-safe main figures (Figs 1, 3, 8 changed; others unchanged from out/nocap)."""
import numpy as np, shutil, os
from PIL import Image, ImageDraw, ImageFont
os.makedirs('out/final', exist_ok=True)
def trim(src, pad=24):
    a = np.array(Image.open(src).convert('RGB')); g = a.min(2) < 245; ys = np.where(g.any(1))[0]; xs = np.where(g.any(0))[0]
    return Image.fromarray(a[max(0, ys[0] - pad):ys[-1] + pad + 1, max(0, xs[0] - pad):xs[-1] + pad + 1])
trim('out/cvd/map_01_study_area.png').save('out/final/map_01_study_area.png', dpi=(300, 300))
trim('out/cvd/map_22_population_exposure.png').save('out/final/map_22_population_exposure.png', dpi=(300, 300))
a, b = trim('out/cvd/ml_01_roc_all_models.png'), Image.open('out/nocap/ml_02b_cv_random_vs_spatial.png').convert('RGB')
h = 1250; a = a.resize((round(a.width * h / a.height), h), Image.LANCZOS); b = b.resize((round(b.width * h / b.height), h), Image.LANCZOS)
gap, top = 60, 70; c = Image.new('RGB', (a.width + b.width + gap, h + top), 'white'); c.paste(a, (0, top)); c.paste(b, (a.width + gap, top))
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 52); d = ImageDraw.Draw(c)
d.text((10, 5), '(a)', fill='black', font=F); d.text((a.width + gap + 10, 5), '(b)', fill='black', font=F)
c.save('out/final/fig_main_roc_cv.png', dpi=(300, 300))
for n in ['chart_00_methodology_flowchart', 'shap_04_beeswarm_6models', 'shap_03_dependence_RF', 'ml_06_susceptibility_maps_6_models', 'fig7_ensemble_uncertainty_consensus']:
    shutil.copy(f'out/nocap/{n}.png', f'out/final/{n}.png')
print(sorted(os.listdir('out/final')))
