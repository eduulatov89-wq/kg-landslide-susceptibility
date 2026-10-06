"""Remove in-image titles and footer notes from the main figures (captions carry this text)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
R = {  # file: row bands (y0, y1) to blank
 'map_01_study_area': [(36, 81), (1526, 1554)],
 'chart_00_methodology_flowchart': [(33, 70)],
 'ml_01_roc_all_models': [(33, 63)],
 'ml_02b_cv_random_vs_spatial': [(22, 55), (978, 1002)],
 'shap_04_beeswarm_6models': [(2948, 2979)],
 'shap_03_dependence_RF': [(22, 53)],
 'ml_06_susceptibility_maps_6_models': [(33, 73)],
 'ml_10a_ensemble_mean_classes': [(33, 66), (1126, 1146)],
 'ml_10b_model_disagreement': [(33, 66), (1126, 1146)],
 'ml_10c_high_class_agreement': [(33, 66), (1126, 1146)],
 'map_22_population_exposure': [(26, 59), (1119, 1139)],
}
def clean(name, pad=24):
    im = Image.open(f'out/{name}.png').convert('RGB'); a = np.array(im)
    for y0, y1 in R[name]: a[max(0, y0 - 3):y1 + 3] = 255
    g = a.min(2) < 245; ys = np.where(g.any(1))[0]; xs = np.where(g.any(0))[0]
    a = a[max(0, ys[0] - pad):ys[-1] + pad + 1, max(0, xs[0] - pad):xs[-1] + pad + 1]
    out = Image.fromarray(a); out.save(f'out/nocap/{name}.png', dpi=(300, 300)); return out
C = {n: clean(n) for n in R}
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 52)
# Fig. 3: (a) ROC + (b) CV boxplots
a, b = C['ml_01_roc_all_models'], C['ml_02b_cv_random_vs_spatial']
h = 1250; a = a.resize((round(a.width * h / a.height), h), Image.LANCZOS); b = b.resize((round(b.width * h / b.height), h), Image.LANCZOS)
gap, top = 60, 70; c = Image.new('RGB', (a.width + b.width + gap, h + top), 'white'); c.paste(a, (0, top)); c.paste(b, (a.width + gap, top))
d = ImageDraw.Draw(c); d.text((10, 5), '(a)', fill='black', font=F); d.text((a.width + gap + 10, 5), '(b)', fill='black', font=F)
c.save('out/nocap/fig_main_roc_cv.png', dpi=(300, 300))
# Fig. 7: (a) ensemble, (b) disagreement, (c) consensus
ims = [C[k] for k in ['ml_10a_ensemble_mean_classes', 'ml_10b_model_disagreement', 'ml_10c_high_class_agreement']]
W = max(i.width for i in ims); pad, lab = 30, 80; H = sum(i.height for i in ims) + pad * 2
c = Image.new('RGB', (W + lab, H), 'white'); d = ImageDraw.Draw(c); f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 56); y = 0
for k, im in enumerate(ims): c.paste(im, (lab, y)); d.text((8, y + 8), f'({"abc"[k]})', fill='black', font=f); y += im.height + pad
c.save('out/nocap/fig7_ensemble_uncertainty_consensus.png', dpi=(300, 300))
for n in ['map_01_study_area', 'chart_00_methodology_flowchart', 'fig_main_roc_cv', 'shap_04_beeswarm_6models', 'shap_03_dependence_RF',
          'ml_06_susceptibility_maps_6_models', 'fig7_ensemble_uncertainty_consensus', 'map_22_population_exposure']:
    print(n, Image.open(f'out/nocap/{n}.png').size)
