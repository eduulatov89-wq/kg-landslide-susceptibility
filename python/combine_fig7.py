from PIL import Image, ImageDraw, ImageFont
fs = ['out/ml_10a_ensemble_mean_classes.png', 'out/ml_10b_model_disagreement.png', 'out/ml_10c_high_class_agreement.png']
ims = [Image.open(f).convert('RGB') for f in fs]
W = max(i.width for i in ims); pad = 30; lab = 80
H = sum(i.height for i in ims) + pad * 2
C = Image.new('RGB', (W + lab, H), 'white'); d = ImageDraw.Draw(C)
f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 56)
y = 0
for k, im in enumerate(ims):
    C.paste(im, (lab, y)); d.text((8, y + 8), f'({"abc"[k]})', fill='black', font=f); y += im.height + pad
C.save('out/fig7_ensemble_uncertainty_consensus.png', dpi=(220, 220)); print(C.size)
