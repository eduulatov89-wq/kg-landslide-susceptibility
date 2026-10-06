"""Side-by-side normal / deuteranopia / protanopia / tritanopia previews (Machado et al. 2009, severity 100)."""
import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
from colorspacious import cspace_convert
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
def sim(a, t):
    s = cspace_convert(a / 255.0, {'name': 'sRGB1+CVD', 'cvd_type': t, 'severity': 100}, 'sRGB1')
    return (np.clip(s, 0, 1) * 255).astype(np.uint8)
src, dst, w = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 700
im = Image.open(src).convert('RGB'); im.thumbnail((w, w)); a = np.asarray(im).astype(float)
tiles = [('normal', a.astype(np.uint8))] + [(t, sim(a, t)) for t in ['deuteranomaly', 'protanomaly', 'tritanomaly']]
H, W = tiles[0][1].shape[:2]; C = Image.new('RGB', (W * 2, (H + 30) * 2), 'white'); d = ImageDraw.Draw(C)
for k, (n, t) in enumerate(tiles):
    x, y = (k % 2) * W, (k // 2) * (H + 30); C.paste(Image.fromarray(t), (x, y + 30)); d.text((x + 6, y + 3), n.replace('anomaly', 'anopia'), fill='black', font=F)
C.save(dst, quality=85)
