"""Mosaic the Earth Engine export tiles of each export into raw250/<name>.tif on the 3736 x 1844 grid."""
import json, base64, glob, os, re, sys, numpy as np, rasterio
from rasterio.transform import from_origin
# Earth Engine writes each export as tiles named <export>-<rowOffset>-<colOffset>.tif (Drive folder kg_landslide_gee).
# Put the downloaded tiles in raw250/tiles/; the mosaics are written to raw250/.
T = 'raw250/tiles/'; R = 'raw250/'
os.makedirs(R, exist_ok=True)
W, H = 3736, 1844
groups = {}
for p in glob.glob(T + '*.tif'):
    m = re.match(r'(kg250_\w+?)-(\d+)-(\d+)\.tif', os.path.basename(p)); groups.setdefault(m.group(1).replace('_s', ''), []).append((int(m.group(2)), int(m.group(3)), p))
for name, tiles in groups.items():
    out = name.replace('_s', '') if name.endswith('_s') else name
    with rasterio.open(tiles[0][2]) as r0: nb, prof, names = r0.count, r0.profile, r0.descriptions
    arr = np.full((nb, H, W), -32768, np.int16); cover = np.zeros((H, W), bool)
    for ro, co, p in tiles:
        with rasterio.open(p) as r:
            a = r.read(); arr[:, ro:ro + a.shape[1], co:co + a.shape[2]] = a; cover[ro:ro + a.shape[1], co:co + a.shape[2]] = True
    done = cover.all()
    print(name, len(tiles), 'tiles, complete' if done else f'INCOMPLETE {cover.mean():.2f}', names)
    if done and not os.path.exists(R + out + '.tif'):
        prof.update(width=W, height=H, transform=from_origin(3500, 4796000, 250, 250), compress='lzw')
        with rasterio.open(R + out + '.tif', 'w', **prof) as w:
            w.write(arr); w.descriptions = names
