import sys, io, requests, numpy as np, pandas as pd
from PIL import Image, ImageDraw
d = pd.read_csv(sys.argv[1]); d = d[d.tier.isin(["A", "B"])].sort_values(["tier", "ra"])
def ls(ra, dec):
    x = requests.get("https://www.legacysurvey.org/viewer/cutout.jpg", timeout=90,
                     params=dict(ra=ra, dec=dec, layer="ls-dr10", pixscale=0.25, size=160))
    im = Image.open(io.BytesIO(x.content)).convert("RGB")
    return None if np.asarray(im).std() < 3 else im
def ps1(ra, dec):
    t = requests.get("https://ps1images.stsci.edu/cgi-bin/ps1filenames.py", params=dict(ra=ra, dec=dec, filters="gri"), timeout=60).text
    fn = {l.split()[4]: l.split()[7] for l in t.splitlines()[1:] if l.strip()}
    x = requests.get("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi", timeout=90,
                     params=dict(ra=ra, dec=dec, size=160, format="jpg", red=fn["i"], green=fn["r"], blue=fn["g"]))
    return Image.open(io.BytesIO(x.content)).convert("RGB")
ims = []
for r in d.itertuples():
    im, src = ls(r.ra, r.dec), "LS DR10"
    if im is None:
        im, src = ps1(r.ra, r.dec), "PS1"
    im = im.resize((240, 240)); dr = ImageDraw.Draw(im)
    dr.ellipse([105, 105, 135, 135], outline=(0, 255, 0)); dr.text((4, 4), f"{r.tier} {r.source_id}", fill=(255, 255, 0))
    dr.text((4, 226), src, fill=(255, 255, 0)); ims.append(im)
W = 6; H = (len(ims) + W - 1) // W
c = Image.new("RGB", (240 * W, 240 * H)); [c.paste(im, ((i % W) * 240, (i // W) * 240)) for i, im in enumerate(ims)]
c.save(sys.argv[2], quality=90); print(len(ims))
