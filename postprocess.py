"""Turn the finished real-sky search into a catalogue ready for vetting by eye, Zenodo and VSX.

Reads every shard dossier on the realsky-results branch (checked out at argv[1]), keeps each star once,
re-runs the prior-art check at the end of the search (catalogues such as VSX change while the run goes),
and makes sky-image contact sheets (Legacy Survey DR10, Pan-STARRS where DR10 has no data) so every
grade A and B star gets a look for blends, galaxies and artefacts.

usage: python postprocess.py <realsky-results checkout> <out dir>
"""
import glob, io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from PIL import Image, ImageDraw
from prior_art import prior_art

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
WEAK = {"GaiaDR2_LAV_Mowlavi2021", "TESS_10k_unvetted", "YSO_Delfini2025"}   # flags with no period or type

frames = []
for f in glob.glob(os.path.join(SRC, "shards", "*", "*", "dossier.csv")):
    d = pd.read_csv(f)
    if len(d):
        run, shard = f.split(os.sep)[-3:-1]
        d["run"], d["shard"] = run, shard
        frames.append(d)
D = pd.concat(frames, ignore_index=True)
D = D.sort_values("score", ascending=False, na_position="last").drop_duplicates("source_id").reset_index(drop=True)
D["tier"] = D["auto_grade"]
print("vetted stars:", len(D), D.tier.value_counts().to_dict(), flush=True)

states = [json.load(open(f)) for f in glob.glob(os.path.join(SRC, "shards", "*", "*", "search_state.json"))]
done = set().union(*[set(map(str, s["done"])) for s in states]) if states else set()
skipped = sorted(set().union(*[set(map(str, s.get("skipped", []))) for s in states])) if states else []


def recheck(r):
    for k in range(3):
        try:
            pa = prior_art(float(r.ra), float(r.dec), 5)
            err = any(v == "ERROR" for v in pa.values())
            if not err or k == 2:
                return r.source_id, ";".join(sorted(pa)), err
        except Exception:
            pass
        time.sleep(5)
    return r.source_id, "", True


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(recheck, D.itertuples()))
R = pd.DataFrame(res, columns=["source_id", "prior_art_final", "prior_art_error"])
D = D.merge(R, on="source_id", how="left")


def status(row):
    if row.prior_art_error:
        return "UNVERIFIED"
    hits = set(filter(None, str(row.prior_art_final or "").split(";")))
    return "KNOWN" if hits - WEAK else ("NEW_FIRST_PERIOD" if hits else "NEW")


D["status_final"] = D.apply(status, axis=1)
D.to_csv(os.path.join(OUT, "realsky_catalogue_all.csv"), index=False)
C = D[D.tier.isin(["A", "B"]) & D.status_final.isin(["NEW", "NEW_FIRST_PERIOD"])].sort_values(["tier", "ra"])
C.to_csv(os.path.join(OUT, "realsky_catalogue_new_AB.csv"), index=False)
print("new A/B:", len(C), C.tier.value_counts().to_dict(), flush=True)


def get(url, **p):
    for k in range(3):
        try:
            x = requests.get(url, params=p, timeout=90)
            if x.ok:
                return x
        except Exception:
            pass
        time.sleep(3)
    return None


def cutout(ra, dec):
    x = get("https://www.legacysurvey.org/viewer/cutout.jpg", ra=ra, dec=dec, layer="ls-dr10", pixscale=0.25, size=160)
    if x is not None:
        try:
            im = Image.open(io.BytesIO(x.content)).convert("RGB")
            if np.asarray(im).std() >= 3:
                return im, "LS DR10"
        except Exception:
            pass
    t = get("https://ps1images.stsci.edu/cgi-bin/ps1filenames.py", ra=ra, dec=dec, filters="gri")
    try:
        fn = {l.split()[4]: l.split()[7] for l in t.text.splitlines()[1:] if l.strip()}
        x = get("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi", ra=ra, dec=dec, size=160, format="jpg",
                red=fn["i"], green=fn["r"], blue=fn["g"])
        return Image.open(io.BytesIO(x.content)).convert("RGB"), "PS1"
    except Exception:
        return Image.new("RGB", (160, 160)), "no image"


rows = list(C.itertuples())
with ThreadPoolExecutor(6) as ex:
    ims = list(ex.map(lambda r: cutout(r.ra, r.dec), rows))
per = 36   # 6 x 6 stars per sheet, each 40" across with a 3.75" green ring on the target
for s in range(0, len(rows), per):
    sheet = Image.new("RGB", (240 * 6, 240 * 6))
    for i, (r, (im, src)) in enumerate(zip(rows[s:s + per], ims[s:s + per])):
        im = im.resize((240, 240)); dr = ImageDraw.Draw(im)
        dr.ellipse([105, 105, 135, 135], outline=(0, 255, 0))
        dr.text((4, 4), f"{r.tier} {r.source_id}", fill=(255, 255, 0))
        dr.text((4, 226), f"{src}  P={r.period:.5g} d", fill=(255, 255, 0))
        sheet.paste(im, ((i % 6) * 240, (i // 6) * 240))
    sheet.save(os.path.join(OUT, "sheets", f"sheet_{s // per + 1:03d}.jpg"), quality=88)

summary = dict(tiles_done=len(done), tiles_skipped=skipped, vetted=len(D),
               vetted_by_grade=D.tier.value_counts().to_dict(),
               status=D.status_final.value_counts().to_dict(),
               new_AB=len(C), new_AB_by_grade=C.tier.value_counts().to_dict(),
               sheets=(len(rows) + per - 1) // per, made_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
