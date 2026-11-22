"""Merge dossiers from all run folders, re-check prior art with the current catalogue list, assign grades."""
import glob, sys, pandas as pd, numpy as np
from prior_art import prior_art
GRADE = {  # assigned by eye from the folded ZTF light curves; everything else PASS defaults to B
    "A": [1006101831461460224, 1006043282467183104, 991692486764480896, 3093565751039683584, 991466468405767936,
          369344097712308096, 369814482530214656, 3093934431033115136, 1006266689486589312, 891444690197542784, 2843905349438427008,
          1122774858575469696, 2277532032203943808, 3152117768877723776, 2277649950530447360, 3152168105894464896, 3152266301729883264],
    "C": [982588221248516480, 982616641047840000, 1894863674633099136, 882425842990765696, 991513339384706944, 2844004202405885696, 1894541345926085888, 3151964593164238336, 3151976378554519424]}
frames = []
for f in glob.glob("runs/*/dossier.csv"):
    d = pd.read_csv(f); d["run"] = f.split("/")[1]; frames.append(d)
D = pd.concat(frames, ignore_index=True).drop_duplicates("source_id")
import json, os
CACHE = "runs/prior_art_cache.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
from concurrent.futures import ThreadPoolExecutor
todo = [r for r in D.itertuples() if str(r.source_id) not in cache or "ERROR" in cache[str(r.source_id)]]
def job(r):
    return str(r.source_id), ";".join(sorted(prior_art(r.ra, r.dec, 5)))
with ThreadPoolExecutor(8) as ex:
    for k, v in ex.map(job, todo):
        cache[k] = v; json.dump(cache, open(CACHE, "w"))
hits = [cache[str(r.source_id)] for r in D.itertuples()]
D["catalogue_hits_5as"] = hits
def grade(r):
    if r.catalogue_hits_5as and "TESS_10k_unvetted" != r.catalogue_hits_5as:
        return "K"
    for g, ids in GRADE.items():
        if r.source_id in ids:
            return g
    return "B"
D["tier"] = [grade(r) for r in D.itertuples()]
D.to_csv(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/out/realsky_candidates.csv", index=False)
print(D.tier.value_counts().to_dict()); print(D[D.tier == "K"][["source_id", "catalogue_hits_5as"]])
