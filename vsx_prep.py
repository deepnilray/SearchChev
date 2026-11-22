"""Build VSX submission rows (ZTF r band) for grade A/B candidates from their full ZTF DR23 light curves."""
import sys, numpy as np, pandas as pd
from vet import partition_rows
from realsky import order5_tiles
TYPES = {1006101831461460224: ("EA", 2), 1006043282467183104: ("EA", 2), 991692486764480896: ("EA", 2),
         1006266689486589312: ("EA", 2), 3093565751039683584: ("EW|HADS", 2), 991466468405767936: ("EW|HADS", 1),
         369344097712308096: ("EW|RRC", 1), 369814482530214656: ("ROT", 1), 3093934431033115136: ("VAR", 1),
         2843905349438427008: ("EA", 2), 891444690197542784: ("EW", 1),
         1122774858575469696: ("EB|EW", 2), 2277532032203943808: ("EA", 2), 3152117768877723776: ("EA|EB", 2),
         2277649950530447360: ("EA", 2), 3152168105894464896: ("EW|HADS", 2), 3152266301729883264: ("HADS|EW", 1)}
REM = {1006101831461460224: "equal eclipses; orbital period is half this if the secondary eclipse is undetected",
       1006043282467183104: "equal eclipses; orbital period is half this if the secondary eclipse is undetected",
       991692486764480896: "equal eclipses; orbital period is half this if the secondary eclipse is undetected",
       1006266689486589312: "primary 0.08, secondary 0.03 mag; Gaia DSCT variable 21 arcsec away",
       3093565751039683584: "EW at this period or HADS at half of it",
       991466468405767936: "EW at this period or HADS at half of it",
       369344097712308096: "RRc at this period or EW at twice it",
       2843905349438427008: "primary 0.45, secondary 0.12 mag (ZTF r)",
       891444690197542784: "unequal minima",
       1122774858575469696: "unequal minima 0.55 and 0.5 mag (ZTF r)",
       2277532032203943808: "primary 0.3, secondary 0.1 mag (ZTF r)",
       3152117768877723776: "primary 0.35, secondary 0.1 mag (ZTF r)",
       2277649950530447360: "primary 0.25 mag, weak secondary (ZTF r)",
       3152168105894464896: "EW at this period or HADS at half of it",
       3152266301729883264: "HADS at this period or EW at twice it"}
d = pd.read_csv(sys.argv[1]); d = d[d.tier.isin(["A", "B"])]
tiles = {t["id"]: t for t in order5_tiles("/home/claude/work/cache", 0, 90, -90)}
out = []
for tid, g in d.groupby("tile"):
    lcs = partition_rows(tiles[int(tid)], g.ra.values, g.dec.values)
    for i, r in enumerate(g.itertuples()):
        rows = [x for x in lcs[i] if x["band"] == "r" and len(x["t"]) >= 40]
        ref = max(rows, key=lambda x: len(x["t"]))
        t = np.concatenate([x["t"] for x in rows])
        m = np.concatenate([x["m"] - np.median(x["m"]) + np.median(ref["m"]) for x in rows])
        typ, mult = TYPES.get(r.source_id, ("ROT:" if r.period_refined > 2 else "VAR", 1))
        P = r.period_refined * mult
        ph = ((t - 58000.0) / P) % 1
        nb = 100; idx = (ph * nb).astype(int)
        med = np.array([np.median(m[idx == j]) if np.any(idx == j) else np.nan for j in range(nb)])
        j = int(np.nanargmax(med))                          # faintest bin = minimum light
        tmin = 58000.0 + (np.floor((np.median(t) - 58000.0) / P) + (j + 0.5) / nb) * P
        out.append(dict(name=f"Gaia DR3 {r.source_id}", tier=r.tier, ra=round(r.ra, 6), dec=round(r.dec, 6),
                        type=typ, max_r=round(float(np.nanpercentile(med, 2)), 2), min_r=round(float(np.nanmax(med)), 2),
                        passband="ZTF r", period_d=round(P, 6), epoch_min_HJD=round(tmin + 2400000.5, 4),
                        n_obs_r=len(t), baseline_d=round(float(np.ptp(t)), 0), G=round(r.G, 2),
                        remarks=REM.get(r.source_id, "")))
        print(out[-1]["name"], typ, P, flush=True)
pd.DataFrame(out).sort_values(["tier", "name"]).to_csv(sys.argv[2], index=False)
