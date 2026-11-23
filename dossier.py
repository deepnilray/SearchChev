"""Final prior-art re-check for every vetted PASS candidate (fixed catalogue cones, SIMBAD, all-VizieR)."""
import os as _os
OUT = _os.environ.get("OUT", "runs/realsky1")
import glob, json, pandas as pd
from prior_art import prior_art, all_vizier
from vet import simbad
V = pd.concat([pd.DataFrame(columns=["source_id","verdict"])] + [pd.read_csv(f).assign(dir=f.rsplit("/", 2)[-2]) for f in glob.glob(OUT + "/vet_t*/vetting.csv")])
V = V[V.verdict == "PASS"].drop_duplicates("source_id")
# stars in one tile sharing a period (x0.5, x1, x2 within 1%) point to a shared systematic
def _shares(r):
    others = V[(V.dir == r.dir) & (V.source_id != r.source_id)]
    return any(abs(r.period_refined / (q * f) - 1) < 0.002 for q in others.period_refined for f in (0.5, 1, 2))
if len(V):
    V["shares_period_in_tile"] = [_shares(r) for r in V.itertuples()]
    if "auto_grade" in V:
        V.loc[V.shares_period_in_tile, "auto_flags"] = V.loc[V.shares_period_in_tile, "auto_flags"].fillna("").astype(str) + "; shares period with another star in this tile"
        V.loc[V.shares_period_in_tile & V.auto_grade.isin(["A", "B"]), "auto_grade"] = "C"
import sys as _s
if not _os.path.exists(OUT + "/search_hits.csv"): _s.exit(0)
H = pd.read_csv(OUT + "/search_hits.csv").drop_duplicates("source_id")
V = V.drop(columns=[c for c in ("bp_rp", "amp", "amp_over_err", "channel", "score", "n", "span", "tile") if c in V]).merge(H[["source_id", "bp_rp", "amp", "amp_over_err", "channel", "score", "n", "span", "tile"]], on="source_id", how="left")
import os, sys
P = sys.argv[1] if len(sys.argv) > 1 else OUT + "/dossier.csv"
done = set(pd.read_csv(P).source_id) if os.path.exists(P) else set()
V = V[~V.source_id.isin(done)]
out = []
for r in V.itertuples():
    pa = prior_art(r.ra, r.dec, 5); nb = prior_art(r.ra, r.dec, 30); sb = simbad(r.ra, r.dec, 10)
    av = []
    nb_per = []
    for k, v in nb.items():
        if isinstance(v, dict):
            for key in ("Period", "Per", "period", "fp-lngfitper", "ls-Pday"):
                try:
                    p2 = float(v.get(key))
                    if p2 > 0:
                        nb_per.append(p2)
                except (TypeError, ValueError):
                    pass
    P0 = float(r.period_refined)
    leak = any(abs(q / (P0 * f) - 1) < 0.01 for q in nb_per for f in (0.5, 1, 2))
    out.append(dict(r._asdict(), catalogue_hits_5as=";".join(sorted(pa)), catalogue_hits_30as=";".join(sorted(nb)),
                    simbad_10as=sb, other_vizier_5as=";".join(av),
                    neighbour_periods=";".join(f"{q:.6g}" for q in nb_per), neighbour_period_match=leak))
    print(r.source_id, sorted(pa), sorted(nb), "SIMBAD" if sb else "-", av, flush=True)
    pd.DataFrame([out[-1]]).drop(columns=["Index"]).to_csv(P, mode="a", header=not os.path.exists(P), index=False)
