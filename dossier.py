"""Final prior-art re-check for every vetted PASS candidate (fixed catalogue cones, SIMBAD, all-VizieR)."""
import glob, json, pandas as pd
from prior_art import prior_art, all_vizier
from vet import simbad
V = pd.concat([pd.read_csv(f).assign(dir=f.rsplit("/", 2)[-2]) for f in glob.glob("runs/realsky1/vet_t*/vetting.csv")])
V = V[V.verdict == "PASS"].drop_duplicates("source_id")
H = pd.read_csv("runs/realsky1/search_hits.csv").drop_duplicates("source_id")
V = V.merge(H[["source_id", "G", "bp_rp", "amp", "amp_over_err", "channel", "score", "n", "span", "tile"]], on="source_id", how="left")
import os, sys
P = sys.argv[1] if len(sys.argv) > 1 else "runs/realsky1/dossier.csv"
done = set(pd.read_csv(P).source_id) if os.path.exists(P) else set()
V = V[~V.source_id.isin(done)]
out = []
for r in V.itertuples():
    pa = prior_art(r.ra, r.dec, 5); nb = prior_art(r.ra, r.dec, 30); sb = simbad(r.ra, r.dec, 10)
    av = []
    out.append(dict(r._asdict(), catalogue_hits_5as=";".join(sorted(pa)), catalogue_hits_30as=";".join(sorted(nb)),
                    simbad_10as=sb, other_vizier_5as=";".join(av)))
    print(r.source_id, sorted(pa), sorted(nb), "SIMBAD" if sb else "-", av, flush=True)
    pd.DataFrame([out[-1]]).drop(columns=["Index"]).to_csv(P, mode="a", header=not os.path.exists(P), index=False)
