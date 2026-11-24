"""scope_index.py - compact index of every source the ZTF SCoPe catalogue (Healy et al. 2024, Zenodo 8410826) calls
variable, for prior-art checks. SCoPe lists every light curve in 40 ZTF fields with a period and class probabilities;
a source counts here when either model gives the variable/non-variable class (vnv) a probability of at least 0.5.
Streams one field file (up to 6.5 GB) without storing it.

usage: python scope_index.py <field> <out.csv.gz>
"""
import sys, io, requests, pandas as pd

URL = "https://zenodo.org/api/records/8410826/files/field_{}.csv/content"
KEEP = ["Gaia_EDR3___id", "period", "field", "filter", "vnv_dnn", "vnv_xgb"]
CLASSES = ["e", "ew", "ea", "eb", "wuma", "el", "rscvn", "bis", "blyr", "pnp", "puls", "dscu", "rrlyr", "rrab", "rrc",
           "ceph", "lpv", "srv", "yso", "agn", "cv", "dip", "fla", "sin"]

field, out = sys.argv[1], sys.argv[2]
parts = []
with requests.get(URL.format(field), stream=True, timeout=600) as r:
    r.raise_for_status(); r.raw.decode_content = True
    for chunk in pd.read_csv(r.raw, chunksize=500_000):
        v = chunk[(chunk.vnv_dnn >= 0.5) | (chunk.vnv_xgb >= 0.5)]
        if v.empty:
            continue
        sc = pd.DataFrame({c: v[[f"{c}_dnn", f"{c}_xgb"]].max(axis=1) for c in CLASSES if f"{c}_dnn" in v})
        top = sc.idxmax(axis=1); best = sc.max(axis=1)
        parts.append(v[KEEP].assign(top_class=top, top_prob=best.round(2)))
d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEEP + ["top_class", "top_prob"])
d.to_csv(out, index=False)
print(field, len(d), "variable rows")
