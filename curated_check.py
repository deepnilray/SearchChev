import sys, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from prior_art import cone
CATS = {"Arevalo2026_ZTFclass": "J/A+A/705/A247", "Mowlavi2021_GaiaDR2_LAV": "J/A+A/648/A44", "Kounkel2022_TESSper": "J/AJ/164/137",
        "Chen2020_ZTF": "J/ApJS/249/18/table2", "Chen2020_susp": "J/ApJS/249/18/table3", "VSX": "B/vsx/vsx",
        "ASASSN": "II/366/catv2021", "ATLAS": "J/AJ/156/241/table4", "GaiaDR3_class": "I/358/vclassre",
        "GaiaDR3_rot": "I/358/vrm", "GaiaDR3_EB": "I/358/veb", "WISE_Chen2018": "J/ApJS/237/28", "CRTS_Drake2014": "J/ApJS/213/9",
        "TESS_EB_Prsa": "J/ApJS/258/16", "TESS10k_EB": "J/ApJS/279/50/table3", "TESS10k_unvetted": "J/ApJS/279/50/table2",
        "Delfini2025_YSO": "J/A+A/699/A145"}
D = pd.read_csv(sys.argv[1]); D = D[D.tier.isin(["A", "B"])]
def job(r):
    hits = []
    for k, s in CATS.items():
        try:
            if len(cone(s, r.ra, r.dec, 5)): hits.append(k)
        except Exception:
            hits.append(k + ":ERR")
    return r.source_id, r.tier, hits
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(job, D.itertuples()))
pd.DataFrame(res, columns=["source_id", "tier", "hits"]).to_csv(sys.argv[2], index=False)
for s, t, h in res:
    if h: print(t, s, h)
print("checked", len(res))
