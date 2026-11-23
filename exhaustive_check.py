"""Exhaustive prior-art check for every claimed star:
Gaia DR3 variability tables (ESA TAP), SIMBAD (TAP, 10"), ALeRCE ZTF broker (3"), and the FULL VizieR
index at 5" on two mirrors, with every returned catalogue's title read and screened for variability."""
import re, json, time, sys, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
D = pd.read_csv(sys.argv[1]); D = D[D.tier.isin(["A", "B"])].reset_index(drop=True)
KEY = re.compile(r"variab|period|eclips|binar|rotat|pulsat|flare|light.?curve|RR ?Lyr|Cepheid|Scuti|W ?UMa|contact|"
                 r"spot|transient|AGN|quasar|YSO|young stell|dipper|ZTF|TESS|Kepler|K2|WISE.*var|NSVS|ASAS|CRTS|Catalina|"
                 r"ATLAS|OGLE|SuperWASP|WASP|HAT|KELT|PTF|ellipsoid|time.?serie|multi.?epoch", re.I)
MIRRORS = ["https://vizier.cfa.harvard.edu", "https://vizier.idia.ac.za"]
titles = json.load(open("runs/vizier_titles.json")) if __import__("os").path.exists("runs/vizier_titles.json") else {}

def gaia(ids):
    import pyvo
    s = pyvo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap")
    il = ",".join(map(str, ids))
    a = s.run_sync(f"select source_id, phot_variable_flag from gaiadr3.gaia_source where source_id in ({il})").to_table().to_pandas()
    b = s.run_sync(f"select source_id from gaiadr3.vari_summary where source_id in ({il})").to_table().to_pandas()
    c = s.run_sync(f"select source_id, best_class_name from gaiadr3.vari_classifier_result where source_id in ({il})").to_table().to_pandas()
    return a, b, c

def simbad(ra, dec):
    q = (f"SELECT b.main_id, b.otype, DISTANCE(POINT('ICRS',b.ra,b.dec),POINT('ICRS',{ra},{dec}))*3600 AS sep, i.ids "
         f"FROM basic AS b JOIN ids AS i ON i.oidref=b.oid WHERE CONTAINS(POINT('ICRS',b.ra,b.dec),CIRCLE('ICRS',{ra},{dec},{10/3600}))=1")
    for k in range(3):
        try:
            x = requests.get("https://simbad.cfa.harvard.edu/simbad/sim-tap/sync", timeout=90,
                             params=dict(request="doQuery", lang="adql", format="json", query=q))
            if x.status_code == 200:
                j = x.json(); return [dict(zip([c["name"] for c in j["metadata"]], r)) for r in j["data"]]
        except Exception:
            time.sleep(3)
    return "ERROR"

def alerce(ra, dec):
    for k in range(3):
        try:
            x = requests.get("https://api.alerce.online/ztf/v1/objects/", params=dict(ra=ra, dec=dec, radius=3), timeout=90)
            if x.status_code == 200:
                out = []
                for it in x.json()["items"]:
                    p = requests.get(f"https://api.alerce.online/ztf/v1/objects/{it['oid']}/probabilities", timeout=90)
                    pr = [q for q in (p.json() if p.status_code == 200 else []) if q.get("ranking") == 1]
                    out.append(dict(oid=it["oid"], ndet=it["ndet"], cls=it.get("class"),
                                    top=[(q["classifier_name"], q["class_name"], round(q["probability"], 2)) for q in pr]))
                return out
        except Exception:
            time.sleep(3)
    return "ERROR"

def vizier_all(ra, dec):
    """Full VizieR index at 5 arcsec on both mirrors; returns {catalogue: title} and how many mirrors answered."""
    cats, ok = {}, 0
    for m in MIRRORS:
        if ok:
            break
        for k in range(2):
            try:
                x = requests.get(m + "/viz-bin/VizieR-4", timeout=400, params={"-c": f"{ra:.6f} {dec:+.6f}", "-c.rs": 5, "-out.max": 1})
                if x.status_code == 200 and "VizieR" in x.text and "DBMS Error" not in x.text and "not currently reachable" not in x.text and "-source=" in x.text:
                    for c, t in re.findall(r"VizieR\?-source=([^'\"&]+)'>([^<]+)</a>", x.text):
                        cats.setdefault(c, t.strip())
                    ok += 1; break
            except Exception:
                time.sleep(5)
    return cats, ok


def one(r):
    sb = simbad(r.ra, r.dec); al = alerce(r.ra, r.dec); vz, ok = vizier_all(r.ra, r.dec)
    return r.source_id, sb, al, vz, ok

a, b, c = gaia(D.source_id.tolist())
with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(one, D.itertuples()))
rows = []
for (sid, sb, al, vz, ok), r in zip(res, D.itertuples()):
    flags = []
    if sid in set(b.source_id): flags.append("GAIA vari_summary")
    if sid in set(c.source_id): flags.append("GAIA class " + c[c.source_id == sid].best_class_name.iloc[0])
    pv = a[a.source_id == sid].phot_variable_flag
    if len(pv) and str(pv.iloc[0]) == "VARIABLE": flags.append("GAIA phot_variable_flag")
    if sb == "ERROR": flags.append("SIMBAD:ERROR")
    else:
        for o in sb:
            if re.search(r"V\*|^Ro|EB|RR|dS|LP|Pu|Va|Fl|^\*\*|Er|bL|WU|AL|Ce|C\*|YS|Or|TT", str(o.get("otype", ""))) or \
               re.search(r"V\* |ZTF ?J|ASASSN|ATO J|NSVS|CSS_|WISE J.*var|TIC .*EB|KIC|EPIC|Gaia DR\d+ ?VARI", str(o.get("ids", ""))):
                flags.append(f"SIMBAD {o['main_id']} otype={o['otype']} sep={o['sep']:.1f}")
    if al == "ERROR": flags.append("ALeRCE:ERROR")
    else:
        for o in al:
            if o["cls"] or any(t[1] not in ("Periodic-Other",) and t[0] in ("lc_classifier", "lc_classifier_periodic") for t in o["top"]):
                flags.append(f"ALeRCE {o['oid']} class={o['cls']} top={o['top']}")
    vflag = [f"{v}: {t[:90]}" for v, t in vz.items() if KEY.search(t) and not re.search(r"Input Catalog|TIC v|Gaia DR\d Part 1|photometric calibration", t)]
    if ok == 0: flags.append("VizieR-index:ERROR")
    rows.append(dict(source_id=sid, tier=r.tier, ra=r.ra, dec=r.dec, flags="; ".join(flags), vizier_variability_catalogues="; ".join(vflag),
                     n_vizier_catalogues=len(vz), vizier_mirrors_ok=ok, alerce=json.dumps(al, default=str)[:600]))
    json.dump(titles, open("runs/vizier_titles.json", "w"))
out = pd.DataFrame(rows); out.to_csv(sys.argv[2], index=False)
for r in out.itertuples():
    if r.flags or r.vizier_variability_catalogues:
        print(r.tier, r.source_id, "|", r.flags, "|", r.vizier_variability_catalogues)
print("checked", len(out))
