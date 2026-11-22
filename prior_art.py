"""Cone checks against published variable-star catalogues through the CfA VizieR mirror."""
import time, io, requests, pandas as pd
CFA = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"
CATS = {"VSX": "B/vsx/vsx", "ZTF_Chen2020": "J/ApJS/249/18/table2", "ZTF_Chen2020_susp": "J/ApJS/249/18/table3",
        "ASASSN": "II/366/catv2021", "ATLAS_Heinze2018": "J/AJ/156/241/table4",
        "Gaia_DR3_vari": "I/358/vclassre", "TESS_EB": "J/ApJS/258/16/tess-ebs",
        "TESS_10k_EB": "J/ApJS/279/50/table3", "TESS_10k_unvetted": "J/ApJS/279/50/table2"}

def cone(src, ra, dec, r_arcsec):
    for k in range(3):
        try:
            x = requests.get(CFA, params={"-source": src, "-c": f"{ra:.6f} {dec:+.6f}", "-c.rs": r_arcsec,
                                         "-out.max": 20, "-out.add": "_r", "-out.all": ""}, timeout=60)
            if x.status_code == 200 and "#RESOURCE=" in x.text and "QUERY_STATUS=ERROR" not in x.text:
                lines = [l for l in x.text.splitlines() if l and not l.startswith("#")]
                if len(lines) <= 3:
                    return pd.DataFrame()
                return pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[3:])), sep="\t")
        except Exception:
            pass
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"VizieR error for {src}")

def prior_art(ra, dec, r_arcsec=5):
    hits = {}
    for name, src in CATS.items():
        try:
            d = cone(src, ra, dec, r_arcsec)
            if len(d):
                hits[name] = d.iloc[0].to_dict()
        except Exception as ex:
            hits[name] = "ERROR"
    return hits


SURVEYS = {"I/220", "I/239", "I/243", "I/246", "I/252", "I/259", "I/284", "I/289", "I/297", "I/305", "I/311",
           "I/317", "I/319", "I/329", "I/337", "I/339", "I/340", "I/345", "I/347", "I/350", "I/352", "I/353",
           "I/354", "I/355", "I/356", "I/357", "I/360", "I/42", "II/246", "II/281", "II/292", "II/294", "II/311",
           "II/312", "II/319", "II/328", "II/335", "II/336", "II/349", "II/358", "II/363", "II/365", "II/367",
           "II/371", "II/379", "II/381", "II/389", "III/279", "III/283", "IV/38", "IV/39", "V/147", "V/154", "J/A"}


def all_vizier(ra, dec, r_arcsec=5):
    """Every VizieR catalogue with an entry within r; returns the ones that are not plain all-sky surveys."""
    import re
    for k in range(2):
        try:
            x = requests.get("https://vizier.cfa.harvard.edu/viz-bin/VizieR-4", timeout=150,
                             params={"-c": f"{ra:.6f} {dec:+.6f}", "-c.rs": r_arcsec, "-out.max": 3})
            if x.status_code == 200:
                ids = set(re.findall(r"\b(?:J/[A-Za-z+]+/\d+/\d+|(?:I|II|III|IV|V|VI|VII|VIII|IX|B)/\d+)", x.text))
                return sorted(ids - SURVEYS)
        except Exception:
            pass
        time.sleep(5)
    return ["ERROR"]
