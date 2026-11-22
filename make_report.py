import json, glob, shutil, os, pandas as pd, numpy as np
C = pd.read_csv("/home/claude/out/realsky_candidates.csv")
V = pd.read_csv("/home/claude/out/vsx_submission_draft.csv")
st = [json.load(open(f)) for f in glob.glob("runs/*/search_state.json")]
done = sorted({t for s in st for t in s["done"]})
scanned = sum(s["scanned"] for s in st); scored = sum(s["scored"] for s in st)
H = pd.concat([pd.read_csv(f) for f in glob.glob("runs/*/search_hits.csv")]).drop_duplicates("source_id")
nhit, nnew = len(H), int((H.status == "NEW_CANDIDATE").sum())
nvet = len(C); tiers = C.tier.value_counts().to_dict()
os.makedirs("/home/claude/out/realsky_figs", exist_ok=True)
A = C[C.tier == "A"]
for s in A.source_id:
    f = glob.glob(f"runs/*/vet_t*/{s}.png")[0]; shutil.copy(f, f"/home/claude/out/realsky_figs/{s}.png")
NOTES = {1006101831461460224: "detached EB; equal eclipses 0.40--0.47 mag; orbit $P$ or $2P$",
 1006043282467183104: "detached EB; eclipses 0.14--0.16 mag; orbit $P$ or $2P$",
 991692486764480896: "detached EB; narrow eclipses 0.15--0.2 mag; orbit $P$ or $2P$",
 2843905349438427008: "detached EB at $2P=0.750753$ d; primary 0.45, secondary 0.12 mag in $r$",
 3093565751039683584: "0.2 mag; HADS at $P$ or contact binary at $2P$",
 991466468405767936: "0.1 mag; contact binary at $P$ or HADS at $P/2$",
 891444690197542784: "contact binary, unequal minima, 0.12 mag",
 369344097712308096: "0.15--0.25 mag, asymmetric; RRc at $P$ or EW at $2P$",
 369814482530214656: "0.1 mag, asymmetric; spotted rotator",
 3093934431033115136: "0.1 mag modulation with a dip; EB or spotted star",
 1006266689486589312: "EB at $2P$, primary 0.08, secondary 0.03 mag; earlier only in an unvetted TESS list",
 1122774858575469696: "EB at $2P=0.215916$ d, minima 0.55 and 0.5 mag in $r$; near the contact-binary period limit",
 2277532032203943808: "EB at $2P=0.490102$ d; primary 0.3, secondary 0.1 mag",
 3152117768877723776: "EB at $2P=0.371979$ d; primary 0.35, secondary 0.1 mag",
 2277649950530447360: "EB at $2P=0.440968$ d; primary 0.25 mag, weak secondary",
 3152168105894464896: "0.2 mag; contact binary at $2P$ or HADS at $P$",
 3152266301729883264: "0.2 mag, asymmetric; HADS at $P$ or contact binary at $2P$"}
def f2(x, fmt): return "--" if pd.isna(x) else format(x, fmt)
def row(r): return (f"{r.source_id} & {r.ra:.5f} & {r.dec:+.5f} & {f2(r.G,'.2f')} & {f2(r.bp_rp,'.2f')} & "
                    f"{r.period_refined:.6f} & {f2(r.amp,'.2f')} & {int(r.n_agree)}/{int(r.n_lc)} & {r.bands_agree} \\\\")
order = [s for s in NOTES if s in set(A.source_id)]
rows = {t: "\n".join(row(r) for r in C[C.tier == t].sort_values("ra").itertuples()) for t in "BCK"}
rows["A"] = "\n".join(row(C[C.source_id == s].iloc[0]) for s in order)
notes = "\n".join(f"{s} & {NOTES[s]} \\\\" for s in order)
kn = "; ".join(f"{r.source_id} ({r.catalogue_hits_5as.replace('_', ' ')})" for r in C[C.tier == "K"].itertuples())
vsx = "\n".join(f"{r.name.replace('Gaia DR3 ', '')} & {r.type} & {r.max_r:.2f}--{r.min_r:.2f} & {r.period_d:.6f} & {r.epoch_min_HJD:.4f} \\\\"
                for r in V.sort_values(['tier', 'name']).itertuples())
figs = "\n".join(f"\\begin{{figure}}[p]\\centering\\includegraphics[width=0.92\\textwidth]{{realsky_figs/{s}.png}}"
                 f"\\caption{{Gaia DR3 {s}: every ZTF DR23 light curve within 2$''$, median-subtracted per field, folded on $P$ (left) and $2P$ (right); black line is the 50-bin median.}}\\end{{figure}}" for s in order)
T = open("report_template.tex").read()
for k, v in dict(NTILES=str(len(done)), SCANNED=f"{scanned:,}".replace(",", "{,}"), SCORED=f"{scored:,}".replace(",", "{,}"),
                 NHIT=str(nhit), NNEW=str(nnew), NPASS=str(nvet), NA=str(tiers.get("A", 0)), NB=str(tiers.get("B", 0)),
                 NC=str(tiers.get("C", 0)), NEB=str(sum(1 for s in order if NOTES[s].startswith(("detached EB", "EB at")))), NK=str(tiers.get("K", 0)), NAB=str(tiers.get("A", 0) + tiers.get("B", 0)),
                 TILELIST=", ".join(map(str, done)), ROWSA=rows["A"], ROWSB=rows["B"], ROWSC=rows["C"], ROWSK=rows["K"],
                 NOTES=notes, KNOWN=kn, VSXROWS=vsx, FIGS=figs).items():
    T = T.replace("@" + k + "@", v)
open("/home/claude/out/searchchev_realsky_report.tex", "w").write(T)
