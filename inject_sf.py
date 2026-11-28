"""Injection-recovery for the selection function (paper checklist A6).

Plants synthetic sinusoids and detached-eclipse signals into real ZTF r-band light curves of Gaia-flagged and
Gaia-quiet stars that the detector did not flag, in the same partitions, and runs the full search pipeline on them
(pre-filter, features, frozen classifier, channels with alias cuts, independent-halves test). If the detector is
equally sensitive to both groups at fixed period, amplitude and G, the flagged fraction C compares like with like.
Run from a checkout of realsky-validation:
    python inject_sf.py --tiles 889,8157,3126 --per-group 600 --out inject_sf.csv
"""
import argparse, os, sys, time, json
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--tiles", required=True); ap.add_argument("--per-group", type=int, default=600)
ap.add_argument("--out", default="inject_sf.csv"); ap.add_argument("--cache", default="cache"); ap.add_argument("--scratch", default="scratch")
ap.add_argument("--model", default="results/realsky1/model"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seed", type=int, default=20261128)
a = ap.parse_args()
sys.path.insert(0, os.getcwd())
import joblib
from realsky import ztf_tile, gaia_tile, cheap_stats, extra_order_tiles, high_latitude_tiles, GAIA_TAP, GAIA_TAP_FALLBACK
from eulerattack import SHAPE_FEATURES, features
from varwise_gaia_hunt import temporal_consistency_lc, temporal_consistency_ecl
from multiprocessing import Pool
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sfrerun.py")).read()
exec(src[src.index("ALIASES ="):src.index("os.makedirs(a.out")])          # is_alias, gaia_all, match: identical code
pack = joblib.load(os.path.join(a.model, "frozen.joblib")); clf, thr = pack["model"], pack["threshold"]
va = pd.read_parquet(os.path.join(a.model, "val_features.parquet")); cut = float(np.quantile(va[va.label == 1].iqr_over_err, 0.05))
man = json.load(open(os.path.join(a.model, "manifest.json")))
alltiles = {str(t["id"]): t for t in man["search"] + extra_order_tiles(a.cache) + high_latitude_tiles(a.cache)}
rng = np.random.default_rng(a.seed)

def inject(t, m, kind, P, A, ph):
    x = ((t / P) + ph) % 1
    if kind == "sin":
        return m + A * np.sin(2 * np.pi * x)
    w = 0.04                                                       # eclipse: primary depth A, secondary A/2, width 4% of P
    d = np.where(np.abs(x - 0.25) < w / 2, A, 0.0) + np.where(np.abs(x - 0.75) < w / 2, A / 2, 0.0)
    return m + d

def run(job):
    t, m, e, kind, P, A, ph = job
    mm = inject(t, m, kind, P, A, ph)
    if cheap_stats(t, mm, e) < cut:
        return dict(pre=False, det_clf=False, det_free=False)
    try:
        f = features(pd.DataFrame(dict(mjd=t, mag=mm, magerr=e)))
    except Exception:
        f = None
    if not f or not (f["ls_pass"] or f["ecl_pass"]):
        return dict(pre=True, det_clf=False, det_free=False)
    ecl_ok = f["ecl_pass"] and not is_alias(float(f["ecl_period"]))
    s = float(clf.predict_proba(pd.DataFrame([f])[SHAPE_FEATURES])[:, 1][0])
    ecl = f["ecl_pass"] and not f["ls_pass"]; per = float(f["ecl_period"] if ecl else f["period"])
    lc = pd.DataFrame(dict(mjd=t, mag=mm, magerr=e))
    halves = bool((temporal_consistency_ecl if ecl else temporal_consistency_lc)(lc, per)["confirmed"])
    chan = f["ls_pass"] or ecl_ok
    ok_P = any(abs(per / (P * k) - 1) < 0.02 for k in (0.5, 1, 2))
    return dict(pre=True, det_clf=bool(s >= thr and chan and halves and ok_P), det_free=bool(chan and halves and ok_P), score=s, per=per)

DET = set(pd.read_csv(os.environ.get('DETECTED', 'freeze/sf_ids.csv')).source_id)    # already-detected stars are not used as bases
rows = []
for tid in a.tiles.split(","):
    tile = alltiles[tid]; z = ztf_tile(tile, a.scratch); g = gaia_tile(tile, a.cache); allg = gaia_all(tile, a.cache)
    tgt = g[g.ruwe < 1.4].copy(); tgt["group"] = np.where(tgt.vari_id.notna() | (tgt.phot_variable_flag == "VARIABLE"), "flagged", "quiet")
    tgt = tgt[~tgt.best_class_name.isin(["AGN", "GALAXY"])]
    mt = match(z, tgt, allg); mt = mt[mt.isolated & ~mt.shared_ztf & ~mt.source_id.isin(DET)]; zz = z.reset_index(drop=True)
    jobs, meta = [], []
    for grp in ("flagged", "quiet"):
        pool_ = mt[mt.group == grp]
        pick = pool_.sample(min(len(pool_), a.per_group), random_state=int(rng.integers(1e9)))
        for r in pick.itertuples():
            t, m, e = zz.t.iat[r.zi], zz.m.iat[r.zi], zz.e.iat[r.zi]
            for kind in ("sin", "ecl"):
                P = float(np.exp(rng.uniform(np.log(0.15), np.log(100 if kind == "sin" else 30))))
                A = float(np.exp(rng.uniform(np.log(0.01), np.log(0.4))))
                jobs.append((t, m, e, kind, P, A, float(rng.uniform())))
                meta.append(dict(tile=tid, group=grp, source_id=int(r.source_id), G=r.phot_g_mean_mag, kind=kind, P=P, A=A))
    t0 = time.time()
    with Pool(a.workers) as p:
        res = p.map(run, jobs, chunksize=10)
    rows += [dict(mi, **ri) for mi, ri in zip(meta, res)]
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(tid, len(jobs), f"{time.time() - t0:.0f}s", flush=True)
    try: os.remove(os.path.join(a.scratch, f"ztf_{tid}.pkl"))
    except OSError: pass
