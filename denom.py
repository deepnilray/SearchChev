"""Selection-function denominator (pre-registered in PLAN_26NOV.md, Amendment 2, 24 Nov 2026).

Runs the frozen SearchChev detector, unchanged, on Gaia DR3 stars that Gaia DID flag as variable, in a seeded
random 10% of the sky tiles the search has completed. For every flagged star with a ZTF r-band light curve it records
the detector score and whether the full search rule (score >= threshold, periodic or eclipse channel passes, period
confirmed early vs late) fires. Comparing these detections with the search hits among Gaia-quiet stars in the same
tiles gives, per bin of period, amplitude and G, the fraction of ZTF-detectable variables that Gaia DR3 flagged.

Run from a checkout of the realsky-validation branch (same realsky.py the search uses):
    python denom.py --out work/denom<K> --shard K --nshards N --tiles tiles_sample.json --hours 5
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--cache", default="cache"); ap.add_argument("--scratch", default="scratch")
ap.add_argument("--model", default="results/realsky1/model"); ap.add_argument("--tiles", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--hours", type=float, default=5.0); ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
sys.path.insert(0, os.getcwd())
import joblib
from realsky import ztf_tile, gaia_tile, crossmatch, cheap_stats, _feat, extra_order_tiles, high_latitude_tiles
from eulerattack import SHAPE_FEATURES
from varwise_gaia_hunt import temporal_consistency_lc, temporal_consistency_ecl
from multiprocessing import Pool

os.makedirs(a.out, exist_ok=True); os.makedirs(a.cache, exist_ok=True); os.makedirs(a.scratch, exist_ok=True)
pack = joblib.load(os.path.join(a.model, "frozen.joblib")); clf, thr = pack["model"], pack["threshold"]
va = pd.read_parquet(os.path.join(a.model, "val_features.parquet"))
cut = float(np.quantile(va[va.label == 1].iqr_over_err, 0.05))       # identical pre-filter to the search
man = json.load(open(os.path.join(a.model, "manifest.json")))
alltiles = {str(t["id"]): t for t in man["search"] + extra_order_tiles(a.cache) + high_latitude_tiles(a.cache)}
want = [str(t) for t in json.load(open(a.tiles)) if str(t) in alltiles]
want = [w for i, w in enumerate(want) if i % a.nshards == a.shard]
st_p = os.path.join(a.out, "denom_state.json"); st = json.load(open(st_p)) if os.path.exists(st_p) else dict(done=[])
out_p = os.path.join(a.out, "denom_rows.csv"); t0 = time.time()
for tid in want:
    if tid in st["done"] or time.time() - t0 > a.hours * 3600:
        continue
    tile = alltiles[tid]
    try:
        z = ztf_tile(tile, a.scratch); g = gaia_tile(tile, a.cache)
    except Exception as ex:
        att = st.setdefault("attempts", {}); att[tid] = att.get(tid, 0) + 1
        if att[tid] >= 2:
            st["done"].append(tid); st.setdefault("skipped", []).append(tid)
        json.dump(st, open(st_p, "w")); print(f"denom tile {tid}: data error {ex!r}", flush=True); continue
    flagged = g[(g.vari_id.notna() | (g.phot_variable_flag == "VARIABLE")) & (g.ruwe < 1.4)]
    m = crossmatch(z, flagged); zz = z.reset_index(drop=True)
    io = np.array([cheap_stats(zz.t.iat[i], zz.m.iat[i], zz.e.iat[i]) for i in m.zi.values]) if len(m) else np.array([])
    keep = io >= cut if len(m) else np.array([], bool)
    jobs = [(zz.t.iat[i], zz.m.iat[i], zz.e.iat[i]) for i in m.zi.values[keep]]
    with Pool(a.workers) as pool:
        fs = pool.map(_feat, jobs, chunksize=20)
    rows = []
    for (k, r), f, zi in zip(m[keep].iterrows(), fs, m.zi.values[keep]):
        rec = dict(tile=tid, source_id=int(r.source_id), G=r.phot_g_mean_mag, bp_rp=r.bp_rp, gaia_class=r.best_class_name,
                   gaia_class_score=r.best_class_score, phot_g_n_obs=r.phot_g_n_obs, featurized=bool(f))
        if f:
            s = float(clf.predict_proba(pd.DataFrame([f])[SHAPE_FEATURES])[:, 1][0])
            ecl = f["ecl_pass"] and not f["ls_pass"]; per = float(f["ecl_period"] if ecl else f["period"])
            rule = s >= thr and (f["ls_pass"] or f["ecl_pass"])
            conf = False
            if rule:
                lc = pd.DataFrame(dict(mjd=zz.t.iat[zi], mag=zz.m.iat[zi], magerr=zz.e.iat[zi]))
                conf = bool((temporal_consistency_ecl if ecl else temporal_consistency_lc)(lc, per)["confirmed"])
            rec.update(score=s, period=per, channel="eclipse" if ecl else "periodic", amp=f["amp"],
                       amp_over_err=f["amp_over_err"], rule_fires=bool(rule), detected=bool(rule and conf))
        rows.append(rec)
    n_flag, n_match = len(flagged), len(m)
    if rows:
        pd.DataFrame(rows).to_csv(out_p, mode="a", header=not os.path.exists(out_p), index=False)
    st["done"].append(tid); st.setdefault("counts", {})[str(tid)] = dict(flagged=n_flag, ztf_matched=n_match,
                                                                         prefilter_kept=int(keep.sum()))
    json.dump(st, open(st_p, "w"))
    try: os.remove(os.path.join(a.scratch, f"ztf_{tid}.pkl"))
    except OSError: pass
    print(f"denom tile {tid}: {n_flag} Gaia-flagged, {n_match} with ZTF, {int(keep.sum())} featurized, "
          f"{sum(r.get('detected', False) for r in rows)} detected ({time.time() - t0:.0f}s)", flush=True)
