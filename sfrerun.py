"""Selection-function rerun (paper checklist A1-A3, 28 Nov 2026).

Runs Gaia-DR3-flagged and Gaia-quiet stars through ONE identical pipeline in each of the 280 denominator partitions:
  * isolation: a star is dropped if ANY other Gaia DR3 source (all magnitudes, from the same partition) lies within 3",
    so both groups face the same neighbour list (the search and denom.py tested isolation within each group only);
  * a ZTF object matched to more than one target star is dropped for all of them (no light curve counted twice);
  * the alias cut (0.5, 1, 2 d, lunar month, year, within 2%; also at twice the period) is applied to the eclipse
    channel as well as the periodic channel;
  * every star records both detection flags: det_clf (frozen classifier score >= threshold, as in the search) and
    det_free (no classifier: periodic or eclipse gate plus the independent-halves test only);
  * Gaia classes are kept so AGN / galaxies can be removed at analysis time.
Run from a checkout of realsky-validation (same realsky.py, model and pre-filter as the search):
    python sfrerun.py --out work/sf<K> --shard K --nshards N --tiles denominator_tiles.json --hours 4.8
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
from realsky import ztf_tile, gaia_tile, cheap_stats, _feat, extra_order_tiles, high_latitude_tiles, GAIA_TAP, GAIA_TAP_FALLBACK
from eulerattack import SHAPE_FEATURES
from varwise_gaia_hunt import temporal_consistency_lc, temporal_consistency_ecl
from multiprocessing import Pool

ALIASES = (0.5, 1.0, 2.0, 29.53, 365.25)
def is_alias(p):
    return p is not None and np.isfinite(p) and any(abs(q / x - 1) < 0.02 for x in ALIASES for q in (p, 2 * p))

def gaia_all(tile, cache):
    """Every Gaia DR3 source in the partition, all magnitudes (positions only), for the isolation test."""
    p = os.path.join(cache, f"gaiaall_{tile['id']}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    import pyvo
    k, npix = tile.get("order", 5), tile.get("npix", tile["id"])
    span = 4 ** (12 - k) * 2 ** 35; lo = npix * span
    q = f"SELECT source_id, ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE source_id BETWEEN {lo} AND {lo + span - 1}"
    for t in range(6):
        try:
            d = pyvo.dal.TAPService(GAIA_TAP if t % 2 == 0 else GAIA_TAP_FALLBACK).run_sync(q, maxrec=5_000_000).to_table().to_pandas(); break
        except Exception:
            if t == 5: raise
            time.sleep(10 * (t + 1))
    d.to_parquet(p); return d

def match(z, tgt, allg, radius=1.5, iso=3.0):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    cz = SkyCoord(z.ra.values * u.deg, z.dec.values * u.deg); ct = SkyCoord(tgt.ra.values * u.deg, tgt.dec.values * u.deg)
    ca = SkyCoord(allg.ra.values * u.deg, allg.dec.values * u.deg)
    # isolation against every Gaia source: the nearest one is the star itself, so look at the second neighbour
    i2, s2, _ = ct.match_to_catalog_sky(ca, nthneighbor=2)
    i1, s1, _ = ct.match_to_catalog_sky(ca, nthneighbor=1)
    self_ok = s1.arcsec < 0.1
    isolated = np.where(self_ok, s2.arcsec > iso, s1.arcsec > iso)
    near = cz.search_around_sky(ct, radius * u.arcsec)
    best = {}
    zz = z.reset_index(drop=True)
    for ti, zi in zip(*near[:2]):
        if ti not in best or len(zz.t.iat[zi]) > len(zz.t.iat[best[ti]]):
            best[ti] = zi
    m = tgt.reset_index(drop=True).copy()
    m["zi"] = [best.get(i, -1) for i in range(len(m))]; m["isolated"] = isolated
    m = m[m.zi >= 0]
    dup = m.zi.duplicated(keep=False); m["shared_ztf"] = dup.values
    return m

os.makedirs(a.out, exist_ok=True); os.makedirs(a.cache, exist_ok=True); os.makedirs(a.scratch, exist_ok=True)
pack = joblib.load(os.path.join(a.model, "frozen.joblib")); clf, thr = pack["model"], pack["threshold"]
va = pd.read_parquet(os.path.join(a.model, "val_features.parquet"))
cut = float(np.quantile(va[va.label == 1].iqr_over_err, 0.05))
man = json.load(open(os.path.join(a.model, "manifest.json")))
alltiles = {str(t["id"]): t for t in man["search"] + extra_order_tiles(a.cache) + high_latitude_tiles(a.cache)}
want = [str(t) for t in json.load(open(a.tiles)) if str(t) in alltiles]
want = [w for i, w in enumerate(want) if i % a.nshards == a.shard]
st_p = os.path.join(a.out, "sf_state.json"); st = json.load(open(st_p)) if os.path.exists(st_p) else dict(done=[])
out_p = os.path.join(a.out, "sf_rows.csv"); t0 = time.time()
for tid in want:
    if tid in st["done"] or time.time() - t0 > a.hours * 3600:
        continue
    tile = alltiles[tid]
    try:
        z = ztf_tile(tile, a.scratch); g = gaia_tile(tile, a.cache); allg = gaia_all(tile, a.cache)
    except Exception as ex:
        att = st.setdefault("attempts", {}); att[tid] = att.get(tid, 0) + 1
        if att[tid] >= 2:
            st["done"].append(tid); st.setdefault("skipped", []).append(tid)
        json.dump(st, open(st_p, "w")); print(f"sf tile {tid}: data error {ex!r}", flush=True); continue
    tgt = g[g.ruwe < 1.4].copy()
    tgt["group"] = np.where(tgt.vari_id.notna() | (tgt.phot_variable_flag == "VARIABLE"), "flagged", "quiet")
    m = match(z, tgt, allg); zz = z.reset_index(drop=True)
    cnt = {grp: dict(targets=int((tgt.group == grp).sum()), matched=int((m.group == grp).sum()),
                     isolated=int(((m.group == grp) & m.isolated).sum()),
                     unique_ztf=int(((m.group == grp) & m.isolated & ~m.shared_ztf).sum())) for grp in ("flagged", "quiet")}
    m = m[m.isolated & ~m.shared_ztf]
    io = np.array([cheap_stats(zz.t.iat[i], zz.m.iat[i], zz.e.iat[i]) for i in m.zi.values]) if len(m) else np.array([])
    m = m[io >= cut] if len(m) else m
    for grp in cnt: cnt[grp]["prefilter"] = int((m.group == grp).sum())
    jobs = [(zz.t.iat[i], zz.m.iat[i], zz.e.iat[i]) for i in m.zi.values]
    with Pool(a.workers) as pool:
        fs = pool.map(_feat, jobs, chunksize=20)
    rows = []
    for (k, r), f in zip(m.iterrows(), fs):
        if not f:
            continue
        ecl_alias = is_alias(float(f["ecl_period"])) if f["ecl_pass"] else False
        ecl_ok = f["ecl_pass"] and not ecl_alias
        if not (f["ls_pass"] or f["ecl_pass"]):
            continue                                       # no channel passes: not a detection under any variant
        s = float(clf.predict_proba(pd.DataFrame([f])[SHAPE_FEATURES])[:, 1][0])
        ecl = f["ecl_pass"] and not f["ls_pass"]; per = float(f["ecl_period"] if ecl else f["period"])
        zi = int(r.zi); lc = pd.DataFrame(dict(mjd=zz.t.iat[zi], mag=zz.m.iat[zi], magerr=zz.e.iat[zi]))
        halves = bool((temporal_consistency_ecl if ecl else temporal_consistency_lc)(lc, per)["confirmed"])
        chan_new = f["ls_pass"] or ecl_ok
        rows.append(dict(tile=tid, source_id=int(r.source_id), group=r.group, objectid=int(zz.objectid.iat[zi]),
                         G=r.phot_g_mean_mag, bp_rp=r.bp_rp, phot_g_n_obs=r.phot_g_n_obs, gaia_class=r.best_class_name,
                         score=s, channel="eclipse" if ecl else "periodic", period=per, amp=f["amp"],
                         amp_over_err=f["amp_over_err"], ecl_alias=bool(ecl_alias), halves=halves,
                         det_orig=bool(s >= thr and halves), det_clf=bool(s >= thr and chan_new and halves),
                         det_free=bool(chan_new and halves)))
    if rows:
        pd.DataFrame(rows).to_csv(out_p, mode="a", header=not os.path.exists(out_p), index=False)
    st["done"].append(tid); st.setdefault("counts", {})[tid] = cnt
    json.dump(st, open(st_p, "w"))
    try: os.remove(os.path.join(a.scratch, f"ztf_{tid}.pkl"))
    except OSError: pass
    print(f"sf tile {tid}: {cnt} -> {sum(r['det_clf'] for r in rows)} det_clf, {sum(r['det_free'] for r in rows)} det_free "
          f"({time.time() - t0:.0f}s)", flush=True)
