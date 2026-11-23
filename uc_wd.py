"""uc_wd.py - search for ultracompact binaries (periods 3-60 min) among Gaia white dwarfs in ZTF DR23 light curves.

Targets: Gentile Fusillo et al. (2021) Gaia EDR3 white dwarfs with P_WD > 0.75 and Dec > -28 (VizieR J/MNRAS/508/3877).
For each target, every ZTF DR23 g and r light curve within 1.5" is read from the IRSA HATS archive. A Lomb-Scargle
search runs over 24-480 cycles/day (periods 3-60 min) in each band. A candidate needs:
  * log10 FAP < -8 in the band with the stronger peak,
  * the same frequency (or its double or half, ellipsoidal binaries show twice the orbital frequency) in the other
    band at log10 FAP < -3, which removes single-band and single-field artefacts,
  * a frequency away from integer cycles per day (sidereal and daily aliases).
Positive control: ZTF J153932.16+502738.8 (6.91-min eclipsing double white dwarf, Burdge et al. 2019) must be found.

usage: python uc_wd.py targets <cache dir>                         # fetch and partition the white-dwarf list
       python uc_wd.py run <cache dir> <out dir> --shard k --nshards n --hours h
       python uc_wd.py control                                       # positive control
"""
import argparse, io, json, math, os, sys, time
import numpy as np, pandas as pd, requests

FMIN, FMAX = 24.0, 480.0
CONTROL = (234.88400, 50.46078, 6.91)       # ZTF J1539+5027: RA, Dec, period in minutes (Burdge et al. 2019)


def targets(cache):
    p = os.path.join(cache, "wd_targets.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    url = ("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=J/MNRAS/508/3877/maincat"
           "&-out=GaiaEDR3,RA_ICRS,DE_ICRS,Gmag,Pwd&Pwd=>0.75&DE_ICRS=>-28&-out.max=unlimited")
    for k in range(5):
        try:
            r = requests.get(url, timeout=1800); r.raise_for_status()
            lines = [l for l in r.text.splitlines() if l and not l.startswith("#")]
            d = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[3:])), sep="\t")
            break
        except Exception:
            if k == 4:
                raise
            time.sleep(30)
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    d.columns = ["source_id", "ra", "dec", "G", "pwd"]
    from followup import pinfo
    from astropy_healpix import HEALPix
    import astropy.units as u
    pi = pinfo(); d["order"] = -1; d["npix"] = -1
    for k in sorted(pi.Norder.unique()):
        have = set(pi.Npix[pi.Norder == k])
        hp = HEALPix(nside=2 ** k, order="nested").lonlat_to_healpix(d.ra.values * u.deg, d.dec.values * u.deg)
        m = (d.order < 0) & np.isin(hp, list(have))
        d.loc[m, "order"] = k; d.loc[m, "npix"] = hp[m]
    d = d[d.order >= 0]; d.to_parquet(p)
    return d


def lightcurves(order, npix, ra, dec, rad=1.5, min_points=60):
    """{target index: [dict(band, field, t, m, e)]} for all targets in one HATS partition."""
    import s3fs, pyarrow.parquet as pq
    from followup import HATS
    fs = s3fs.S3FileSystem(anon=True)
    f = pq.ParquetFile(fs.open(f"{HATS}/dataset/Norder={order}/Dir={(npix // 10000) * 10000}/Npix={npix}/part0.snappy.parquet",
                               block_size=32 << 20))
    from scipy.spatial import cKDTree
    def xyz(a, d):
        a, d = np.radians(a), np.radians(d)
        return np.c_[np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)]
    tree = cKDTree(xyz(ra, dec)); r = 2 * math.sin(math.radians(rad / 3600) / 2)
    out = {}
    for g in range(f.metadata.num_row_groups):
        pos = f.read_row_group(g, columns=["objra", "objdec", "nepochs"]).to_pandas()
        ok = (pos.nepochs >= min_points).values
        if not ok.any():
            continue
        idx = np.where(ok)[0]
        hits = tree.query_ball_point(xyz(pos.objra.values[idx], pos.objdec.values[idx]), r)
        take = [(int(idx[i]), h) for i, h in enumerate(hits) if h]
        if not take:
            continue
        rows = f.read_row_group(g, columns=["filterid", "fieldid", "lightcurve"]).take([j for j, _ in take]).to_pylist()
        for (j, h), row in zip(take, rows):
            lc = row["lightcurve"]; k = np.array(lc["catflags"]) == 0
            rec = dict(band="gri"[row["filterid"] - 1], field=row["fieldid"], t=np.array(lc["hmjd"])[k],
                       m=np.array(lc["mag"])[k], e=np.array(lc["magerr"])[k])
            for ti in h:
                out.setdefault(ti, []).append(rec)
    return out


def _band(rows, b):
    rb = [r for r in rows if r["band"] == b and len(r["t"]) >= 30]
    if not rb:
        return None
    t = np.concatenate([r["t"] for r in rb]); m = np.concatenate([r["m"] - np.median(r["m"]) for r in rb])
    e = np.concatenate([r["e"] for r in rb]); ok = np.isfinite(t) & np.isfinite(m) & (e > 0)
    return (t[ok], m[ok], e[ok]) if ok.sum() >= 60 else None


def search_one(rows):
    from astropy.timeseries import LombScargle
    res = {}
    for b in "gr":
        d = _band(rows, b)
        if d is None:
            continue
        t, m, e = d; ls = LombScargle(t, m, e)
        fr, pw = ls.autopower(minimum_frequency=FMIN, maximum_frequency=FMAX, samples_per_peak=2, method="fast")
        bad = np.abs(fr - np.round(fr)) < 0.02                      # integer cycles per day
        pw = np.where(bad, 0, pw); j = int(np.argmax(pw))
        fz = np.linspace(fr[j] - 2 * (fr[1] - fr[0]), fr[j] + 2 * (fr[1] - fr[0]), 61); pz = ls.power(fz)
        f0 = float(fz[np.argmax(pz)])
        res[b] = dict(f=f0, power=float(pz.max()), n=len(t), ls=ls,
                      log_fap=float(np.log10(max(ls.false_alarm_probability(pz.max(), minimum_frequency=FMIN,
                                                                            maximum_frequency=FMAX), 1e-300))))
    if not res:
        return None
    best = min(res, key=lambda b: res[b]["log_fap"]); other = "r" if best == "g" else "g"
    out = dict(best_band=best, freq=res[best]["f"], period_min=1440 / res[best]["f"], log_fap=res[best]["log_fap"],
               power=res[best]["power"], n_best=res[best]["n"])
    if other in res:
        ls = res[other]["ls"]; agree = -0.0
        for mult in (1.0, 0.5, 2.0):
            f1 = res[best]["f"] * mult
            if not FMIN / 2 <= f1 <= FMAX * 2:
                continue
            w = 3e-4 * f1; fz = np.linspace(f1 - w, f1 + w, 41); p = float(ls.power(fz).max())
            fap = float(np.log10(max(ls.false_alarm_probability(p, minimum_frequency=f1 - w, maximum_frequency=f1 + w), 1e-300)))
            agree = min(agree, fap)
        out["other_log_fap"] = agree; out["n_other"] = res[other]["n"]
    else:
        out["other_log_fap"] = 0.0; out["n_other"] = 0
    out["candidate"] = bool(out["log_fap"] < -8 and out["other_log_fap"] < -3)
    return out


def run(a):
    T = targets(a.cache)
    parts = sorted(T.groupby(["order", "npix"]).groups)
    mine = [p for i, p in enumerate(parts) if i % a.nshards == a.shard]
    os.makedirs(a.out, exist_ok=True); t_end = time.time() + a.hours * 3600
    from multiprocessing import Pool
    for order, npix in mine:
        tag = f"{order}_{npix}"; fn = os.path.join(a.out, f"uc_{tag}.csv")
        if os.path.exists(fn) or (a.done_from and os.path.exists(os.path.join(a.done_from, f"uc_{tag}.csv"))):
            continue
        if time.time() > t_end:
            break
        g = T[(T.order == order) & (T.npix == npix)].reset_index(drop=True)
        try:
            lcs = lightcurves(order, npix, g.ra.values, g.dec.values)
        except Exception as ex:
            print("partition", tag, "failed:", str(ex)[:200], flush=True); continue
        items = [(i, lcs[i]) for i in lcs]
        with Pool(os.cpu_count() or 2) as pool:
            outs = pool.map(_job, items, chunksize=4)
        rows = []
        for (i, _), o in zip(items, outs):
            if o:
                o.update(source_id=int(g.source_id[i]), ra=g.ra[i], dec=g.dec[i], G=g.G[i]); rows.append(o)
        pd.DataFrame(rows).to_csv(fn, index=False)
        c = sum(r["candidate"] for r in rows)
        print(json.dumps(dict(partition=tag, targets=len(g), with_lc=len(items), candidates=int(c))), flush=True)


def _job(item):
    try:
        return search_one(item[1])
    except Exception:
        return None


def control():
    import followup as FU
    ra, dec, pmin = CONTROL
    rows = [dict(band=r["band"], field=r["field"], t=r["t"] - 2400000.5, m=r["m"], e=r["e"]) for r in FU.ztf(ra, dec, 1.5)]
    o = search_one(rows)
    print(json.dumps({k: v for k, v in o.items()}, indent=1))
    ok = o["candidate"] and any(abs(o["period_min"] * mult - pmin) / pmin < 0.002 for mult in (1, 2, 0.5))
    print("POSITIVE CONTROL", "PASSED" if ok else "FAILED", "(expected", pmin, "min or its harmonic)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["targets", "run", "control"])
    ap.add_argument("cache", nargs="?", default="cache"); ap.add_argument("out", nargs="?", default="uc_out")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--hours", type=float, default=5.0); ap.add_argument("--done-from", default=None)
    a = ap.parse_args()
    if a.cmd == "targets":
        T = targets(a.cache); print(len(T), "white dwarfs in", T.groupby(["order", "npix"]).ngroups, "partitions")
    elif a.cmd == "run":
        run(a)
    else:
        sys.exit(0 if control() else 1)
