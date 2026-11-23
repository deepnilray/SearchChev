"""followup.py - independent checks for dr4engine candidates: ZTF DR23 light curves and NEOWISE infrared.

ztf(ra, dec)      every ZTF DR23 light curve within 2" (all fields, CCDs, bands), read from the IRSA HATS archive on S3
neowise(ra, dec)  NEOWISE-R single exposures within 3" (plus AllWISE multi-epoch), quality-cut and binned per visit
assess_dip(...)   does ZTF show a dimming episode independent of Gaia; does the star brighten in the infrared
                  (warm dust, the giant-impact signature) and when, relative to the optical dimming

usage: python followup.py <candidates.csv> <out.csv>     (needs columns source_id, ra, dec; optional dip_ep_t)
"""
import io, math, os, sys, time
import numpy as np, pandas as pd, requests

HATS = "ipac-irsa-ztf/contributed/dr23/lc/hats/ztf_dr23_lc-hats"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
_PINFO = None


def pinfo():
    global _PINFO
    if _PINFO is None:
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        _PINFO = pd.read_csv(io.BytesIO(fs.cat(f"{HATS}/partition_info.csv")))
    return _PINFO


def _partition(ra, dec):
    from astropy_healpix import HEALPix
    import astropy.units as u
    p = pinfo()
    for k in sorted(p.Norder.unique()):
        npix = int(HEALPix(nside=2 ** k, order="nested").lonlat_to_healpix(ra * u.deg, dec * u.deg))
        if ((p.Norder == k) & (p.Npix == npix)).any():
            return k, npix, (npix // 10000) * 10000
    return None


def ztf(ra, dec, rad=2.0):
    """List of dict(band, field, rcid, sep, t, m, e) for every ZTF DR23 light curve within rad arcsec."""
    import s3fs, pyarrow.parquet as pq
    part = _partition(ra, dec)
    if part is None:
        return []
    k, npix, d = part
    fs = s3fs.S3FileSystem(anon=True)
    f = pq.ParquetFile(fs.open(f"{HATS}/dataset/Norder={k}/Dir={d}/Npix={npix}/part0.snappy.parquet", block_size=32 << 20))
    out = []
    for g in range(f.metadata.num_row_groups):
        pos = f.read_row_group(g, columns=["objra", "objdec"]).to_pandas()
        sep = 3600 * np.hypot((pos.objra.values - ra) * math.cos(math.radians(dec)), pos.objdec.values - dec)
        hit = np.where(sep < rad)[0]
        if not hit.size:
            continue
        for j, r in zip(hit, f.read_row_group(g, columns=["filterid", "fieldid", "rcid", "lightcurve"]).take(hit).to_pylist()):
            lc = r["lightcurve"]; ok = np.array(lc["catflags"]) == 0
            out.append(dict(band="gri"[r["filterid"] - 1], field=r["fieldid"], rcid=r["rcid"], sep=float(sep[j]),
                            t=np.array(lc["hmjd"])[ok] + 2400000.5, m=np.array(lc["mag"])[ok], e=np.array(lc["magerr"])[ok]))
    return out


def _tap(q, tries=4):
    for k in range(tries):
        try:
            r = requests.get(IRSA_TAP, params=dict(QUERY=q, FORMAT="csv"), timeout=300)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception:
            time.sleep(10 * (k + 1))
    raise RuntimeError("IRSA TAP failed")


def neowise(ra, dec, rad=3.0):
    """Per-visit W1/W2 medians (visits every ~6 months, 2010-2024). Columns: jd, w1, w1e, w2, w2e, n."""
    cone = f"CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{rad / 3600}))=1"
    q1 = ("SELECT mjd,w1mpro,w1sigmpro,w2mpro,w2sigmpro FROM neowiser_p1bs_psd WHERE " + cone +
          " AND qual_frame>0 AND qi_fact=1 AND saa_sep>5 AND moon_masked='00' AND cc_flags LIKE '00%'")
    q2 = ("SELECT mjd,w1mpro_ep AS w1mpro,w1sigmpro_ep AS w1sigmpro,w2mpro_ep AS w2mpro,w2sigmpro_ep AS w2sigmpro "
          "FROM allwise_p3as_mep WHERE " + cone + " AND qi_fact=1 AND saa_sep>5 AND moon_masked='00' AND cc_flags LIKE '00%'")
    d = pd.concat([_tap(q1), _tap(q2)], ignore_index=True)
    if d.empty:
        return d
    d = d.sort_values("mjd"); visit = np.r_[0, np.cumsum(np.diff(d.mjd.values) > 30)]
    rows = []
    for _, g in d.groupby(visit):
        rows.append(dict(jd=g.mjd.median() + 2400000.5, w1=g.w1mpro.median(), w2=g.w2mpro.median(), n=len(g),
                         w1e=1.2533 * g.w1mpro.std() / math.sqrt(len(g)) if len(g) > 2 else g.w1sigmpro.median(),
                         w2e=1.2533 * g.w2mpro.std() / math.sqrt(len(g)) if len(g) > 2 else g.w2sigmpro.median()))
    return pd.DataFrame(rows)


def ir_brightening(nw, min_sig=5.0, min_amp=0.15):
    """Largest sustained W1 brightening relative to the earliest three visits: (amp_mag, sigma, jd_peak, w1w2_change)."""
    if len(nw) < 6:
        return np.nan, np.nan, np.nan, np.nan
    base = nw.iloc[:3]; b1, b12 = base.w1.median(), (base.w1 - base.w2).median()
    s = np.sqrt(nw.w1e.fillna(0.05).clip(lower=0.01) ** 2 + 0.01 ** 2)
    dm = b1 - nw.w1.values                                   # positive = brighter
    sustained = np.minimum(dm[:-1], dm[1:])                  # two consecutive visits
    j = int(np.nanargmax(sustained))
    return float(sustained[j]), float(sustained[j] / s.iloc[j]), float(nw.jd.iat[j]), float((nw.w1 - nw.w2).iat[j] - b12)


def zdip(rows, min_depth=0.1):
    """Dimming episodes in each ZTF g/r light curve. Faint points (>5 sigma) are split into episodes wherever a normal
    point lies between them. A single dimming (disk, ring, debris cloud) gives one or two episodes; an eclipsing
    binary gives many short ones spread over the years. Returns one dict per light curve that has faint points."""
    out = []
    for r in rows:
        if r["band"] not in "gr" or len(r["t"]) < 30:
            continue
        o = np.argsort(r["t"]); t, m, e = r["t"][o], r["m"][o], r["e"][o]
        med = np.median(m); mad = 1.4826 * np.median(np.abs(m - med)) + 1e-3
        k = (m - med) / np.sqrt(e ** 2 + mad ** 2) > 5
        if k.sum() < 3 or (m[k] - med).max() < min_depth:
            continue
        idx = np.where(k)[0]; eps = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        eps = [g for g in eps if g.size >= 2]
        if not eps:
            continue
        main = max(eps, key=lambda g: g.size)
        out.append(dict(band=r["band"], field=r["field"], depth=float((m[k] - med).max()), n_faint=int(k.sum()),
                        n_episodes=len(eps), start=float(t[main[0]]), end=float(t[main[-1]]),
                        main_frac=float(main.size / k.sum())))
    return out


def assess(c):
    out = dict(source_id=c.source_id)
    try:
        rows = ztf(c.ra, c.dec); h = zdip(rows)
        single = [x for x in h if x["n_episodes"] <= 2 and x["main_frac"] >= 0.6]
        recur = [x for x in h if x["n_episodes"] >= 4]
        out.update(ztf_lcs=len(rows), ztf_dip_lcs=len(h), ztf_dip_depth=max((x["depth"] for x in h), default=np.nan),
                   ztf_single_episode_lcs=len(single), ztf_recurrent_lcs=len(recur),
                   ztf_dip_start=min((x["start"] for x in single), default=np.nan),
                   ztf_dip_end=max((x["end"] for x in single), default=np.nan),
                   # a single dimming seen in two independent light curves, overlapping in time, and nothing recurrent
                   ztf_confirms=bool(len({(x["band"], x["field"]) for x in single}) >= 2 and not recur and
                                     min(x["end"] for x in single) >= max(x["start"] for x in single) - 30),
                   ztf_eclipsing=bool(len(recur) >= 2))
    except Exception as ex:
        out["ztf_error"] = str(ex)[:120]
    try:
        nw = neowise(c.ra, c.dec)
        amp, sig, jd, dcol = ir_brightening(nw)
        out.update(neowise_visits=len(nw), ir_amp=amp, ir_sig=sig, ir_peak_jd=jd, ir_w1w2_change=dcol,
                   ir_brightening=bool(amp >= 0.15 and sig >= 5) if np.isfinite(amp) else False)
    except Exception as ex:
        out["neowise_error"] = str(ex)[:120]
    return out


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    C = pd.read_csv(sys.argv[1])
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(assess, C.itertuples()))
    C.merge(pd.DataFrame(res), on="source_id", how="left").to_csv(sys.argv[2], index=False)
    print(pd.DataFrame(res).describe(include="all").T.head(20))
