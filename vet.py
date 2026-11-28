"""vet.py - independent vetting of candidates on real data.

For each candidate:
  * every ZTF DR23 light curve within 2" (both g and r, every field / CCD) is analysed separately;
    a real star must show the same period in independent fields and in both bands;
  * Gaia DR3 neighbours within 15" of any brightness (blending check, ZTF PSF ~2");
  * phase-folded plots at P and 2P for each band.
"""
import sys, os, json, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realsky import HATS, GAIA_TAP, GAIA_TAP_FALLBACK
from varwise_gaia_hunt import lc_features


def partition_rows(tile, ra, dec, rad=2.0):
    import s3fs, pyarrow.parquet as pq
    fs = s3fs.S3FileSystem(anon=True)
    f = pq.ParquetFile(fs.open(f"{HATS}/dataset/Norder={tile.get('order', 5)}/Dir={tile['dir']}/Npix={tile.get('npix', tile['id'])}/part0.snappy.parquet",
                               block_size=32 << 20))
    out = {i: [] for i in range(len(ra))}
    for g in range(f.metadata.num_row_groups):
        pos = f.read_row_group(g, columns=["objra", "objdec"]).to_pandas()
        hit = {}
        for i, (a, d) in enumerate(zip(ra, dec)):
            sep = 3600 * np.hypot((pos.objra.values - a) * math.cos(math.radians(d)), pos.objdec.values - d)
            for j in np.where(sep < rad)[0]:
                hit.setdefault(int(j), []).append((i, float(sep[j])))
        if not hit:
            continue
        tb = f.read_row_group(g, columns=["objectid", "filterid", "fieldid", "rcid", "lightcurve"]).take(sorted(hit))
        recs = tb.to_pylist()
        for j, r in zip(sorted(hit), recs):
            lc = r["lightcurve"]
            k = np.array(lc["catflags"]) == 0
            for i, s in hit[j]:
                out[i].append(dict(objectid=r["objectid"], band="gri"[r["filterid"] - 1], field=r["fieldid"],
                                   rcid=r["rcid"], sep=s, t=np.array(lc["hmjd"])[k], m=np.array(lc["mag"])[k],
                                   e=np.array(lc["magerr"])[k]))
    return out


def gaia_neighbours(ra, dec, rad=15.0):
    import pyvo
    q = (f"SELECT source_id, ra, dec, phot_g_mean_mag, DISTANCE(POINT('ICRS',ra,dec),POINT('ICRS',{ra},{dec}))*3600 AS sep "
         f"FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{rad / 3600}))")
    import time
    for k in range(5):
        try:
            return pyvo.dal.TAPService(GAIA_TAP if k % 2 == 0 else GAIA_TAP_FALLBACK).run_sync(q).to_table().to_pandas().sort_values("sep")
        except Exception:
            time.sleep(10 * (k + 1))
    raise RuntimeError("Gaia TAP unavailable")


def simbad(ra, dec, rad=10):
    import requests
    for k in range(3):
        try:
            x = requests.get("https://simbad.cfa.harvard.edu/simbad/sim-coo", timeout=60, params=dict(
                Coord=f"{ra} {dec:+}", Radius=rad, **{"Radius.unit": "arcsec", "output.format": "ASCII"}))
            if x.status_code == 200:
                return "" if "No astronomical object found" in x.text else x.text[:400].replace("\n", " | ")
        except Exception:
            pass
    return "ERROR"


def refine(rows, P):
    """Joint multi-harmonic Lomb-Scargle over all light curves (each median-subtracted) near P and 2P."""
    from astropy.timeseries import LombScargle
    t = np.concatenate([r["t"] for r in rows]); m = np.concatenate([r["m"] - np.median(r["m"]) for r in rows])
    e = np.concatenate([r["e"] for r in rows])
    best = None
    for Q in (P, 2 * P):
        f = np.linspace(1 / Q * 0.995, 1 / Q * 1.005, 4001)
        pw = LombScargle(t, m, e, nterms=4).power(f)
        j = int(np.argmax(pw))
        if best is None or pw[j] > best[1] * 1.02:
            best = (1 / f[j], pw[j])
    return best[0]


def near(p, ref, tol=0.01):
    return any(abs(p / (ref * k) - 1) <= tol for k in (0.5, 1.0, 2.0))


def plot_candidate(rows, Ps, title, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    bands = [b for b in "gr" if any(r["band"] == b for r in rows)]
    fig, ax = plt.subplots(len(bands), 2, figsize=(10, 2.7 * len(bands)), squeeze=False)
    col = {"g": "tab:green", "r": "tab:red"}
    for i, b in enumerate(bands):
        rb = [r for r in rows if r["band"] == b]
        ref = np.median(np.concatenate([r["m"] for r in rb]))
        for k, P in enumerate(Ps):
            ph = np.concatenate([((r["t"] - 58000.0) / P) % 1 for r in rb])
            mm = np.concatenate([r["m"] - np.median(r["m"]) + ref for r in rb])
            a = ax[i, k]
            a.plot(np.r_[ph, ph + 1], np.r_[mm, mm], ".", ms=1.5, alpha=.25, color=col[b])
            bins = np.linspace(0, 1, 51); idx = np.digitize(ph, bins) - 1
            med = np.array([np.median(mm[idx == j]) if np.any(idx == j) else np.nan for j in range(50)])
            xc = (bins[:-1] + bins[1:]) / 2
            a.plot(np.r_[xc, xc + 1], np.r_[med, med], "-", color="k", lw=1)
            lo, hi = np.nanpercentile(mm, [0.5, 99.5]); pad = 0.15 * (hi - lo)
            a.set_ylim(hi + pad, lo - pad); a.set_ylabel(f"ZTF {b} (mag)")
            a.set_xlabel(f"phase, P = {P:.6f} d")
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def season_test(rows, P):
    """Is the signal present season by season, whatever its shape? A 40-bin median template is built at P and
    at 2P from all g and r data (each light curve median-subtracted); each observing season (gap > 60 d) with
    >= 40 points that spans >= 3 cycles is fitted with that template (free offset and scale) and counts as a
    detection when the fit lowers chi^2 by more than chance allows (p < 1e-3, two parameters).
    Returns (n_seasons_tested, n_seasons_detected) for whichever of P, 2P gives more detections."""
    from scipy.stats import chi2 as chi2d
    rows = [r for r in rows if r["band"] in "gr"]
    if not rows or P > 60:
        return 0, 0
    best = (0, 0)
    for Q in (P, 2 * P):
        det = tes = 0
        for band in "gr":
            rb = [r for r in rows if r["band"] == band]
            if not rb:
                continue
            t = np.concatenate([r["t"] for r in rb]); m = np.concatenate([r["m"] - np.median(r["m"]) for r in rb])
            e = np.concatenate([r["e"] for r in rb]); o = np.argsort(t); t, m, e = t[o], m[o], e[o]
            ph = ((t - 58000.0) / Q) % 1; k = (ph * 40).astype(int)
            tmpl = np.array([np.median(m[k == j]) if np.any(k == j) else 0.0 for j in range(40)])
            x = tmpl[k]
            for idx in np.split(np.arange(len(t)), np.where(np.diff(t) > 60)[0] + 1):
                if len(idx) < 40 or np.ptp(t[idx]) < 3 * Q:
                    continue
                tes += 1
                w = 1 / np.maximum(e[idx], 0.005) ** 2; y = m[idx]; X = np.vstack([np.ones(len(idx)), x[idx]]).T
                c0 = np.sum(w * (y - np.average(y, weights=w)) ** 2)
                beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
                c1 = np.sum(w * (y - X @ beta) ** 2)
                det += (beta[1] > 0.3) and chi2d.sf(c0 - c1, 1) < 1e-3
        if det > best[1] or (det == best[1] and tes > best[0]):
            best = (tes, det)
    return best


def phase_sampling(rows, P):
    """Fraction of 40 phase bins holding >= 5 g/r points, and the fullest bin over the median bin, at period P.
    A period produced by the survey's cadence piles data into a few phases."""
    t = np.concatenate([r["t"] for r in rows if r["band"] in "gr"] or [np.array([])])
    if len(t) == 0:
        return 0.0, 99.0
    c = np.bincount((((t - 58000.0) / P) % 1 * 40).astype(int), minlength=40)
    return float((c >= 5).mean()), float(c.max() / max(np.median(c), 1))


ALIASES = (1.0, 0.5, 1 / 3, 2.0, 29.53, 14.77, 365.25, 182.6)


def auto_grade(r):
    """Rule-based grade mirroring the by-eye grading: A strong, B coherent, C flagged as a possible artefact."""
    flags = []
    if r["seasons_tested"] >= 2 and r["seasons_detected"] < max(2, 0.5 * r["seasons_tested"]):
        flags.append("period not stable season to season")
    if any(abs(r["period_refined"] / a - 1) < 0.01 or abs(2 * r["period_refined"] / a - 1) < 0.01 for a in ALIASES):
        flags.append("period near a sampling alias")
    if r["period_refined"] > 1 and (r["phase_coverage"] < 0.9 or r["phase_clumping"] >= 2.5):
        flags.append("uneven phase sampling: period may come from the survey cadence")
    if r["brighter_neighbour_15as"] > 0:
        flags.append("brighter Gaia star within 6 arcsec, or one 2 mag brighter within 12 arcsec")
    if r["verdict"] != "PASS":
        return "-", "; ".join(flags)
    if flags:
        return "C", "; ".join(flags)
    if (r["amp_r"] >= 0.08 and r["n_agree"] >= 3 and len(set(r["bands_agree"]) & set("gr")) == 2) or \
       (r["amp_r"] >= 0.15 and r["n_agree"] >= 2):
        return "A", ""
    return "B", ""


def vet(cands, tiles, outdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    res = []
    for tid, grp in cands.groupby("tile"):
        tile = tiles[str(tid)]
        lcs = partition_rows(tile, grp.ra.values, grp.dec.values)
        for i, c in enumerate(grp.itertuples()):
            rows = [r for r in lcs[i] if len(r["t"]) >= 40]
            per_lc = []
            for r in rows:
                f = lc_features(r["t"], r["m"], r["e"], fmax=max(10.0, 1.6 / c.period))
                if f:
                    per_lc.append(dict(band=r["band"], field=r["field"], rcid=r["rcid"], n=len(r["t"]),
                                       period=f["period"], log_fap=f["log_fap"], amp=f["amp"],
                                       agrees=near(f["period"], c.period) and f["log_fap"] < -3))
            nb = gaia_neighbours(c.ra, c.dec)
            G = float(nb.phot_g_mean_mag.iloc[0]) if len(nb) else np.nan
            bright_nb = nb[(nb.sep > 1.0) & (nb.phot_g_mean_mag < G + 1.0)]
            bands_ok = sorted({p["band"] for p in per_lc if p["agrees"]})
            n_ind = sum(p["agrees"] for p in per_lc)
            verdict = ("PASS" if n_ind >= 2 and len(bright_nb[bright_nb.sep < 8]) == 0 else
                       "WEAK" if n_ind >= 1 else "FAIL")
            res.append(dict(source_id=c.source_id, ra=c.ra, dec=c.dec, period=c.period, n_lc=len(per_lc),
                            n_agree=n_ind, bands_agree="".join(bands_ok),
                            bright_neighbour_8as=int((bright_nb.sep < 8).sum()),
                            nearest_bright_as=float(bright_nb.sep.min()) if len(bright_nb) else np.nan,
                            verdict=verdict, detail=json.dumps(per_lc, default=float)))
            Pr = refine(rows, c.period) if rows else c.period
            res[-1]["period_refined"] = Pr
            res[-1]["simbad_10as"] = simbad(c.ra, c.dec)
            np.savez_compressed(os.path.join(outdir, f"{c.source_id}_lc.npz"), **{f"{r['band']}_{r['field']}_{r['rcid']}_{k}": r[k]
                                for r in rows for k in ("t", "m", "e")})
            st, sd = season_test(rows, Pr)
            res[-1]["phase_coverage"], res[-1]["phase_clumping"] = phase_sampling(rows, Pr)
            res[-1].update(seasons_tested=st, seasons_detected=sd, G=G,
                           brighter_neighbour_15as=int((((nb.sep > 1.0) & (nb.sep < 6) & (nb.phot_g_mean_mag < G)) |
                                                           ((nb.sep > 1.0) & (nb.sep < 12) & (nb.phot_g_mean_mag < G - 2))).sum()),
                           amp_r=float(np.median([p["amp"] for p in per_lc if p["band"] == "r" and p["agrees"]] or [0])))
            res[-1]["auto_grade"], res[-1]["auto_flags"] = auto_grade(res[-1])
            Ps = (Pr / 2, Pr) if Pr > 1.5 * c.period else (Pr, 2 * Pr)
            plot_candidate(rows, Ps, f"Gaia DR3 {c.source_id}  ({c.ra:.5f}, {c.dec:+.5f})  G = {G:.2f}  verdict {verdict}",
                           os.path.join(outdir, f"{c.source_id}.png"))
            print(res[-1]["source_id"], verdict, n_ind, "/", len(per_lc), bands_ok, "bright nb<8\":",
                  res[-1]["bright_neighbour_8as"], flush=True)
    return pd.DataFrame(res)


if __name__ == "__main__":
    import glob
    from realsky import order5_tiles, extra_order_tiles
    cache = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("CACHE", "/home/claude/work/cache")
    tiles = {str(t["id"]): t for t in order5_tiles(cache, 0, 90, -90) + extra_order_tiles(cache, 0, 90, -90)}
    c = pd.read_csv(sys.argv[1])
    vet(c, tiles, sys.argv[2]).to_csv(os.path.join(sys.argv[2], "vetting.csv"), index=False)
