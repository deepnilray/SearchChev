"""realsky.py - real-sky training, sealed validation and search for SearchChev.

Data, all real, nothing synthetic:
  * ZTF DR23 light curves, read straight from the IRSA HATS parquet on S3 (anonymous).
  * Gaia DR3 labels by source_id range: the level-12 HEALPix index sits in source_id // 2**35,
    so one HATS order-5 pixel p is the source_id interval [p*4**7*2**35, (p+1)*4**7*2**35).
Positives: Gaia DR3 vari_classifier_result (an independent instrument and pipeline).
Negatives: Gaia stars with no vari_summary row, not VARIABLE, RUWE < 1.4.
           Some are unrecognised variables, so the measured false-positive rate is an upper bound.
Tiles: HATS order-5 pixels with |b| >= 15 deg. A seeded manifest fixes train / validation / search
before any download; validation tiles are opened only after the model and threshold are frozen.
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eulerattack import features, SHAPE_FEATURES, hanley_se, wilson_upper, GATE, magnitude_match, fit

HATS = "ipac-irsa-ztf/contributed/dr23/lc/hats/ztf_dr23_lc-hats"
GAIA_TAP = "https://gaia.ari.uni-heidelberg.de/tap"     # ARI Heidelberg partner centre: same gaiadr3 tables;
GAIA_TAP_FALLBACK = "https://gea.esac.esa.int/tap-server/tap"   # ESA archive is down in the run-up to DR4
SID_PER_PIX5 = 4 ** 7 * 2 ** 35
NOT_STELLAR = {"AGN", "GALAXY", "MICROLENSING"}
# Pre-registered 2026-11-22 from TRAINING tiles only, before any validation tile was featurized:
# Gaia classes whose signals ZTF r-band can see. SOLAR_LIKE (790 of 1599 training positives) and S pass the
# Lomb-Scargle gate only 5-6% of the time, so they measure Gaia's precision, not this detector.
ZTF_VISIBLE = {"ECL", "RR", "CEP", "T2CEP", "LPV", "RS", "YSO", "CV"}


# ------------------------------------------------------------------ tiles ---
def partition_info(cache):
    p = os.path.join(cache, "pinfo.csv")
    if not os.path.exists(p):
        import s3fs, io
        fs = s3fs.S3FileSystem(anon=True)
        pd.read_csv(io.BytesIO(fs.cat(f"{HATS}/partition_info.csv"))).to_csv(p, index=False)
    return pd.read_csv(p)


def order5_tiles(cache, bmin=15.0, bmax=90.0, dec_min=-25.0):
    from astropy_healpix import HEALPix
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    pi = partition_info(cache)
    pi = pi[pi.Norder == 5]
    hp = HEALPix(nside=2 ** 5, order="nested")
    lon, lat = hp.healpix_to_lonlat(pi.Npix.values)
    c = SkyCoord(lon.to(u.deg), lat.to(u.deg))
    b = np.abs(c.galactic.b.deg)
    ok = (b >= bmin) & (b <= bmax) & (c.dec.deg >= dec_min)
    out = pi[ok].assign(ra=c.ra.deg[ok], dec=c.dec.deg[ok], b=b[ok])
    return [dict(id=int(r.Npix), dir=int(r.Dir), ra=float(r.ra), dec=float(r.dec), b=float(r.b),
                 rows=int(r.num_rows)) for r in out.itertuples()]


def high_latitude_tiles(cache, dec_min=-25.0):
    """|b| > 35 deg at orders 4 and 5, outside the sealed training and validation band (|b| <= 35)."""
    from astropy_healpix import HEALPix
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    pi = partition_info(cache); out = []
    for k in (5, 4):
        p = pi[pi.Norder == k]
        lon, lat = HEALPix(nside=2 ** k, order="nested").healpix_to_lonlat(p.Npix.values)
        c = SkyCoord(lon.to(u.deg), lat.to(u.deg)); b = np.abs(c.galactic.b.deg)
        ok = (b > 35) & (c.dec.deg >= dec_min)
        for r, ra, de, bb in zip(p[ok].itertuples(), c.ra.deg[ok], c.dec.deg[ok], b[ok]):
            tid = int(r.Npix) if k == 5 else f"o{k}_{int(r.Npix)}"
            out.append(dict(id=tid, order=k, npix=int(r.Npix), dir=int(r.Dir), ra=float(ra), dec=float(de),
                            b=float(bb), rows=int(r.num_rows)))
    return out


def extra_order_tiles(cache, bmin=15.0, bmax=35.0, dec_min=-25.0):
    """HATS partitions stored at orders 4 and 6 (sparser and denser sky). They never overlap order-5 partitions,
    so adding them to the search pool cannot touch the sealed training or validation sky."""
    from astropy_healpix import HEALPix
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    pi = partition_info(cache); out = []
    for k in (4, 6):
        p = pi[pi.Norder == k]
        lon, lat = HEALPix(nside=2 ** k, order="nested").healpix_to_lonlat(p.Npix.values)
        c = SkyCoord(lon.to(u.deg), lat.to(u.deg)); b = np.abs(c.galactic.b.deg)
        ok = (b >= bmin) & (b <= bmax) & (c.dec.deg >= dec_min)
        for r, ra, de, bb in zip(p[ok].itertuples(), c.ra.deg[ok], c.dec.deg[ok], b[ok]):
            out.append(dict(id=f"o{k}_{int(r.Npix)}", order=k, npix=int(r.Npix), dir=int(r.Dir), ra=float(ra),
                            dec=float(de), b=float(bb), rows=int(r.num_rows)))
    return out


def manifest(out, tiles, n_train, n_val, seed):
    path = os.path.join(out, "manifest.json")
    rng = np.random.default_rng(seed)
    order = [tiles[i] for i in rng.permutation(len(tiles))]
    man = dict(seed=seed, train=order[:n_train], validation=order[n_train:n_train + n_val],
               search=order[n_train + n_val:], created=time.strftime("%Y-%m-%d %H:%M:%S"))
    man["fingerprint"] = hashlib.sha256(json.dumps([man["train"], man["validation"]],
                                                   sort_keys=True).encode()).hexdigest()
    if os.path.exists(path):
        old = json.load(open(path))
        if old["fingerprint"] != man["fingerprint"]:
            raise SystemExit("REFUSED: existing manifest differs; use a new --out")
        return old
    json.dump(man, open(path, "w"), indent=1)
    return man


class Seal:
    """Every data access names its phase; touching a tile outside that phase raises."""
    def __init__(self, man, out):
        self.man, self.phase = man, "train"
        self.log = open(os.path.join(out, "access_log.jsonl"), "a")

    def check(self, what, tile):
        ok = any(t["id"] == tile["id"] for t in self.man[self.phase])
        self.log.write(json.dumps(dict(t=time.time(), phase=self.phase, what=what,
                                       tile=tile["id"], ok=ok)) + "\n"); self.log.flush()
        if not ok:
            raise PermissionError(f"{what} on tile {tile['id']} outside phase {self.phase}")


# ------------------------------------------------------------------- data ---
def gaia_tile(tile, cache):
    p = os.path.join(cache, f"gaia_{tile['id']}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    import pyvo
    k, npix = tile.get("order", 5), tile.get("npix", tile["id"])
    span = 4 ** (12 - k) * 2 ** 35
    lo = npix * span
    q = f"""SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.phot_g_mean_mag, g.bp_rp,
      g.phot_g_n_obs, g.phot_g_mean_flux_over_error, g.ruwe, g.phot_variable_flag,
      c.best_class_name, c.best_class_score, v.source_id AS vari_id
      FROM gaiadr3.gaia_source AS g
      LEFT JOIN gaiadr3.vari_summary AS v ON v.source_id = g.source_id
      LEFT JOIN gaiadr3.vari_classifier_result AS c ON c.source_id = g.source_id
      WHERE g.source_id BETWEEN {lo} AND {lo + span - 1}
      AND g.phot_g_mean_mag BETWEEN 13 AND 18.5"""
    for k in range(6):
        try:
            d = pyvo.dal.TAPService(GAIA_TAP if k % 2 == 0 else GAIA_TAP_FALLBACK).run_sync(
                q, maxrec=2_000_000).to_table().to_pandas(); break
        except Exception as ex:
            if k == 5:
                raise
            time.sleep(10 * (k + 1))
    for c in ("best_class_name", "phot_variable_flag"):
        d[c] = d[c].astype(str).replace({"nan": "", "None": "", "--": ""})
    d.to_parquet(p)
    return d


def ztf_tile(tile, cache, min_epochs=60):
    """r-band objects with >= min_epochs, light curves kept as numpy arrays (catflags == 0)."""
    p = os.path.join(cache, f"ztf_{tile['id']}.pkl")
    if os.path.exists(p):
        return pd.read_pickle(p)
    import s3fs, pyarrow.parquet as pq, pyarrow.compute as pc
    fs = s3fs.S3FileSystem(anon=True)
    f = pq.ParquetFile(fs.open(f"{HATS}/dataset/Norder={tile.get('order', 5)}/Dir={tile['dir']}/Npix={tile.get('npix', tile['id'])}/"
                               "part0.snappy.parquet", block_size=32 << 20))
    rows = []
    for g in range(f.metadata.num_row_groups):
        tb = f.read_row_group(g, columns=["objectid", "filterid", "objra", "objdec", "nepochs", "lightcurve"])
        tb = tb.filter(pc.and_(pc.equal(tb["filterid"], 2), pc.greater_equal(tb["nepochs"], min_epochs)))
        lc = tb["lightcurve"].combine_chunks()
        cols = {k: lc.field(k) for k in ("hmjd", "mag", "magerr", "catflags")}
        oid, ra, de = (tb[c].to_numpy() for c in ("objectid", "objra", "objdec"))
        for i in range(tb.num_rows):
            t = np.asarray(cols["hmjd"][i].values, float)
            m = np.asarray(cols["mag"][i].values, float)
            e = np.asarray(cols["magerr"][i].values, float)
            fl = np.asarray(cols["catflags"][i].values)
            k = fl == 0
            if k.sum() >= min_epochs:
                rows.append((int(oid[i]), float(ra[i]), float(de[i]), t[k], m[k], e[k]))
        del tb, lc
    d = pd.DataFrame(rows, columns=["objectid", "ra", "dec", "t", "m", "e"])
    d["n"] = d.t.map(len)
    d = d.sort_values("n", ascending=False)
    d.to_pickle(p)
    return d


def crossmatch(z, g, radius=1.5):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if z.empty or g.empty:
        return g.iloc[:0].assign(zi=[])
    cz = SkyCoord(z.ra.values * u.deg, z.dec.values * u.deg)
    cg = SkyCoord(g.ra.values * u.deg, g.dec.values * u.deg)
    idx, sep, _ = cg.match_to_catalog_sky(cz)                    # nearest ZTF row per Gaia star
    _, sep2, _ = cg.match_to_catalog_sky(cg, nthneighbor=2)      # isolation from other Gaia stars
    ok = (sep.arcsec < radius) & (sep2.arcsec > 3.0)
    m = g[ok].assign(zi=idx[ok], zsep=sep.arcsec[ok])
    # several ZTF rows (fields) per star: matching picks the nearest; keep the one with most epochs
    zz = z.reset_index(drop=True)
    near = cz.search_around_sky(cg[ok], radius * u.arcsec)
    best = {}
    for gi, zi in zip(*near[:2]):
        if gi not in best or zz.n.iat[zi] > zz.n.iat[best[gi]]:
            best[gi] = zi
    m["zi"] = [best.get(i, j) for i, j in enumerate(m.zi.values)]
    return m


def labelled(tile, cache, max_neg_ratio=3, seed=0):
    g = gaia_tile(tile, cache)
    cls = g.best_class_name
    pos = g[(cls != "") & ~cls.isin(NOT_STELLAR)].assign(label=1)
    neg = g[(g.vari_id.isna()) & (g.phot_variable_flag != "VARIABLE") & (g.ruwe < 1.4)].assign(label=0)
    return pd.concat([pos, neg], ignore_index=True)


def _feat(args):
    t, m, e = args
    try:
        return features(pd.DataFrame(dict(mjd=t, mag=m, magerr=e)))
    except Exception:
        return None


def featurize_tile(tile, seal, phase_what, cache, workers=2, max_neg=None, seed=0):
    seal.check(phase_what, tile)
    p = os.path.join(cache, f"feat_{tile['id']}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    z = ztf_tile(tile, cache)
    lab = labelled(tile, cache)
    m = crossmatch(z, lab)
    zz = z.reset_index(drop=True)
    if max_neg is not None:
        P = m[m.label == 1]; N = m[m.label == 0]
        N = N.sample(min(len(N), max(max_neg, 3 * len(P))), random_state=seed)
        m = pd.concat([P, N])
    jobs = [(zz.t.iat[i], zz.m.iat[i], zz.e.iat[i]) for i in m.zi.values]
    from multiprocessing import Pool
    with Pool(workers) as pool:
        fs = pool.map(_feat, jobs, chunksize=20)
    keep = [i for i, f in enumerate(fs) if f]
    F = pd.DataFrame([fs[i] for i in keep])
    base = m.iloc[keep].reset_index(drop=True)
    base = base.assign(objectid=zz.objectid.values[base.zi.values], tile=tile["id"])
    out = pd.concat([base.drop(columns=["zi"]), F.reset_index(drop=True)], axis=1)
    out.to_parquet(p)
    return out


# ------------------------------------------------------------- evaluation ---
def report(clf, tr, va, thr, seed=0, n_ctl=10):
    from sklearn.metrics import roc_auc_score
    y = va.label.to_numpy(); s = clf.predict_proba(va[SHAPE_FEATURES])[:, 1]
    auc = float(roc_auc_score(y, s)); n1, n0 = int(y.sum()), int((1 - y).sum())
    se = hanley_se(auc, n1, n0)
    rng = np.random.default_rng(seed); ctl = []
    for _ in range(n_ctl):
        sh = tr.copy(); sh["label"] = rng.permutation(sh.label.to_numpy())
        ctl.append(roc_auc_score(y, fit(sh, seed).predict_proba(va[SHAPE_FEATURES])[:, 1]))
    rule = (s >= thr) & (va.ls_pass.to_numpy() | va.ecl_pass.to_numpy())   # the full search decision rule
    rfp, rtp = int((rule & (y == 0)).sum()), int((rule & (y == 1)).sum())
    sel = s >= thr
    fp, tp = int((sel & (y == 0)).sum()), int((sel & (y == 1)).sum())
    per = {}
    for c, g in va[va.label == 1].groupby("best_class_name"):
        sc = clf.predict_proba(g[SHAPE_FEATURES])[:, 1] >= thr
        per[c] = dict(n=len(g), recall=float(sc.mean()),
                      rule_recall=float((sc & (g.ls_pass.to_numpy() | g.ecl_pass.to_numpy())).mean()))
    return dict(n_val_pos=n1, n_val_neg=n0, auc=auc, auc_se=se, auc_lower95=auc - 1.96 * se,
                shuffled_control=float(np.mean(ctl)), threshold=float(thr), fp=fp, tp=tp,
                fpr=fp / max(n0, 1), fpr_upper95=wilson_upper(fp, n0), recall=tp / max(n1, 1),
                precision=tp / max(tp + fp, 1), recall_by_class=per,
                rule_fp=rfp, rule_tp=rtp, rule_fpr=rfp / max(n0, 1), rule_fpr_upper95=wilson_upper(rfp, n0),
                rule_recall=rtp / max(n1, 1))


def gate(rep, min_val):
    r = []
    if min(rep["n_val_pos"], rep["n_val_neg"]) < min_val:
        r.append(f"holdout too small ({rep['n_val_pos']}/{rep['n_val_neg']})")
    if rep["auc_lower95"] < GATE["auc_lower"]:
        r.append(f"AUC lower bound {rep['auc_lower95']:.3f} < {GATE['auc_lower']}")
    if abs(rep["shuffled_control"] - 0.5) > GATE["control_band"]:
        r.append(f"shuffled control {rep['shuffled_control']:.3f}")
    if rep["fpr_upper95"] > GATE["fpr_upper"]:
        r.append(f"FPR upper bound {rep['fpr_upper95']:.4f} > {GATE['fpr_upper']}")
    if rep["recall"] < GATE["min_recall"]:
        r.append(f"recall {rep['recall']:.3f} < {GATE['min_recall']}")
    return r


def keep_visible(d):
    return d[(d.label == 0) | d.best_class_name.isin(ZTF_VISIBLE)].reset_index(drop=True)


def run_train(a):
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.cache, exist_ok=True)
    tiles = order5_tiles(a.cache, a.bmin, a.bmax)
    man = manifest(a.out, tiles, a.train_tiles, a.val_tiles, a.seed)
    seal = Seal(man, a.out)
    print(f"[manifest] {len(tiles)} tiles: {len(man['train'])} train, {len(man['validation'])} sealed, "
          f"{len(man['search'])} search; fingerprint {man['fingerprint'][:12]}", flush=True)
    seal.phase = "train"; T = []
    for t in man["train"]:
        t0 = time.time(); d = featurize_tile(t, seal, "train_features", a.cache, a.workers, a.max_neg, a.seed)
        T.append(d); print(f"  train tile {t['id']} b={t['b']:.1f}: {int(d.label.sum())} var / "
                           f"{int((1 - d.label).sum())} quiet  ({time.time() - t0:.0f}s)", flush=True)
    tr = keep_visible(pd.concat(T, ignore_index=True))
    tr = tr.assign(mean_mag=tr.phot_g_mean_mag)
    trm = magnitude_match(tr, a.seed)
    # threshold from grouped out-of-fold scores on the UNMATCHED training negatives (real class mix)
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    sc = np.zeros(len(tr)); k = min(5, tr.tile.nunique())
    for i, j in GroupKFold(k).split(tr, tr.label, tr.tile):
        sub = magnitude_match(tr.iloc[i], a.seed)
        sc[j] = fit(sub, a.seed).predict_proba(tr.iloc[j][SHAPE_FEATURES])[:, 1]
    thr = float(np.quantile(sc[tr.label.to_numpy() == 0], 1 - a.target_fpr))
    oof = float(roc_auc_score(tr.label, sc))
    clf = fit(trm, a.seed)
    print(f"[fit] {int(trm.label.sum())}/{int((1 - trm.label).sum())} matched; grouped-CV AUC {oof:.3f}; "
          f"threshold {thr:.3f} (OOF FPR target {a.target_fpr})", flush=True)
    import joblib
    joblib.dump(dict(model=clf, threshold=thr, fingerprint=man["fingerprint"]), os.path.join(a.out, "frozen.joblib"))
    print("[seal] model frozen; opening validation tiles", flush=True)
    seal.phase = "validation"; V = []
    for t in man["validation"]:
        d = featurize_tile(t, seal, "val_features", a.cache, a.workers, a.max_neg, a.seed); V.append(d)
        print(f"  val tile {t['id']} b={t['b']:.1f}: {int(d.label.sum())} var / {int((1 - d.label).sum())} quiet", flush=True)
    va = keep_visible(pd.concat(V, ignore_index=True))
    rep = report(clf, trm, va, thr, a.seed)          # unmatched validation: real magnitude mix
    vam = magnitude_match(va.assign(mean_mag=va.phot_g_mean_mag), a.seed)
    rep["matched"] = {k: v for k, v in report(clf, trm, vam, thr, a.seed, 3).items() if k != "recall_by_class"}
    rep.update(oof_auc=oof, fingerprint=man["fingerprint"], features=SHAPE_FEATURES)
    rep["fail_reasons"] = gate(rep, a.min_val); rep["validated"] = not rep["fail_reasons"]
    json.dump(rep, open(os.path.join(a.out, "validation_report.json"), "w"), indent=1, default=float)
    tr.to_parquet(os.path.join(a.out, "train_features.parquet"))
    va.to_parquet(os.path.join(a.out, "val_features.parquet"))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("features",)}, indent=1, default=float))


# ----------------------------------------------------------------- search ---
def short_period(args):
    """Short-period channel (periods 20 min to 2.4 h), outside the trained detector's 0.1 d floor. A star is kept
    when the periodogram peak is far beyond chance (log FAP < -10), the semi-amplitude is >= 4x the median error,
    the frequency is not a whole number of cycles per day, and each half of the light curve recovers it."""
    from astropy.timeseries import LombScargle
    t, m, e = (np.asarray(x, float) for x in args[:3])
    p_main = args[3] if len(args) > 3 else None     # significant long-period signal, if the main channel found one
    if len(t) < 80:
        return None
    med = np.median(m); mad = 1.4826 * np.median(np.abs(m - med)) + 1e-6
    k = np.abs(m - med) < 5 * mad + 3 * np.median(e); t, m, e = t[k], m[k], e[k]
    ls = LombScargle(t, m, e)
    f, p = ls.autopower(minimum_frequency=10.0, maximum_frequency=72.0, samples_per_peak=2, method="fast")
    j = int(np.argmax(p)); f0 = f[j]
    fap = ls.false_alarm_probability(p[j], minimum_frequency=10.0, maximum_frequency=72.0)
    if not (fap < 1e-10) or abs(f0 - round(f0)) < 0.02:
        return None
    fg = np.linspace(f0 * 0.999, f0 * 1.001, 201); f0 = fg[np.argmax(ls.power(fg))]
    amp = float(np.ptp(ls.model(np.linspace(0, 1 / f0, 50), f0)) / 2)
    if amp / np.median(e) < 3:
        return None
    if p_main:   # reject harmonics of a longer period (e.g. P/3, P/4 of a contact binary)
        r = p_main * f0
        if any(abs(r - n / d) < 0.003 * r for d in (1, 2, 3) for n in range(1, 60)):
            return None
    o = np.argsort(t)
    for half in np.array_split(o, 2):
        lh = LombScargle(t[half], m[half], e[half]); fh = np.linspace(f0 * 0.995, f0 * 1.005, 401); ph = lh.power(fh)
        if abs(fh[np.argmax(ph)] / f0 - 1) > 1e-3 or lh.false_alarm_probability(ph.max(), minimum_frequency=fh[0],
                                                                                 maximum_frequency=fh[-1]) > 1e-3:
            return None
    return dict(period=float(1 / f0), log_fap=float(np.log10(max(fap, 1e-300))), amp=amp,
                amp_over_err=float(amp / np.median(e)), n=int(len(t)), span=float(np.ptp(t)), mean_mag=float(med))


def cheap_stats(t, m, e):
    q75, q25 = np.percentile(m, [75, 25])
    return (q75 - q25) / np.median(e)


def run_search(a):
    import joblib
    from varwise_gaia_hunt import temporal_consistency_lc, temporal_consistency_ecl
    from prior_art import prior_art
    man = json.load(open(os.path.join(a.out, "manifest.json")))
    rep = json.load(open(os.path.join(a.out, "validation_report.json")))
    pack = joblib.load(os.path.join(a.out, "frozen.joblib"))
    if pack["fingerprint"] != man["fingerprint"] or rep["fingerprint"] != man["fingerprint"]:
        raise SystemExit("STOP: model, report and manifest disagree")
    if not rep["validated"] and not a.allow_unvalidated:
        raise SystemExit("STOP: model failed the gate: " + "; ".join(rep["fail_reasons"]))
    clf, thr = pack["model"], pack["threshold"]
    va = pd.read_parquet(os.path.join(a.out, "val_features.parquet"))
    cut = float(np.quantile(va[va.label == 1].iqr_over_err, 0.05))   # keeps 95% of sealed positives
    if a.extra_orders:
        have = {t["id"] for t in man["search"]}
        sealed = {t["id"] for t in man["train"] + man["validation"]}
        extra = extra_order_tiles(a.cache) + (high_latitude_tiles(a.cache) if a.search_bmax > 35 else [])
        man["search"] = man["search"] + [t for t in extra if t["id"] not in have and t["id"] not in sealed]
    seal = Seal(man, a.out); seal.phase = "search"
    st_p = os.path.join(a.out, "search_state.json")
    st = json.load(open(st_p)) if os.path.exists(st_p) else dict(done=[], scanned=0, scored=0)
    out_p = os.path.join(a.out, "search_hits.csv")
    t0 = time.time()
    pool_tiles = [t for t in man["search"] if t["b"] <= a.search_bmax]
    pool_tiles = [t for i, t in enumerate(pool_tiles) if i % a.nshards == a.shard]
    n_this = 0
    for tile in pool_tiles:
        if tile["id"] in st["done"]:
            continue
        if n_this >= a.max_tiles:
            break
        n_this += 1
        if time.time() - t0 > a.hours * 3600:
            break
        seal.check("search", tile)
        z = ztf_tile(tile, a.scratch)
        g = gaia_tile(tile, a.cache)
        tgt = g[g.vari_id.isna() & (g.phot_variable_flag != "VARIABLE") & (g.ruwe < 1.4)]
        m = crossmatch(z, tgt)
        zz = z.reset_index(drop=True)
        io = np.array([cheap_stats(zz.t.iat[i], zz.m.iat[i], zz.e.iat[i]) for i in m.zi.values])
        m = m[io >= cut]
        jobs = [(zz.t.iat[i], zz.m.iat[i], zz.e.iat[i]) for i in m.zi.values]
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            fs = pool.map(_feat, jobs, chunksize=20)
        rows = []
        for (k, r), f, zi in zip(m.iterrows(), fs, m.zi.values):
            if not f:
                continue
            s = float(clf.predict_proba(pd.DataFrame([f])[SHAPE_FEATURES])[:, 1][0])
            if s < thr or not (f["ls_pass"] or f["ecl_pass"]):
                continue
            lc = pd.DataFrame(dict(mjd=zz.t.iat[zi], mag=zz.m.iat[zi], magerr=zz.e.iat[zi]))
            ecl = f["ecl_pass"] and not f["ls_pass"]
            per = float(f["ecl_period"] if ecl else f["period"])
            tc = (temporal_consistency_ecl if ecl else temporal_consistency_lc)(lc, per)
            if not tc["confirmed"]:
                continue
            rec = dict(source_id=int(r.source_id), ra=r.ra, dec=r.dec, G=r.phot_g_mean_mag, bp_rp=r.bp_rp,
                       ztf_objectid=int(zz.objectid.iat[zi]), score=s, channel="eclipse" if ecl else "periodic",
                       period=per, tile=tile["id"], **{k2: f[k2] for k2 in ("amp", "amp_over_err", "log_fap",
                       "ecl_snr", "ecl_depth", "n", "span", "mean_mag", "skew")},
                       early_period=tc["early_period"], late_period=tc["late_period"])
            pa = prior_art(float(r.ra), float(r.dec), 5)
            nb = {}   # 30-arcsec neighbour check runs later, in dossier.py, for vetted stars only
            rec["prior_art"] = ";".join(sorted(pa)); rec["neighbour_30as"] = ";".join(sorted(nb))
            weak = {"GaiaDR2_LAV_Mowlavi2021", "TESS_10k_unvetted", "YSO_Delfini2025"}   # flags with no period or type
            rec["status"] = ("UNVERIFIED" if any(v == "ERROR" for v in pa.values()) else
                             "KNOWN" if set(pa) - weak else "NEW_CANDIDATE")
            rows.append(rec)
        if a.short_channel:
            hit_ids = {r["source_id"] for r in rows}
            sel = [(r, zi, f) for (k2, r), f, zi in zip(m.iterrows(), fs, m.zi.values)
                   if f and f.get("chi2red", 0) > 2.5 and int(r.source_id) not in hit_ids]
            with Pool(a.workers) as pool:
                sp = pool.map(short_period, [(zz.t.iat[zi], zz.m.iat[zi], zz.e.iat[zi],
                                             f["period"] if f["log_fap"] < -5 else None) for r, zi, f in sel], chunksize=10)
            for (r, zi, _), f in zip(sel, sp):
                if not f:
                    continue
                rec = dict(source_id=int(r.source_id), ra=r.ra, dec=r.dec, G=r.phot_g_mean_mag, bp_rp=r.bp_rp,
                           ztf_objectid=int(zz.objectid.iat[zi]), score=np.nan, channel="short", period=f["period"],
                           tile=tile["id"], amp=f["amp"], amp_over_err=f["amp_over_err"], log_fap=f["log_fap"],
                           n=f["n"], span=f["span"], mean_mag=f["mean_mag"])
                pa = prior_art(float(r.ra), float(r.dec), 5); nb = {}
                rec["prior_art"] = ";".join(sorted(pa)); rec["neighbour_30as"] = ";".join(sorted(nb))
                weak = {"GaiaDR2_LAV_Mowlavi2021", "TESS_10k_unvetted", "YSO_Delfini2025"}
                rec["status"] = ("UNVERIFIED" if any(v == "ERROR" for v in pa.values()) else
                                 "KNOWN" if set(pa) - weak else "NEW_CANDIDATE")
                rows.append(rec)
        if rows:
            pd.DataFrame(rows).to_csv(out_p, mode="a", header=not os.path.exists(out_p), index=False)
        st["done"].append(tile["id"]); st["scanned"] += len(tgt); st["scored"] += len(jobs)
        json.dump(st, open(st_p, "w"))
        os.remove(os.path.join(a.scratch, f"ztf_{tile['id']}.pkl"))
        nnew = sum(r["status"] == "NEW_CANDIDATE" for r in rows)
        print(f"  search tile {tile['id']} b={tile['b']:.1f}: {len(tgt)} Gaia-quiet, {len(jobs)} featurized, "
              f"{len(rows)} confirmed hits, {nnew} new  ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "search"])
    ap.add_argument("--out", default="runs/realsky1"); ap.add_argument("--cache", default="cache")
    ap.add_argument("--train-tiles", type=int, default=8); ap.add_argument("--val-tiles", type=int, default=4)
    ap.add_argument("--bmin", type=float, default=15); ap.add_argument("--bmax", type=float, default=35)
    ap.add_argument("--max-neg", type=int, default=1500); ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--min-val", type=int, default=200); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--scratch", default="/home/claude/work/scratch"); ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--search-bmax", type=float, default=35); ap.add_argument("--allow-unvalidated", action="store_true")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-tiles", type=int, default=10 ** 6)
    ap.add_argument("--extra-orders", action="store_true", help="also search order-4 and order-6 partitions")
    ap.add_argument("--short-channel", action="store_true", help="also search 20 min - 2.4 h periods")
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    run_train(a) if a.cmd == "train" else run_search(a)
