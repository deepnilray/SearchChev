"""dr4engine.py - whole-archive search of Gaia epoch photometry (built on DR3, ready for DR4 on 2 Dec 2026).

One unit of work is one HEALPix range file set from the Gaia CDN: EpochPhotometry, GaiaSource, VariSummary and
VariClassifierResult share identical level-8 ranges, so each range is processed alone on one GitHub runner.

Searches, all run in the same pass over every light curve:
  dips      single long, deep, non-periodic dimming episodes on otherwise quiet stars
            (occulting disks and ring systems, J1407-like events, dust clouds)
  impact    long irregular dimming episodes on stable stars: giant-impact debris candidates (ASASSN-21qj-like),
            later checked for an infrared brightening in NEOWISE (ir_check.py)
  step      a large lasting brightness change, flat on both sides (one edge of a dimming longer than the baseline)
  anomaly   light curves far from everything else in feature space (IsolationForest, per file, top few)
  missed    shape classifier trained on Gaia-flagged variables versus unflagged stars in sealed training files,
            gated on sealed validation files, then applied to unflagged stars in search files (needs DR4:
            DR3 published light curves only for variables, so DR3 has no negatives)
Every candidate keeps the evidence needed to check it (light-curve points, neighbours, flags).

Phases are fixed in advance by a hash of the file range, never by results: 8% train, 4% validation, 88% search.

usage:
  python dr4engine.py discover --release gdr3            # list ranges and products on the CDN
  python dr4engine.py run --release gdr3 --shard 0 --nshards 20 --out work --max-files 1
  python dr4engine.py file --release gdr3 --range 000000-003111 --out work      # one range, local test
"""
import argparse, gzip, hashlib, io, json, math, os, re, sys, time, warnings
import numpy as np, pandas as pd, requests

warnings.filterwarnings("ignore")
BUCKET = "https://gaia.eu-1.cdn77-storage.com/"
CDN = "https://cdn.gea.esac.esa.int/"
PRODUCTS = {"epoch": "EpochPhotometry", "source": "GaiaSource", "vsum": "VariSummary", "vcls": "VariClassifierResult"}
NAME_RX = re.compile(r"^([A-Za-z]+)_(\d+)-(\d+)\.(csv|ecsv|parquet|fits|vot)(\.gz)?$")
GAIA_T0_JD = 2455197.5
SOURCE_COLS = ["source_id", "ra", "dec", "parallax", "parallax_error", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp",
               "ruwe", "phot_variable_flag", "duplicated_source", "ipd_frac_multi_peak", "phot_bp_rp_excess_factor",
               "phot_g_n_obs", "non_single_star"]
# Epoch-photometry column candidates; DR4 names may differ, so every role lists alternatives.
EPOCH_ROLES = {
    "sid": ["source_id"],
    "g_t": ["g_transit_time", "g_obs_time", "obs_time_g", "time_g", "g_time"],
    "g_m": ["g_transit_mag", "g_mag", "mag_g"],
    "g_snr": ["g_transit_flux_over_error", "g_flux_over_error", "flux_over_error_g"],
    "g_f": ["g_transit_flux", "g_flux", "flux_g"], "g_fe": ["g_transit_flux_error", "g_flux_error", "flux_error_g"],
    "bp_t": ["bp_obs_time", "bp_transit_time", "time_bp"], "bp_m": ["bp_mag", "bp_transit_mag", "mag_bp"],
    "rp_t": ["rp_obs_time", "rp_transit_time", "time_rp"], "rp_m": ["rp_mag", "rp_transit_mag", "mag_rp"],
    "rej": ["variability_flag_g_reject", "rejected_by_variability"], "rejp": ["rejected_by_photometry"],
    # long format (one row per transit and band)
    "time": ["obs_time", "time", "transit_time"], "mag": ["mag"], "snr": ["flux_over_error"],
    "flux": ["flux"], "flux_err": ["flux_error"], "band": ["band", "passband"],
}
G_MAX = 20.5          # brightest-decile G; the catalogue mean is not used, since a star dimmed by an event reads faint
PHASES = (("train", 8), ("validation", 4), ("search", 88))


# ------------------------------------------------------------------ CDN discovery -----------------------
def list_keys(prefix):
    keys, marker = [], ""
    while True:
        r = requests.get(BUCKET, params=dict(prefix=prefix, marker=marker), timeout=120); r.raise_for_status()
        ks = re.findall(r"<Key>([^<]+)</Key>", r.text); keys += ks
        if "<IsTruncated>true</IsTruncated>" not in r.text or not ks:
            return keys
        nm = re.search(r"<NextMarker>([^<]+)</NextMarker>", r.text); marker = nm.group(1) if nm else ks[-1]


def discover(release, cache="cache"):
    """{range: {product: key}} for every range that has epoch photometry. Cached to cache/ranges_<release>.json."""
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, f"ranges_{release}.json")
    if os.path.exists(p):
        return json.load(open(p))
    found = {}
    for k in list_keys(f"Gaia/{release}/"):
        m = NAME_RX.match(os.path.basename(k))
        if not m:
            continue
        prod = next((n for n, v in PRODUCTS.items() if m.group(1).lower() == v.lower()), None)
        if prod:
            found.setdefault(f"{m.group(2)}-{m.group(3)}", {})[prod] = k
    out = {r: v for r, v in found.items() if "epoch" in v}
    json.dump(out, open(p, "w"))
    return out


def healpix8(source_id):
    """HEALPix level-8 (nested) index encoded in a Gaia source_id; the bulk-file ranges use this index."""
    return int(source_id) // (2 ** 35) // (4 ** 4)


def phase_of(rng, seed="dr4engine-20261124"):
    h = int(hashlib.sha256(f"{seed}:{rng}".encode()).hexdigest(), 16) % 100
    lo = 0
    for name, pct in PHASES:
        if h < lo + pct:
            return name
        lo += pct


def fetch(key, dest, tries=5):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    for k in range(tries):
        try:
            with requests.get(CDN + key, stream=True, timeout=600) as r:
                r.raise_for_status()
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for ch in r.iter_content(4 << 20):
                        f.write(ch)
            os.replace(tmp, dest); return dest
        except Exception as ex:
            err = ex; time.sleep(15 * (k + 1))
    raise RuntimeError(f"download failed {key}: {err}")


def read_table(path, cols=None, chunksize=None):
    """csv / ecsv (# comment header) / parquet, gzipped or not; only `cols` that exist are read."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        have = pq.ParquetFile(path).schema.names
        return pd.read_parquet(path, columns=[c for c in (cols or have) if c in have])
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        for line in fh:
            if not line.startswith("#"):
                header = line.strip().split(","); break
    use = [c for c in cols if c in header] if cols else None
    return pd.read_csv(path, comment="#", usecols=use, dtype=str if chunksize else None, chunksize=chunksize,
                       low_memory=False)


def epoch_header(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        for line in fh:
            if not line.startswith("#"):
                return line.strip().split(",")
    return []


def resolve(header):
    h = set(header)
    return {role: next((c for c in cands if c in h), None) for role, cands in EPOCH_ROLES.items()}


# ------------------------------------------------------------------ light curves ------------------------
def _arr(s):
    if not isinstance(s, str) or len(s) < 3:
        return np.empty(0)
    return np.fromstring(s.strip()[1:-1].replace("null", "nan").replace('"', ""), sep=",")


def _barr(s, n):
    if not isinstance(s, str) or len(s) < 3:
        return np.zeros(n, bool)
    a = np.array([v.strip().lower() == "true" for v in s.strip()[1:-1].split(",")])
    return a if a.size == n else np.zeros(n, bool)


def light_curves(path, min_points=15):
    """Yield (source_id, dict band -> (t_bjd, mag, magerr)). Handles wide (array per source) and long layouts."""
    header = epoch_header(path); R = resolve(header)
    if R["g_t"] and (R["g_m"] or R["g_f"]):
        cols = [c for c in {R[k] for k in ("sid", "g_t", "g_m", "g_snr", "g_f", "g_fe", "bp_t", "bp_m", "rp_t", "rp_m",
                                           "rej", "rejp")} if c]
        for chunk in read_table(path, cols, chunksize=4000):
            for row in chunk.to_dict("records"):
                out = {}
                t = _arr(row.get(R["g_t"]))
                if R["g_m"]:
                    m = _arr(row.get(R["g_m"]))
                else:
                    f = _arr(row.get(R["g_f"])); m = np.where(f > 0, -2.5 * np.log10(np.where(f > 0, f, 1)) + 25.6874, np.nan)
                if R["g_snr"]:
                    snr = _arr(row.get(R["g_snr"]))
                else:
                    snr = _arr(row.get(R["g_f"])) / _arr(row.get(R["g_fe"]))
                n = min(t.size, m.size, snr.size); t, m, snr = t[:n], m[:n], snr[:n]
                bad = np.zeros(n, bool)
                for r in ("rej", "rejp"):
                    if R[r]:
                        bad |= _barr(row.get(R[r]), n)
                ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(snr) & (snr > 0) & ~bad
                if ok.sum() >= min_points:
                    out["G"] = (t[ok] + GAIA_T0_JD, m[ok], 1.0857 / snr[ok])
                for b in ("bp", "rp"):
                    if R[f"{b}_t"] and R[f"{b}_m"]:
                        tb, mb = _arr(row.get(R[f"{b}_t"])), _arr(row.get(R[f"{b}_m"]))
                        k = min(tb.size, mb.size); tb, mb = tb[:k], mb[:k]; okb = np.isfinite(tb) & np.isfinite(mb)
                        if okb.sum() >= min_points:
                            out[b.upper()] = (tb[okb] + GAIA_T0_JD, mb[okb], None)
                if "G" in out:
                    yield int(row[R["sid"]]), out
        return
    if R["time"] and (R["mag"] or R["flux"]):                                 # long format
        d = read_table(path, [c for c in {R[k] for k in ("sid", "time", "mag", "snr", "flux", "flux_err", "band")} if c])
        d = d.rename(columns={R["sid"]: "sid", R["time"]: "t"})
        if R["mag"]:
            d["m"] = pd.to_numeric(d[R["mag"]], errors="coerce")
        else:
            f = pd.to_numeric(d[R["flux"]], errors="coerce"); d["m"] = -2.5 * np.log10(f.where(f > 0)) + 25.6874
        snr = (pd.to_numeric(d[R["snr"]], errors="coerce") if R["snr"] else
               pd.to_numeric(d[R["flux"]], errors="coerce") / pd.to_numeric(d[R["flux_err"]], errors="coerce"))
        d["e"] = 1.0857 / snr; d["t"] = pd.to_numeric(d["t"], errors="coerce") + GAIA_T0_JD
        d["b"] = d[R["band"]].astype(str).str.upper().str.replace("PHOT_", "") if R["band"] else "G"
        d = d[np.isfinite(d.t) & np.isfinite(d.m)]
        for sid, g in d.groupby("sid", sort=False):
            out = {}
            for b, gb in g.groupby("b"):
                gb = gb[np.isfinite(gb.e) & (gb.e > 0)] if b == "G" else gb
                if len(gb) >= min_points:
                    out[b] = (gb.t.values, gb.m.values, gb.e.values if b == "G" else None)
            if "G" in out:
                yield int(sid), out
        return
    raise RuntimeError("UNKNOWN_EPOCH_SCHEMA: " + ",".join(header[:80]))


# ------------------------------------------------------------------ features ----------------------------
def visits(t, gap=1.0):
    """Index groups of points closer than `gap` days (Gaia field-of-view pairs and same-week revisits)."""
    if t.size == 0:
        return []
    return np.split(np.arange(t.size), np.where(np.diff(t) > gap)[0] + 1)


def _nearest(sorted_t, x):
    """Index of the nearest element of sorted_t for every x."""
    o = np.argsort(sorted_t); st = sorted_t[o]
    j = np.searchsorted(st, x).clip(1, st.size - 1)
    j = np.where(np.abs(st[j - 1] - x) <= np.abs(st[j] - x), j - 1, j)
    return o[j]


def features(lc):
    t, m, e = lc["G"]
    o = np.argsort(t); t, m, e = t[o], m[o], e[o]
    n = t.size; med = np.median(m); mad = 1.4826 * np.median(np.abs(m - med))
    jit = max(mad, float(np.median(e)))
    z = (m - med) / np.sqrt(e ** 2 + 0.25 * jit ** 2 + 1e-8)
    q5, q25, q75, q95 = np.percentile(m, [5, 25, 75, 95])
    f = dict(n=n, span=float(t[-1] - t[0]), med_mag=float(med), mad=float(mad), med_err=float(np.median(e)),
             iqr_over_err=float((q75 - q25) / np.median(e)), amp90=float(q95 - q5),
             chi2red=float(np.sum(((m - np.average(m, weights=1 / e ** 2)) / e) ** 2) / max(n - 1, 1)),
             skew=float(pd.Series(m).skew()), kurt=float(pd.Series(m).kurt()),
             vonneumann=float(np.sum(np.diff(m) ** 2) / max(np.sum((m - m.mean()) ** 2), 1e-12)))
    # trend
    if f["span"] > 30:
        A = np.vstack([np.ones(n), (t - t.mean()) / 365.25]).T
        w = 1 / np.sqrt(e ** 2 + mad ** 2 + 1e-6)
        b, res, *_ = np.linalg.lstsq(A * w[:, None], m * w, rcond=None)
        cov = np.linalg.pinv((A * w[:, None]).T @ (A * w[:, None])); f["trend"] = float(b[1]); f["trend_sig"] = float(b[1] / math.sqrt(max(cov[1, 1], 1e-12)))
    else:
        f["trend"] = f["trend_sig"] = 0.0
    # periodogram, only for stars that vary at all (saves most of the time)
    f.update(period=np.nan, ls_power=0.0, log_fap=0.0)
    if (f["chi2red"] > 2.5 or f["iqr_over_err"] > 2) and n >= 20:
        from astropy.timeseries import LombScargle
        ls = LombScargle(t, m, e)
        fmin, fmax = 2.0 / max(f["span"], 1.0), 12.0          # periods 2 h to span/2
        try:
            fr, pw = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=2, method="cython")
            j = int(np.nanargmax(pw)); df = fr[1] - fr[0]
            fz = np.linspace(max(fmin, fr[j] - 2 * df), fr[j] + 2 * df, 81); pz = ls.power(fz, method="cython")
            k = int(np.nanargmax(pz)); f["period"] = float(1 / fz[k]); f["ls_power"] = float(pz[k])
            f["log_fap"] = float(np.log10(max(ls.false_alarm_probability(pz[k], minimum_frequency=fmin,
                                                                         maximum_frequency=fmax), 1e-300)))
        except Exception:
            pass
    # dimming and brightening episodes
    for sign, tag in ((1, "dip"), (-1, "bri")):
        sel = sign * z > 5
        f[f"{tag}_n"] = int(sel.sum()); f[f"{tag}_depth"] = float(sign * (m[sel] - med).max()) if sel.any() else 0.0
        best = (0, 0.0, 0.0, 0, 0.0)            # points, duration, depth, visits, centre
        if sel.sum() >= 2:
            idx = np.where(sel)[0]
            # an episode: consecutive faint points not separated by a normal point
            groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for g in groups:
                nv = len(visits(t[g]))
                cand = (g.size, float(t[g[-1]] - t[g[0]]), float(sign * (m[g] - med).max()), nv, float(t[g].mean()))
                if (cand[0], cand[3]) > (best[0], best[3]):
                    best = cand
        f[f"{tag}_ep_n"], f[f"{tag}_ep_dur"], f[f"{tag}_ep_depth"], f[f"{tag}_ep_visits"], f[f"{tag}_ep_t"] = best
        f[f"{tag}_frac"] = float(sel.mean())
    # evidence that a dimming lasted: consecutive faint observations >= 1 d (and <= 200 d) apart with nothing normal in between, and how
    # many normal observations sit within 3 d of the faint ones (eclipsing binaries: many; long dimmings: few)
    fz = z > 5
    pairs = fz[:-1] & fz[1:]; dts = np.diff(t)[pairs]
    f["dip_proven_days"] = float(dts[dts <= 200].max()) if (dts <= 200).any() else 0.0
    if fz.any():
        near = np.abs(t[None, :] - t[fz][:, None]) < 3
        nn = int((near & ~fz[None, :]).sum()); fn = int((near & fz[None, :]).sum() - fz.sum())
        f["dip_near_normal_frac"] = nn / max(nn + fn, 1)
    else:
        f["dip_near_normal_frac"] = 1.0
    # largest step: one split in time with both sides flat (the start or the end of a dimming that lasts longer than the
    # baseline, e.g. Gaia17bpp's 7-year eclipse ending in 2018, or a debris cloud arriving)
    f.update(step=0.0, step_sig=0.0, step_t=np.nan, step_flat=np.inf)
    if n >= 12:
        cs, cs2 = np.cumsum(m), np.cumsum(m ** 2)
        k = np.arange(5, n - 5)
        if k.size:
            mu1 = cs[k - 1] / k; mu2 = (cs[-1] - cs[k - 1]) / (n - k)
            v1 = cs2[k - 1] / k - mu1 ** 2; v2 = (cs2[-1] - cs2[k - 1]) / (n - k) - mu2 ** 2
            sd = np.sqrt(np.clip(v1, 0, None) / k + np.clip(v2, 0, None) / (n - k) + np.median(e) ** 2 * (1 / k + 1 / (n - k)))
            sig = np.abs(mu2 - mu1) / sd; j = int(np.argmax(sig))
            f.update(step=float(mu2[j] - mu1[j]), step_sig=float(sig[j]), step_t=float(t[k[j]]),
                     step_flat=float(np.sqrt(max(v1[j], v2[j], 0))))
    # scatter outside the strongest dimming episode
    sel = z > 5
    f["quiet_mad"] = float(1.4826 * np.median(np.abs(m[~sel] - np.median(m[~sel])))) if (~sel).sum() > 5 else float(mad)
    # colour during dims (dust reddens; occulting bodies are grey): BP-RP at dip transits vs elsewhere
    f["dip_dcolour"] = np.nan
    if "BP" in lc and "RP" in lc and sel.any():
        tb, mb, _ = lc["BP"]; tr, mr, _ = lc["RP"]
        if tb.size and tr.size:
            jr = _nearest(tr, tb); ok = np.abs(tr[jr] - tb) < 0.01
            if ok.sum() > 5:
                tc, col = tb[ok], mb[ok] - mr[jr[ok]]
                jd = _nearest(t, tc); near = np.abs(t[jd] - tc) < 0.01
                dipc = near & sel[jd]
                if dipc.sum() >= 2 and (~dipc).sum() >= 5:
                    f["dip_dcolour"] = float(np.median(col[dipc]) - np.median(col[~dipc]))
    return f


# ------------------------------------------------------------------ searches ----------------------------
def main_sequence_like(F):
    """Exclude giants and very red stars (long-period variables) from the giant-impact list: Gaia absolute G > 2 when
    the parallax is good, otherwise BP-RP < 2.5."""
    good = (F.parallax / F.parallax_error) > 5
    mg = F.phot_g_mean_mag + 5 * np.log10(F.parallax.clip(lower=1e-3) / 100)
    return F[(good & (mg > 2)) | (~good & (F.bp_rp < 2.5))]



NOT_STELLAR = {"AGN", "GALAXY", "QSO"}


def stellar(F):
    """Keep stars: significant parallax or proper motion, and not classed extragalactic by Gaia.
    Quasars and galaxies vary stochastically and would otherwise fill the rare-event lists."""
    plx = F.parallax / F.parallax_error
    pm = np.hypot(F.pmra.fillna(0), F.pmdec.fillna(0))
    return F[((plx >= 3) | (pm >= 5)) & ~F.gaia_class.fillna("").str.upper().isin(NOT_STELLAR)]


def _rank(c):
    """Unflagged (unknown) stars first, then by score."""
    c["known_variable"] = c.flagged_variable.fillna(False)
    return c.sort_values(["known_variable", "score"], ascending=[True, False])


def search_dips(F):
    """Single long deep dimming episode on an otherwise quiet star. With sparse Gaia sampling a periodogram aliases a
    long dip into a short period, so periodicity is not used here; eclipsing-binary impostors are reduced by requiring
    no normal points near the faint ones and are removed by the ZTF check in followup.py."""
    F = main_sequence_like(stellar(F))
    c = F[(F.dip_proven_days >= 1) & (F.dip_near_normal_frac < 0.5) & (F.dip_ep_n >= 3) & (F.dip_ep_visits >= 2) & (F.dip_ep_depth >= 0.10) &
          (F.quiet_mad < np.maximum(0.04, F.dip_ep_depth / 4)) & (F.bri_n <= 1) &
          (F.dip_frac < 0.6)].copy()
    c["score"] = c.dip_ep_depth * np.sqrt(c.dip_ep_n) / np.maximum(c.quiet_mad, 0.005)
    return _rank(c)


def search_impact(F):
    """Long irregular dimming lasting >= 30 d with many faint points on a star quiet outside it.
    Main contaminants (YSO dippers, R CrB stars, long-period variables) are labelled, not hidden."""
    F = main_sequence_like(stellar(F))
    c = F[(F.dip_proven_days >= 1) & (F.dip_near_normal_frac < 0.5) & (F.dip_ep_n >= 5) & (F.dip_ep_visits >= 3) & (F.dip_ep_depth >= 0.2) & (F.dip_ep_dur >= 30) & (F.quiet_mad < 0.05)].copy()
    c["score"] = c.dip_depth * np.sqrt(c.dip_n) * np.log10(c.dip_ep_dur.clip(lower=1))
    return _rank(c)


def search_step(F):
    """A large lasting change of brightness, flat on both sides: one edge of a dimming longer than the baseline."""
    F = stellar(F)
    c = F[(F.step.abs() >= 0.3) & (F.step_sig >= 15) & (F.step_flat < np.maximum(0.05, F.step.abs() / 5))].copy()
    c["score"] = c.step.abs() * np.log10(c.step_sig)
    return _rank(c)


ANOMALY_FEATURES = ["iqr_over_err", "amp90", "chi2red", "skew", "kurt", "vonneumann", "trend", "ls_power", "log_fap",
                    "dip_frac", "bri_frac", "dip_ep_dur", "bri_ep_dur", "dip_depth", "bri_depth", "quiet_mad", "step", "step_sig"]


def search_anomaly(F, top=10):
    from sklearn.ensemble import IsolationForest
    X = F[ANOMALY_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).values
    if len(X) < 200:
        return F.iloc[:0].assign(score=[])
    X = np.sign(X) * np.log1p(np.abs(X))
    s = -IsolationForest(n_estimators=200, random_state=0).fit(X).score_samples(X)
    c = F.assign(score=s).sort_values("score", ascending=False)
    return c[(c.chi2red > 3) | (c.iqr_over_err > 2)].head(top)


# ------------------------------------------------------------------ sources, labels, neighbours ---------
def source_table(files, release):
    s = read_table(files["source"], SOURCE_COLS) if "source" in files else pd.DataFrame(columns=SOURCE_COLS)
    for c in SOURCE_COLS:
        if c not in s:
            s[c] = np.nan
    s["source_id"] = s.source_id.astype("int64")
    num = [c for c in SOURCE_COLS if c not in ("source_id", "phot_variable_flag", "duplicated_source")]
    s[num] = s[num].apply(pd.to_numeric, errors="coerce")
    s["flagged_variable"] = s.phot_variable_flag.astype(str).str.upper().eq("VARIABLE")
    if "vsum" in files:
        v = read_table(files["vsum"], ["source_id"]); s["flagged_variable"] |= s.source_id.isin(v.source_id.astype("int64"))
    s["gaia_class"] = ""
    if "vcls" in files:
        v = read_table(files["vcls"], ["source_id", "best_class_name", "best_class_score"])
        v["source_id"] = v.source_id.astype("int64"); v = v.drop_duplicates("source_id")
        s = s.merge(v[["source_id", "best_class_name"]], on="source_id", how="left")
        s["gaia_class"] = s.best_class_name.fillna("").astype(str); s = s.drop(columns="best_class_name")
    # neighbours inside the same range: brighter star within 3" or any within 1.5" can fake variability
    from scipy.spatial import cKDTree
    ok = np.isfinite(s.ra) & np.isfinite(s.dec)
    xyz = np.c_[np.cos(np.radians(s.dec[ok])) * np.cos(np.radians(s.ra[ok])),
                np.cos(np.radians(s.dec[ok])) * np.sin(np.radians(s.ra[ok])), np.sin(np.radians(s.dec[ok]))]
    tree = cKDTree(xyz); r3 = 2 * math.sin(math.radians(3 / 3600) / 2)
    pairs = tree.query_pairs(r3, output_type="ndarray")
    g = s.phot_g_mean_mag[ok].values; sid = s.source_id[ok].values
    flag = np.zeros(ok.sum(), bool)
    for a, b in pairs:
        if g[b] < g[a] + 1 or not np.isfinite(g[b]):
            flag[a] = True
        if g[a] < g[b] + 1 or not np.isfinite(g[a]):
            flag[b] = True
    s["neighbour_3as"] = False; s.loc[ok, "neighbour_3as"] = flag
    return s


# ------------------------------------------------------------------ one range ---------------------------
WORKERS = max(1, (os.cpu_count() or 2))
MISSED_FEATURES = ["iqr_over_err", "amp90", "chi2red", "skew", "kurt", "vonneumann", "ls_power", "log_fap", "log_period",
                   "trend_sig", "dip_frac", "bri_frac", "quiet_over_err"]


def _feature_job(item):
    sid, lc = item
    try:
        f = features(lc)
    except Exception:
        return sid, None, None
    keep = f["dip_n"] >= 3 or f["bri_n"] >= 3 or f["chi2red"] > 3       # light curves worth saving as evidence
    return sid, f, (lc["G"] if keep else None)


def missed_matrix(F):
    X = F.copy()
    X["log_period"] = np.log10(X.period.where(X.period > 0))
    X["quiet_over_err"] = X.quiet_mad / X.med_err.clip(lower=1e-4)
    return X[MISSED_FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)


def load_model(path):
    import joblib
    return joblib.load(path) if path and os.path.exists(path) else None

def process_range(rng, files, release, out, keep_lc=True, model=None):
    """Download one range, compute features for every light curve, run every search, write results."""
    t0 = time.time(); tmp = os.path.join(out, "tmp"); os.makedirs(tmp, exist_ok=True)
    local = {p: fetch(k, os.path.join(tmp, os.path.basename(k))) for p, k in files.items()}
    src = source_table(local, release).set_index("source_id")
    rows, lcs = [], {}
    gmag = src.phot_g_mean_mag

    def batches():
        b = []
        for sid, lc in light_curves(local["epoch"]):
            if np.percentile(lc["G"][1], 10) > G_MAX:     # brightest state, so stars caught dim are kept
                continue
            b.append((sid, lc))
            if len(b) == 2000:
                yield b; b = []
        if b:
            yield b
    from multiprocessing import Pool
    with Pool(WORKERS) as pool:
        for b in batches():
            for sid, f, g in pool.map(_feature_job, b, chunksize=50):
                if f is None:
                    continue
                f["source_id"] = sid; rows.append(f)
                if keep_lc and g is not None:
                    lcs[sid] = g
    F = pd.DataFrame(rows)
    if F.empty:
        return dict(range=rng, n=0)
    F = F.join(src, on="source_id")
    F["phase"] = phase_of(rng)
    d = os.path.join(out, "ranges", rng); os.makedirs(d, exist_ok=True)
    res = dict(range=rng, phase=F.phase.iat[0], n=len(F), n_flagged=int(F.flagged_variable.fillna(False).sum()))
    clean = F[~F.neighbour_3as.fillna(False) & (F.duplicated_source.astype(str).str.lower() != "true")]
    for name, fn in (("dips", search_dips), ("impact", search_impact), ("step", search_step), ("anomaly", search_anomaly)):
        c = fn(clean).head(200)
        c = c.head(50)
        c.to_csv(os.path.join(d, f"{name}.csv"), index=False); res[name] = len(c)
        if keep_lc and len(c):
            np.savez_compressed(os.path.join(d, f"{name}_lc.npz"),
                                **{str(s): np.vstack(lcs[s]) for s in c.source_id if s in lcs})
    # compact sample for the 'missed' classifier: every flagged star (capped) and 3x as many unflagged, train/val only
    if res["phase"] in ("train", "validation"):
        pos = F[F.flagged_variable.fillna(False)]; neg = F[~F.flagged_variable.fillna(False)]
        smp = pd.concat([pos.sample(min(len(pos), 400), random_state=0), neg.sample(min(len(neg), 1200), random_state=0)])
        smp.to_parquet(os.path.join(d, "sample.parquet"))
    else:
        # search files: keep features of unflagged stars that vary at all, for scoring by the frozen model
        unf = clean[~clean.flagged_variable.fillna(False) & ((clean.chi2red > 2.5) | (clean.iqr_over_err > 2))]
        res["unflagged_varying"] = len(unf)
        if model is not None and len(unf):
            sc = model["model"].predict_proba(missed_matrix(unf))[:, 1]
            hit = unf.assign(score=sc)[(sc >= model["threshold"]) & ((unf.ls_power > 0) | (unf.dip_n >= 3))]
            keep = ["source_id", "ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax", "period", "ls_power", "log_fap",
                    "amp90", "chi2red", "n", "score"]
            hit[[c for c in keep if c in hit]].round(6).to_csv(os.path.join(d, "missed.csv.gz"), index=False)
            res["missed"] = len(hit)
    res["seconds"] = round(time.time() - t0); json.dump(res, open(os.path.join(d, "summary.json"), "w"))
    for p in local.values():
        os.remove(p)
    return res


def run(a):
    ranges = discover(a.release, a.cache)
    keys = sorted(ranges)
    mine = [r for i, r in enumerate(keys) if i % a.nshards == a.shard]
    done = set()
    if a.done_from and os.path.isdir(a.done_from):
        for f in os.listdir(a.done_from):
            done.add(f)
    t_end = time.time() + a.hours * 3600; n = 0
    model = load_model(a.model)
    phases = set(a.phases.split(","))
    for r in mine:
        if phase_of(r) not in phases:
            continue
        if r in done or os.path.exists(os.path.join(a.out, "ranges", r, "summary.json")):
            continue
        if time.time() > t_end or n >= a.max_files:
            break
        try:
            res = process_range(r, ranges[r], a.release, a.out, model=model)
            print(json.dumps(res), flush=True)
        except Exception as ex:
            msg = str(ex)
            print(f"RANGE {r} FAILED: {msg[:300]}", flush=True)
            if msg.startswith("UNKNOWN_EPOCH_SCHEMA"):
                os.makedirs(a.out, exist_ok=True); open(os.path.join(a.out, "SCHEMA_ALERT.txt"), "w").write(msg)
                raise
        n += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["discover", "run", "file"])
    ap.add_argument("--release", default="gdr3"); ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="work"); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--max-files", type=int, default=10 ** 9); ap.add_argument("--range")
    ap.add_argument("--done-from", default=None)
    ap.add_argument("--phases", default="train,validation,search")
    ap.add_argument("--model", default="model/missed.joblib")
    a = ap.parse_args()
    if a.cmd == "discover":
        r = discover(a.release, a.cache)
        prods = pd.Series([p for v in r.values() for p in v]).value_counts().to_dict()
        print(len(r), "ranges;", prods, "; phases:", pd.Series([phase_of(k) for k in r]).value_counts().to_dict())
    elif a.cmd == "file":
        r = discover(a.release, a.cache)
        print(json.dumps(process_range(a.range, r[a.range], a.release, a.out, model=load_model(a.model))))
    else:
        run(a)
