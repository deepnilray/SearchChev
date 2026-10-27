#!/usr/bin/env python3
"""
varwise_gaia_hunt.py  -  find optically periodic variable stars that are in the
VarWISE infrared-variable catalogue but in NO optical variable catalogue.

RUN ON YOUR PC (needs internet):
    pip install astropy scikit-learn pandas scipy matplotlib joblib pyvo astroquery requests
    python varwise_gaia_hunt.py selftest                 # offline logic test, ~2 min
    python varwise_gaia_hunt.py brutetest --out stress   # unseen signal shapes (synthetic)
    python varwise_gaia_hunt.py probe                    # tests every archive + shows VarWISE table/columns
    python varwise_gaia_hunt.py smoke --out smoke        # every stage on a tiny box, PASS/FAIL per stage
    python varwise_gaia_hunt.py auto --out run1        # EVERYTHING: preflight, download, train, validate, search, resume
    python varwise_gaia_hunt.py train --ra0 150 --dec0 30 --size 4 --out out_150_30   # trains + saves model.joblib
    python varwise_gaia_hunt.py run --ra0 150 --dec0 30 --size 4 --out out_150_30
    python varwise_gaia_hunt.py run --search-mode signal --skip-gaia --out optical_no_labels

Pipeline (what "run" does)
  1. VarWISE objects in a sky box (IRSA TAP; table found automatically).
  2. Gaia DR3 counterpart within 3" (Gaia TAP); keep G 12-19, unique match.
     Flag whether Gaia already has a variability solution (gaiadr3.vari_summary).
  3. ZTF light curves (IRSA ZTF API), Lomb-Scargle features per source.
  4. When possible, 80/20 model: positives = Gaia-known variables; proxy
     negatives = Gaia objects lacking a published variability solution, NOT
     confirmed constant stars. Sky cells are held out, not individual rows.
     No target source can be used as a proxy negative before discovery.
  5. Score VarWISE sources without Gaia variability solutions. Without enough
     external Gaia training labels, use UNTRAINED SIGNAL mode; no ML scores.
     Both modes require period consistency in two independent temporal halves.
  6. Prior-art cut on survivors: VSX, ASAS-SN, ZTF periodic (Chen 2020), SIMBAD.
  7. Writes candidates.csv, unverified.csv, folded PNGs; model if trained.

v3 additions (verified offline on simulated data only; see brutetest):
  * eclipse channel: Lomb-Scargle misses narrow eclipses (its outlier clip removes them), so a
    box-fit channel with robust gates runs in parallel; 0 false passes in 600 noise-only curves
    (white, under-estimated errors, red noise, 3% bad epochs).
  * varwise_period_agrees: compares the optical period with VarWISE's own period1.

Status of claims: selftest and brutetest measure SIMULATED signals, not real-sky
performance. `probe`/`smoke` test actual archive access. Source entries are
ALREADY in the VarWISE catalogue; candidates are never confirmed discoveries.
Nothing here submits or posts anything anywhere.
"""
import argparse, os, sys, time, json, hashlib, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
ZTF_API = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
MATCH_ARCSEC = 3.0

FEATURES = ["log_fap", "power", "amp", "amp_over_err", "chi2red", "log_period",
            "log_n", "mean_mag", "skew", "iqr_over_err"]

# ---------------------------------------------------------------- features ---
def alias_flag(period):
    """True if the period sits on a common sampling alias (1 d, 0.5 d, 2 d, lunar ~29.5 d)."""
    for a in (0.5, 1.0, 2.0, 29.53, 365.25):
        if abs(period - a) / a < 0.02:
            return True
    return False


def lc_features(t, m, e, fmin=1 / 200.0, fmax=10.0, spp=5):
    """Lomb-Scargle features. Returns dict or None if too few points."""
    from astropy.timeseries import LombScargle
    from scipy.stats import skew
    t = np.asarray(t, float); m = np.asarray(m, float); e = np.asarray(e, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    if len(t) < 40:
        return None
    # 5-sigma clip about the median
    med = np.median(m); mad = 1.4826 * np.median(np.abs(m - med)) + 1e-6
    k = np.abs(m - med) < 5 * mad + 3 * np.median(e)
    t, m, e = t[k], m[k], e[k]
    if len(t) < 40:
        return None
    ls = LombScargle(t, m, e)
    freq, power = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax,
                               samples_per_peak=spp)
    i = int(np.argmax(power))
    pbest = 1.0 / freq[i]
    fap = float(ls.false_alarm_probability(power[i], method="baluev",
                                           minimum_frequency=fmin, maximum_frequency=fmax))
    fap = min(max(fap, 1e-300), 1.0)
    chi2red = float(np.sum(((m - np.average(m, weights=1 / e ** 2)) / e) ** 2) / (len(m) - 1))
    p5, p95 = np.percentile(m, [5, 95])
    q1, q3 = np.percentile(m, [25, 75])
    return dict(log_fap=np.log10(fap), power=float(power[i]), amp=float(p95 - p5),
                amp_over_err=float((p95 - p5) / np.median(e)), chi2red=chi2red,
                log_period=float(np.log10(pbest)), log_n=float(np.log10(len(t))),
                mean_mag=float(np.mean(m)), skew=float(skew(m)),
                iqr_over_err=float((q3 - q1) / np.median(e)),
                period=float(pbest), alias=bool(alias_flag(pbest)), n=int(len(t)))


ECL_GATE = dict(snr=8.0, min_depth=0.10, min_epochs=3, max_depth_scatter=0.40,
                min_concentration=0.85, min_frac_in_faint=0.70)

def eclipse_features(t, m, e):
    """Narrow-dip (detached eclipsing binary) channel. Returns robust statistics, never a verdict."""
    from astropy.timeseries import BoxLeastSquares
    t = np.asarray(t, float); m = np.asarray(m, float); e = np.asarray(e, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    if len(t) < 60:
        return None
    med = np.median(m); mad = 1.4826 * np.median(np.abs(m - med)) + 1e-6
    k = (m - med) > -(5 * mad + 3 * np.median(e))          # clip only implausibly BRIGHT points
    t, m, e = t[k], m[k], e[k]
    if len(t) < 60:
        return None
    y = -(m - np.median(m))
    bls = BoxLeastSquares(t, y, dy=e)
    span = t.max() - t.min()
    best = None
    for pmin, pmax, durs in ((0.5, 3.0, [0.04, 0.08, 0.15, 0.3]), (3.0, 30.0, [0.15, 0.3, 0.6, 1.2, 2.4])):
        nper = int(min(5000, max(2000, (1 / pmin - 1 / pmax) * span ** 2 / (0.3 * min(durs)) / 4)))
        periods = 1.0 / np.linspace(1 / pmax, 1 / pmin, nper)
        r = bls.power(periods, durs, objective="snr")
        i = int(np.argmax(r.power))
        if best is None or r.power[i] > best[0]:
            best = (float(r.power[i]), float(r.period[i]), float(r.duration[i]), float(r.transit_time[i]))
    _, P, dur, t0 = best
    ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P
    inn = np.abs(ph) < 0.6 * dur                              # slightly widened window
    n_in = int(inn.sum())
    if n_in < 4 or (~inn).sum() < 30:
        return dict(ecl_period=P, ecl_duration=dur, ecl_snr=0.0, ecl_depth=0.0, ecl_epochs=0,
                    ecl_depth_scatter=9.0, ecl_concentration=0.0, ecl_frac_in_faint=0.0)
    z = m - np.median(m[~inn])                                # eclipse points are fainter -> z > 0
    sig = max(1.4826 * np.median(np.abs(z[~inn] - np.median(z[~inn]))), 0.5 * np.median(e))
    depth = float(np.median(z[inn]))
    snr = depth / (sig / np.sqrt(n_in)) if depth > 0 else 0.0
    epochs = int(len(np.unique(np.floor((t[inn] - t0) / P).astype(int))))
    dscat = float(1.4826 * np.median(np.abs(z[inn] - np.median(z[inn]))) / depth) if depth > 0 else 9.0
    faint = z > 4 * sig
    conc = float((faint & inn).sum() / max(faint.sum(), 1))
    fin = float((z[inn] > 2 * sig).mean())
    return dict(ecl_period=P, ecl_duration=dur, ecl_snr=float(snr), ecl_depth=depth, ecl_epochs=epochs,
                ecl_depth_scatter=dscat, ecl_concentration=conc, ecl_frac_in_faint=fin)

def ecl_pass(f, gate=ECL_GATE):
    if f is None:
        return False
    return bool(f["ecl_snr"] >= gate["snr"] and f["ecl_depth"] >= gate["min_depth"]
                and f["ecl_epochs"] >= gate["min_epochs"]
                and f["ecl_depth_scatter"] <= gate["max_depth_scatter"]
                and f["ecl_concentration"] >= gate["min_concentration"]
                and f["ecl_frac_in_faint"] >= gate["min_frac_in_faint"])


def add_eclipse_channel(unk, sel, n):
    """Append eclipse-channel survivors (which the sinusoid channel cannot see) to the selection."""
    sel = sel.copy(); sel["channel"] = "periodic"
    if "ecl_pass" not in unk.columns:
        return sel
    e = unk[unk["ecl_pass"].fillna(False).astype(bool) & ~unk["source_id"].isin(sel["source_id"])].copy()
    if e.empty:
        return sel
    e["period"] = e["ecl_period"]; e["channel"] = "eclipse"
    e = e.sort_values("ecl_snr", ascending=False).head(n)
    return pd.concat([sel, e], ignore_index=True)


def varwise_period_agreement(opt_period, vw_period, tol=0.03):
    """True/False if VarWISE's own infrared period matches the optical one (x1, x2, x0.5); None if unknown.
    Independent evidence that the optical variable IS the infrared variable, not a chance neighbour."""
    try:
        o, v = float(opt_period), float(vw_period)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(o) and np.isfinite(v)) or o <= 0 or v <= 0:
        return None
    return bool(any(abs(o / (v * k) - 1) <= tol for k in (0.5, 1.0, 2.0)))


def channel_info(r):
    return dict(channel=getattr(r, "channel", "periodic"),
                varwise_period_agrees=varwise_period_agreement(getattr(r, "period", np.nan),
                                                               getattr(r, "varwise_period1", np.nan)))


def temporal_consistency_ecl(lc, period, tolerance=0.03):
    """Eclipse channel: BOTH temporal halves must independently show a significant dip at the period."""
    if lc is None or len(lc) < 120:
        return dict(confirmed=False, early_period=None, late_period=None, reason="fewer than 120 observations")
    lc = lc.sort_values("mjd")
    first, second = np.array_split(np.arange(len(lc)), 2)
    periods, strong = [], []
    for idx in (first, second):
        q = lc.iloc[idx]
        ef = eclipse_features(q["mjd"], q["mag"], q["magerr"])
        periods.append(ef["ecl_period"] if ef else None)
        strong.append(bool(ef) and ef["ecl_snr"] >= 0.5 * ECL_GATE["snr"]
                      and ef["ecl_depth"] >= 0.5 * ECL_GATE["min_depth"])
    def near(p, ref):
        return p is not None and any(abs(p / (ref * k) - 1) <= tolerance for k in (0.5, 1.0, 2.0))
    ok = all(strong) and all(near(p, period) for p in periods)
    return dict(confirmed=bool(ok), early_period=periods[0], late_period=periods[1],
                reason="eclipse period independently seen in both halves" if ok else
                       "eclipse absent or inconsistent in a temporal half")


def outlier_lead_features(t, m, e):
    """Model-free, NON-PERIODIC variability lead: repeated significant excursions.

    Flags both previously untrained shapes and instrumental contamination.
    ALWAYS export to an unverified queue; this is not astrophysical proof.
    """
    t, m, e = (np.asarray(x, float) for x in (t, m, e))
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    if len(t) < 60:
        return dict(flag=False, excursion_count=0, excursion_nights=0, max_excursion_sigma=0)
    center = np.median(m)
    threshold = np.maximum(5 * e, 0.12)  # both statistical and absolute floor
    high, low = m - center > threshold, center - m > threshold
    ind = high if high.sum() >= low.sum() else low
    n = int(ind.sum())
    nights = len(np.unique(np.floor(t[ind]).astype(int)))
    max_z = float(np.max(np.abs(m - center) / e))
    # A short real event may occupy only a few epochs and hence have a modest
    # global chi-squared. Do not discard it for that reason.
    return dict(flag=bool(n >= 4 and nights >= 3 and max_z >= 6),
                excursion_count=n, excursion_nights=int(nights),
                max_excursion_sigma=max_z)


# ------------------------------------------------------------------ model ---
def sky_cell(ra, dec, cell=1.0):
    return f"{int(np.floor(ra / cell))}_{int(np.floor(dec / cell))}"


def is_test_cell(cellname):
    """Deterministic 20% of sky cells go to the held-out set."""
    h = int(hashlib.md5(cellname.encode()).hexdigest(), 16)
    return (h % 5) == 0


def spatial_test_mask(df, seed=0):
    """Find an ~80/20 SOURCE split while NEVER mixing sky cells.

    A tiny patch may contain only a few giant cells; then a scientifically
    meaningful blocked 80/20 split does not exist and training must refuse.
    """
    from sklearn.model_selection import GroupShuffleSplit
    group = df["cell"].astype(str).to_numpy()
    label = df["label"].to_numpy()
    n = len(df)
    if n < 120 or len(np.unique(group)) < 10:
        raise RuntimeError(f"insufficient spatial diversity for blocked 80/20: "
                           f"{n} sources across {len(np.unique(group))} cells; "
                           "sample larger, separated sky regions")
    splitter = GroupShuffleSplit(n_splits=200, test_size=0.20, random_state=seed)
    best = None
    for tr, te in splitter.split(np.zeros(n), label, groups=group):
        if len(tr) < 90 or len(te) < 20:
            continue
        if np.sum(label[tr] == 1) < 20 or np.sum(label[tr] == 0) < 20:
            continue
        if np.sum(label[te] == 1) < 10 or np.sum(label[te] == 0) < 10:
            continue
        fraction = len(te) / n
        if abs(fraction - 0.20) > 0.05:
            continue
        imbalance = abs(float(label[te].mean()) - float(label.mean()))
        quality = abs(fraction - .20) + .1 * imbalance
        if best is None or quality < best[0]:
            best = (quality, te)
    if best is None:
        raise RuntimeError("cannot construct a geographically disjoint 80/20 "
                           "split with enough examples of both labels; "
                           "enlarge the training sky regions")
    mask = np.zeros(n, dtype=bool)
    mask[best[1]] = True
    return mask


def train_eval(df, seed=0, verbose=True):
    """df needs FEATURES, 'label' (1 known variable, 0 constant), 'cell'.
    Returns (model, auc_test, auc_shuffled_control)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    test = spatial_test_mask(df, seed)
    tr, te = df[~test], df[test]
    if tr["label"].nunique() < 2 or te["label"].nunique() < 2 or len(te) < 10:
        raise RuntimeError(f"split unusable: train={len(tr)} test={len(te)} "
                           f"(need both classes in both). Use a larger box or more stars.")
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                         max_iter=200, random_state=seed)
    clf.fit(tr[FEATURES], tr["label"])
    auc = roc_auc_score(te["label"], clf.predict_proba(te[FEATURES])[:, 1])
    # planted-failure control: shuffled training labels, mean over 10 shuffles, must be ~0.5
    rng = np.random.default_rng(seed)
    a0 = []
    for _ in range(10):
        c0 = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                            max_iter=100, random_state=seed)
        c0.fit(tr[FEATURES], rng.permutation(tr["label"].values))
        a0.append(roc_auc_score(te["label"], c0.predict_proba(te[FEATURES])[:, 1]))
    auc0 = float(np.mean(a0))
    if verbose:
        print(f"[80/20] train n={len(tr)} ({len(tr)/len(df):.1%})  "
              f"held-out n={len(te)} ({len(te)/len(df):.1%}) "
              f"over disjoint sky cells")
        print(f"[80/20] held-out AUC = {auc:.3f}   shuffled-label control AUC = {auc0:.3f}")
    return clf, auc, auc0


# ---------------------------------------------------------------- selftest ---
def synth_lc(rng, periodic, n=160, span=1000.0):
    t = np.sort(rng.uniform(0, span, n))
    e = np.full(n, 0.03) * rng.uniform(0.8, 1.3)
    m0 = rng.uniform(13, 18)
    if periodic:
        P = 10 ** rng.uniform(np.log10(0.25), np.log10(20))
        A = rng.uniform(0.08, 0.5)
        m = m0 + A * np.sin(2 * np.pi * t / P + rng.uniform(0, 6.28))
    else:
        P = np.nan
        m = np.full(n, m0)
    return t, m + rng.normal(0, e), e, P


def selftest():
    rng = np.random.default_rng(42)
    # (a) planted period must be recovered
    t, m, e, P = synth_lc(rng, True)
    P = 3.7; m = 15 + 0.3 * np.sin(2 * np.pi * t / P) + rng.normal(0, e)
    f = lc_features(t, m, e)
    rec = abs(f["period"] - P) / P < 0.01
    print(f"[test 1] planted P=3.7 d recovered as {f['period']:.4f} d -> {'PASS' if rec else 'FAIL'}")
    # (b) flat star must NOT look significant
    t, m, e, _ = synth_lc(rng, False)
    f0 = lc_features(t, m, e)
    flat_ok = f0["log_fap"] > -4
    print(f"[test 2] flat star log10 FAP = {f0['log_fap']:.1f} -> {'PASS' if flat_ok else 'FAIL'}")
    # (c) model with sky-cell split, plus shuffled-label control
    rows = []
    for k in range(600):
        periodic = k % 2 == 0
        t, m, e, P = synth_lc(rng, periodic)
        ft = lc_features(t, m, e, fmax=10.0, spp=4)
        ft["label"] = int(periodic)
        ft["cell"] = sky_cell(rng.uniform(0, 30), rng.uniform(0, 30))
        rows.append(ft)
    df = pd.DataFrame(rows)
    clf, auc, auc0 = train_eval(df)
    ok_model = auc > 0.9 and abs(auc0 - 0.5) < 0.1
    print(f"[test 3] held-out AUC>0.9 and shuffled control within 0.5+-0.1 -> {'PASS' if ok_model else 'FAIL'}")
    # (d) the control itself must be able to fail: feed it labels equal to a feature
    df_leak = df.copy(); df_leak["power"] = df_leak["label"] * 1.0
    _, a_l, a_l0 = train_eval(df_leak, verbose=False)
    ctrl_ok = a_l > 0.95 and abs(a_l0 - 0.5) < 0.1
    print(f"[test 4] leak planted: AUC {a_l:.2f} vs shuffled {a_l0:.2f} (control separates) -> "
          f"{'PASS' if ctrl_ok else 'FAIL'}")
    allok = rec and flat_ok and ok_model and ctrl_ok
    print("SELFTEST", "PASSED" if allok else "FAILED")
    return 0 if allok else 1


# ----------------------------------------------------------------- network ---
def tap(url):
    import pyvo
    svc = pyvo.dal.TAPService(url)
    orig = svc.run_sync

    def run_sync_retry(*a, **k):
        for i in range(5):
            try:
                return orig(*a, **k)
            except Exception as ex:
                if type(ex).__name__ in ("DALQueryError", "DALFormatError") and "ProxyError" not in str(ex) \
                        and "Connection" not in str(ex) and "Timeout" not in str(ex):
                    raise                          # a genuinely bad query: do not retry
                if i == 4:
                    raise
                time.sleep(3 * (i + 1))
    svc.run_sync = run_sync_retry
    return svc


def probe():
    import requests
    hosts = {
        "IRSA TAP": IRSA_TAP + "/capabilities",
        "Gaia TAP": GAIA_TAP + "/capabilities",
        "ZTF API": ZTF_API + "?POS=CIRCLE%20150%2030%200.001&FORMAT=csv",
        "VizieR": "https://vizier.cds.unistra.fr/viz-bin/votable",
        "SIMBAD": "https://simbad.cds.unistra.fr/simbad/sim-tap/capabilities",
    }
    for k, u in hosts.items():
        try:
            r = requests.get(u, timeout=30)
            print(f"{k:9s} HTTP {r.status_code}")
        except Exception as ex:
            print(f"{k:9s} FAILED {type(ex).__name__}: {ex}")
    irsa = tap(IRSA_TAP)
    q = ("SELECT table_name, description FROM TAP_SCHEMA.tables "
         "WHERE lower(table_name) LIKE '%varwise%' OR lower(description) LIKE '%varwise%'")
    t = irsa.run_sync(q).to_table().to_pandas()
    print("\nVarWISE tables found:\n", t.to_string())
    for name in t["table_name"].head(3):
        c = irsa.run_sync(f"SELECT column_name, datatype, description FROM TAP_SCHEMA.columns "
                          f"WHERE table_name='{name}'").to_table().to_pandas()
        print(f"\n--- columns of {name} ---\n", c.to_string())


def pick_radec(cols):
    low = {c.lower(): c for c in cols}
    for a, b in (("ra", "dec"), ("ra_deg", "dec_deg"), ("raj2000", "dej2000"),
                 ("w1_ra", "w1_dec"), ("ra_mean", "dec_mean")):
        if a in low and b in low:
            return low[a], low[b]
    raise RuntimeError(f"cannot guess RA/Dec columns from {list(cols)}; pass --ra-col/--dec-col")


def get_varwise(args):
    irsa = tap(IRSA_TAP)
    table = args.table
    if not table:
        t = irsa.run_sync("SELECT table_name FROM TAP_SCHEMA.tables WHERE "
                          "lower(table_name) LIKE '%varwise%'").to_table().to_pandas()
        if t.empty:
            raise RuntimeError("no VarWISE table found in IRSA TAP; run `probe` and pass --table")
        print("VarWISE tables:", list(t["table_name"]))
        # The associations table contains detection IDs but NO sky positions.
        # It can sort first, causing the original live smoke test to fail.
        names = list(t["table_name"])
        prefer = next((x for x in names if x.lower().split(".")[-1] == "varwisepure"), None)
        table = prefer or next((x for x in names if "assoc" not in x.lower()), None)
        if table is None:
            raise RuntimeError("VarWISE has no positional catalogue; specify --table explicitly")
    cols = irsa.run_sync(f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name='{table}'"
                         ).to_table().to_pandas()["column_name"].tolist()
    rac, decc = (args.ra_col, args.dec_col) if args.ra_col else pick_radec(cols)
    half = args.size / 2.0
    cosd = max(np.cos(np.radians(args.dec0)), 0.05)
    q = (f"SELECT TOP {args.max_vw} * FROM {table} WHERE CONTAINS(POINT('ICRS',{rac},{decc}),"
         f"BOX('ICRS',{args.ra0},{args.dec0},{args.size / cosd},{args.size}))=1")
    print("VarWISE query:", q)
    df = irsa.run_sync(q, maxrec=args.max_vw).to_table().to_pandas()
    df = df.rename(columns={rac: "ra", decc: "dec"})
    print(f"VarWISE objects in box: {len(df)}")
    return df


GAIA_JOIN = """
SELECT m.uid AS uid, g.source_id, g.ra AS g_ra, g.dec AS g_dec, g.phot_g_mean_mag,
       g.bp_rp, g.ruwe, g.phot_variable_flag,
       3600.0*DISTANCE(POINT('ICRS',m.ra,m.dec),POINT('ICRS',g.ra,g.dec)) AS sep
FROM TAP_UPLOAD.t AS m
JOIN gaiadr3.gaia_source AS g
  ON 1=CONTAINS(POINT('ICRS',g.ra,g.dec),CIRCLE('ICRS',m.ra,m.dec,{r}))
WHERE g.phot_g_mean_mag BETWEEN 12 AND 19
"""


def gaia_match(vw):
    from astropy.table import Table
    if vw.empty:
        return pd.DataFrame(columns=["uid", "source_id", "sep", "vari_id", "n_gaia",
                                     "g_ra", "g_dec", "phot_g_mean_mag", "bp_rp", "ruwe"])
    gaia = tap(GAIA_TAP)
    up = Table.from_pandas(vw[["uid", "ra", "dec"]])
    out = []
    for i in range(0, len(up), 5000):
        chunk = up[i:i + 5000]
        try:
            # Run the spatial join separately from vari_summary. The original
            # three-table join was repeatedly aborted by the real Gaia TAP.
            res = gaia.run_sync(GAIA_JOIN.format(r=MATCH_ARCSEC / 3600.0),
                                uploads={"t": chunk}, maxrec=200000)
            frame = res.to_table().to_pandas()
        except Exception as ex:
            # Cones are indexed in Gaia and worked live even when table-upload
            # joins did not. Slow but reproducible fallback for small boxes.
            if len(chunk) > 150:
                raise RuntimeError("Gaia bulk join unavailable; avoid thousands of slow "
                                   "individual queries with --skip-gaia") from ex
            print(f"  Gaia bulk match failed ({type(ex).__name__}: {ex}); "
                  f"falling back to {len(chunk)} individual cones")
            rows = []
            for r in chunk:
                q = ("SELECT TOP 20 g.source_id,g.ra AS g_ra,g.dec AS g_dec,"
                     "g.phot_g_mean_mag,g.bp_rp,g.ruwe,g.phot_variable_flag "
                     "FROM gaiadr3.gaia_source AS g "
                     f"WHERE 1=CONTAINS(POINT('ICRS',g.ra,g.dec),"
                     f"CIRCLE('ICRS',{float(r['ra'])},{float(r['dec'])},{MATCH_ARCSEC / 3600.0})) "
                     "AND g.phot_g_mean_mag BETWEEN 12 AND 19")
                sub = gaia.run_sync(q).to_table().to_pandas()
                if len(sub):
                    sub["uid"] = int(r["uid"])
                    sub["sep"] = 3600 * np.degrees(np.arccos(np.clip(
                        np.sin(np.radians(float(r["dec"]))) * np.sin(np.radians(sub["g_dec"])) +
                        np.cos(np.radians(float(r["dec"]))) * np.cos(np.radians(sub["g_dec"])) *
                        np.cos(np.radians(float(r["ra"]) - sub["g_ra"])), -1, 1)))
                    rows.append(sub)
            frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out.append(frame)
        print(f"  Gaia match chunk {i // 5000 + 1}: {len(out[-1])} rows")
    g = pd.concat(out, ignore_index=True)
    if g.empty:
        return pd.DataFrame(columns=["uid", "source_id", "sep", "vari_id", "n_gaia",
                                     "g_ra", "g_dec", "phot_g_mean_mag", "bp_rp", "ruwe"])
    # Gaia variable labels are only queried AFTER finding sparse counterparts.
    known = []
    ids = g["source_id"].dropna().astype("int64").drop_duplicates().tolist()
    for i in range(0, len(ids), 400):
        id_csv = ",".join(str(x) for x in ids[i:i + 400])
        q = f"SELECT source_id AS vari_id FROM gaiadr3.vari_summary WHERE source_id IN ({id_csv})"
        known.append(gaia.run_sync(q).to_table().to_pandas())
    v = pd.concat(known, ignore_index=True) if known else pd.DataFrame(columns=["vari_id"])
    g = g.merge(v, left_on="source_id", right_on="vari_id", how="left")
    n = g.groupby("uid")["source_id"].transform("count")
    g["n_gaia"] = n
    g = g.sort_values("sep").drop_duplicates("uid")
    return g


def gaia_training_stars(args, n_each):
    """Known Gaia variables (positives) and non-variables (negatives) in the same box."""
    gaia = tap(GAIA_TAP)
    cosd = max(np.cos(np.radians(args.dec0)), 0.05)
    ra_lo, ra_hi = args.ra0 - args.size / 2 / cosd, args.ra0 + args.size / 2 / cosd
    de_lo, de_hi = args.dec0 - args.size / 2, args.dec0 + args.size / 2
    base = (f"g.ra BETWEEN {ra_lo} AND {ra_hi} AND g.dec BETWEEN {de_lo} AND {de_hi} "
            f"AND g.phot_g_mean_mag BETWEEN 12 AND 19")
    pos = gaia.run_sync(
        f"SELECT TOP {n_each} g.source_id,g.ra,g.dec,g.phot_g_mean_mag FROM gaiadr3.gaia_source g "
        f"JOIN gaiadr3.vari_summary v ON v.source_id=g.source_id WHERE {base} "
        f"AND v.num_selected_g_fov > 40 ORDER BY random_index").to_table().to_pandas()
    neg = gaia.run_sync(
        f"SELECT TOP {n_each} g.source_id,g.ra,g.dec,g.phot_g_mean_mag FROM gaiadr3.gaia_source g "
        f"LEFT JOIN gaiadr3.vari_summary v ON v.source_id=g.source_id WHERE {base} "
        f"AND v.source_id IS NULL AND g.phot_variable_flag <> 'VARIABLE' ORDER BY random_index"
    ).to_table().to_pandas()
    pos["label"] = 1; neg["label"] = 0
    return pd.concat([pos, neg], ignore_index=True)


def ztf_lc(ra, dec, cache):
    import requests
    key = os.path.join(cache, f"{ra:.5f}_{dec:.5f}.pkl")
    if os.path.exists(key):
        return pickle.load(open(key, "rb"))
    best = None
    for band in ("r", "g"):
        for attempt in range(3):
            try:
                r = requests.get(ZTF_API, params={"POS": f"CIRCLE {ra} {dec} 0.0006",
                                                  "BANDNAME": band, "FORMAT": "csv"}, timeout=60)
                if r.status_code != 200 or len(r.text) < 50:
                    break
                from io import StringIO
                d = pd.read_csv(StringIO(r.text))
                d = d[d["catflags"] == 0]
                if len(d):
                    top = d["oid"].value_counts().index[0]      # one object only
                    d = d[d["oid"] == top]
                    if best is None or len(d) > len(best):
                        best = d[["mjd", "mag", "magerr"]].copy()
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        time.sleep(0.25)
        if best is not None and len(best) >= 60:
            break
    pickle.dump(best, open(key, "wb"))
    return best


def ztf_single_band(ra, dec, band, cache):
    """Fetch ZTF g or r independently (not a second observatory)."""
    import requests
    from io import StringIO
    key = os.path.join(cache, f"band_{ra:.5f}_{dec:.5f}_{band}.pkl")
    if os.path.exists(key):
        return pickle.load(open(key, "rb"))
    response = requests.get(ZTF_API, params={"POS": f"CIRCLE {ra} {dec} 0.0006",
                                             "BANDNAME": band, "FORMAT": "csv"}, timeout=80)
    response.raise_for_status()
    d = pd.read_csv(StringIO(response.text))
    d = d[d["catflags"] == 0]
    if d.empty:
        return None
    oid = d["oid"].value_counts().index[0]
    d = d[d["oid"] == oid][["mjd", "mag", "magerr"]].copy()
    pickle.dump(d, open(key, "wb"))
    return d


def ztf_two_band_check(ra, dec, cache):
    """Year-scale agreement in g and r; supporting evidence, not confirmation."""
    from scipy.stats import spearmanr
    try:
        r = ztf_single_band(ra, dec, "r", cache)
        g = ztf_single_band(ra, dec, "g", cache)
        if r is None or g is None:
            raise ValueError("one filter has no valid data")
        def annual(lc):
            d = lc.copy()
            d["bin"] = ((d["mjd"] - 58000) // 365).astype(int)
            group = d.groupby("bin")["mag"]
            return group.median()[group.size() >= 5]
        rb, gb = annual(r), annual(g)
        common = sorted(set(rb.index) & set(gb.index))
        if len(common) < 5:
            raise ValueError("fewer than 5 jointly observed year bins")
        rv, gv = rb.loc[common], gb.loc[common]
        rho = float(spearmanr(rv, gv).statistic)
        ramp, gamp = float(rv.max() - rv.min()), float(gv.max() - gv.min())
        consistent = bool(np.isfinite(rho) and rho >= 0.7 and
                          ramp >= 0.3 and gamp >= 0.3)
        return dict(crossband_consistent=consistent, crossband_rho=rho,
                    r_year_amp_mag=ramp, g_year_amp_mag=gamp,
                    common_year_bins=len(common), crossband_error="")
    except Exception as ex:
        return dict(crossband_consistent=False, crossband_rho=np.nan,
                    r_year_amp_mag=np.nan, g_year_amp_mag=np.nan,
                    common_year_bins=0, crossband_error=f"{type(ex).__name__}: {ex}")


def featurize(df, cache, label_col=None, workers=3, eclipse=False):
    """Download independent ZTF light curves concurrently at a modest rate."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = []

    def one(r):
        ra = getattr(r, "g_ra", None) if hasattr(r, "g_ra") else r.ra
        de = getattr(r, "g_dec", None) if hasattr(r, "g_dec") else r.dec
        lc = ztf_lc(float(ra), float(de), cache)
        if lc is None:
            return None
        f = lc_features(lc["mjd"], lc["mag"], lc["magerr"])
        if f is None:
            return None
        if eclipse:
            ef = eclipse_features(lc["mjd"], lc["mag"], lc["magerr"])
            if ef:
                f.update(ef)
            f["ecl_pass"] = ecl_pass(ef)
        f["source_id"] = int(r.source_id)
        f["ra"] = float(ra); f["dec"] = float(de)
        f["cell"] = sky_cell(float(ra), float(de))
        if label_col:
            f["label"] = int(getattr(r, label_col))
        return f

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        futures = [pool.submit(one, r) for r in df.itertuples()]
        for k, fut in enumerate(as_completed(futures), start=1):
            try:
                f = fut.result()
                if f is not None:
                    rows.append(f)
            except Exception as ex:
                print(f"  light-curve error ({type(ex).__name__}): {str(ex)[:100]}")
            if k % 10 == 0 or k == len(futures):
                print(f"  light curves processed: {k}/{len(df)} (usable {len(rows)})", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- prior art ---
VAR_OTYPES = {"V*", "Mi*", "Ir*", "RR*", "Ce*", "Pu*", "EB*", "El*", "RS*", "BY*", "Er*", "Or*",
              "TT*", "Fl*", "LP*", "SX*", "dS*", "gD*", "bC*", "WV*", "Al*", "Psr", "cC*", "RV*",
              "AB*", "SB*", "Ro*", "Eb*", "V*?", "EB?", "RR?", "Mi?", "Pu?", "Ir?", "Or?", "LP?"}
VIZIER_CATS = {"VSX": "B/vsx/vsx", "ASAS-SN variables": "II/366",
               "ZTF periodic (Chen+2020)": "J/ApJS/249/18", "Gaia DR3 vari classifier": "I/358/vclassre"}


def prior_art(ra, dec):
    """Return list of services that already list a variable here (empty = none found)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.vizier import Vizier
    from astroquery.simbad import Simbad
    c = SkyCoord(ra * u.deg, dec * u.deg)
    hits = []
    for name, cat in VIZIER_CATS.items():
        try:
            r = Vizier(row_limit=5).query_region(c, radius=3 * u.arcsec, catalog=cat)
            if len(r) and len(r[0]):
                hits.append(name)
        except Exception as ex:
            hits.append(f"{name}:ERROR({type(ex).__name__})")      # an error blocks acceptance
    try:
        s = Simbad.query_region(c, radius=3 * u.arcsec)
        if s is not None and len(s):
            col = "OTYPE" if "OTYPE" in s.colnames else ("otype" if "otype" in s.colnames else None)
            if col:
                for ot in s[col]:
                    if str(ot).strip() in VAR_OTYPES:
                        hits.append(f"SIMBAD:{str(ot).strip()}")
    except Exception as ex:
        hits.append(f"SIMBAD:ERROR({type(ex).__name__})")
    return hits


# -------------------------------------------------------------------- run ---
def fold_plot(ra, dec, period, cache, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lc = ztf_lc(ra, dec, cache)
    ph = (lc["mjd"].values % period) / period
    plt.figure(figsize=(6, 3.5))
    plt.errorbar(np.r_[ph, ph + 1], np.r_[lc["mag"], lc["mag"]],
                 np.r_[lc["magerr"], lc["magerr"]], fmt=".", ms=3, alpha=0.6)
    plt.gca().invert_yaxis(); plt.xlabel(f"phase (P = {period:.5f} d)"); plt.ylabel("ZTF mag")
    plt.title(f"RA {ra:.5f} Dec {dec:.5f}"); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def raw_plot(ra, dec, cache, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lc = ztf_lc(ra, dec, cache)
    plt.figure(figsize=(7, 3.5))
    plt.errorbar(lc["mjd"], lc["mag"], lc["magerr"], fmt=".", ms=3, alpha=0.7)
    plt.gca().invert_yaxis()
    plt.xlabel("MJD"); plt.ylabel("ZTF magnitude")
    plt.title(f"Unverified light-curve anomaly RA {ra:.5f} Dec {dec:.5f}")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def two_band_plot(ra, dec, cache, path):
    """Show independent ZTF g and r light curves on one aligned time axis."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, band, color in zip(axes, ("g", "r"), ("#17a398", "#d57b32")):
        lc = ztf_single_band(ra, dec, band, cache)
        if lc is None:
            ax.text(.5, .5, f"ZTF {band}: no data", ha="center", transform=ax.transAxes)
            continue
        ax.errorbar(lc["mjd"], lc["mag"], lc["magerr"], fmt=".",
                    ms=2.5, alpha=.35, color=color)
        d = lc.copy()
        d["bin"] = ((d["mjd"] - 58000) // 365).astype(int)
        annual = d.groupby("bin")["mag"].median()
        ax.plot(58000 + (annual.index.to_numpy() + .5) * 365,
                annual.to_numpy(), color=color, lw=2, marker="o", ms=3)
        ax.invert_yaxis()
        ax.set_ylabel(f"ZTF {band} magnitude")
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("Modified Julian Date")
    fig.suptitle(f"Unverified two-filter optical variability | RA {ra:.5f}, Dec {dec:.5f}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def collect_anomaly_leads(unk, args, cache, out, already_handled=None):
    """Second discovery channel for shapes missed by Lomb-Scargle/ML.

    Never labels these as verified: multi-band/imaging confirmation is required.
    """
    if not args.include_anomalies:
        return []
    already_handled = set(already_handled or [])
    prospects = unk[unk["amp_over_err"] >= 4]
    prospects = prospects.sort_values("amp_over_err", ascending=False)
    leads = []
    for r in prospects.itertuples():
        if int(r.source_id) in already_handled:
            continue
        lc = ztf_lc(r.ra, r.dec, cache)
        if lc is None:
            continue
        signature = outlier_lead_features(lc["mjd"], lc["mag"], lc["magerr"])
        if not signature["flag"]:
            continue
        hits = prior_art(r.ra, r.dec)
        if any("ERROR" not in h for h in hits):
            continue
        rec = r._asdict(); rec.pop("Index", None); rec.update(channel_info(r))
        rec.update(signature)
        bands = ztf_two_band_check(r.ra, r.dec, cache)
        rec.update(bands)
        stage = ("correlated g/r filters" if bands["crossband_consistent"] else
                 "g/r evidence missing or inconsistent")
        rec.update(prior_art=";".join(hits), status="UNVERIFIED ANOMALY - " + stage +
                   "; still requires imaging and astrophysical confirmation")
        leads.append(rec)
        raw_plot(r.ra, r.dec, cache, os.path.join(out, f"raw_{r.source_id}.png"))
        if not bands["crossband_error"]:
            two_band_plot(r.ra, r.dec, cache, os.path.join(out, f"bands_{r.source_id}.png"))
        if len(leads) >= args.max_prior:
            break
    return leads


VARWISE_PROVENANCE = ("cluster_id", "designation", "vartype", "confidence",
                      "blended_source", "latent_artifact", "period1", "w1_amp", "w2_amp",
                      "variability_snr", "period_significance", "suspect_period", "simbad_type",
                      "known_extragalactic", "n_obs")


def attach_varwise_provenance(g, vw):
    """Keep the original catalogue identifier and artifact warnings in exports."""
    cols = [c for c in VARWISE_PROVENANCE if c in vw.columns]
    renamed = vw[["uid"] + cols].rename(columns={c: "varwise_" + c for c in cols})
    return g.merge(renamed, on="uid", how="left")


def gaia_export_columns(g):
    return [c for c in ("source_id", "phot_g_mean_mag", "bp_rp", "ruwe") if c in g.columns] + [
        c for c in g.columns if c.startswith("varwise_")]


def varwise_targets_without_gaia(vw):
    """Optical target positions directly from VarWISE; Gaia is NOT mandatory.

    Negative source_id values identify VarWISE clusters, never Gaia IDs.
    Existing optical catalogues are checked later before accepting a candidate.
    """
    d = vw.copy()
    # VarWISE cluster IDs are catalogue-global; row uids reset for every sky
    # box and would overwrite plots when a resumable survey spans many boxes.
    stable = d["cluster_id"] if "cluster_id" in d.columns else d["uid"]
    d["source_id"] = -(stable.astype("int64").abs() + 1)
    d["g_ra"] = d["ra"]
    d["g_dec"] = d["dec"]
    d["vari_id"] = np.nan
    d["n_gaia"] = 0
    d["phot_g_mean_mag"] = np.nan
    d["bp_rp"] = np.nan
    d["ruwe"] = np.nan
    for c in VARWISE_PROVENANCE:
        if c in d.columns:
            d["varwise_" + c] = d[c]
    return d


def run(args):
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "ztf_cache"); os.makedirs(cache, exist_ok=True)
    vw = get_varwise(args)
    vw = vw.reset_index(drop=True); vw["uid"] = np.arange(len(vw))
    if args.skip_gaia:
        g_unk = varwise_targets_without_gaia(vw)
        print(f"Gaia match deliberately skipped: searching {len(g_unk)} VarWISE positions "
              "(optical catalogue novelty checked later).")
    else:
        g = gaia_match(vw)
        g = attach_varwise_provenance(g[(g["sep"] <= MATCH_ARCSEC)], vw)
        print(f"VarWISE objects with a Gaia G 12-19 match: {len(g)}")
        g_known = g[g["vari_id"].notna()]
        g_unk = g[g["vari_id"].isna() & (g["n_gaia"] == 1)]      # unique match only
        print(f"  already have a Gaia variability solution: {len(g_known)}   unknown: {len(g_unk)}")

    clf = None
    if args.search_mode != "signal":
        try:
            tr = gaia_training_stars(args, args.n_train)
            # A Gaia source missing vari_summary can be a genuine VarWISE
            # target. It must NOT be learned as a negative before discovery.
            tr = tr[~tr["source_id"].isin(g_unk["source_id"])].copy()
            tr = tr.rename(columns={"ra": "g_ra", "dec": "g_dec"})
            ft = featurize(tr, cache, label_col="label")
            print(f"usable Gaia proxy labels: {len(ft)} "
                  f"(known variables {int(ft['label'].sum())}, "
                  f"not Gaia-flagged {int((1 - ft['label']).sum())})")
            clf, auc, auc0 = train_eval(ft)
            if auc < 0.85 or abs(auc0 - 0.5) > 0.12:
                raise RuntimeError(f"validation failed: AUC={auc:.3f}, null={auc0:.3f}")
            import joblib
            joblib.dump(dict(model=clf, features=FEATURES, auc=auc, auc_control=auc0),
                        os.path.join(args.out, "model.joblib"))
        except Exception as ex:
            if args.search_mode == "ml":
                print(f"STOP: model required but training failed: {ex}")
                return 2
            print(f"NO VALIDATED MODEL ({type(ex).__name__}: {ex}); "
                  "continuing in UNTRAINED SIGNAL mode, no ML score claims.")
            clf = None

    unk = featurize(g_unk, cache, eclipse=True)
    if unk.empty:
        print("No targets have usable optical light curves.")
        pd.DataFrame().to_csv(os.path.join(args.out, "candidates.csv"), index=False)
        return 0
    unk["score"] = clf.predict_proba(unk[FEATURES])[:, 1] if clf is not None else np.nan
    unk = unk.merge(g_unk[gaia_export_columns(g_unk)], on="source_id", how="left")
    sel = unk[passes_cuts(unk, require_model=clf is not None)]
    sel = sel.sort_values("score" if clf is not None else "log_fap",
                          ascending=clf is None)
    print(f"optical periodic-signal selections: {len(sel)} of {len(unk)} scored" +
          (" (Gaia variability unknown; match skipped)" if args.skip_gaia else ""))
    sel = add_eclipse_channel(unk, sel.head(args.max_prior), args.max_prior)
    keep, unverified, rejected = [], [], []
    for r in sel.itertuples():
        temporal = split_epoch_confirmation(r.ra, r.dec, r.period, cache, channel=getattr(r, "channel", "periodic"))
        rec = r._asdict(); rec.pop("Index", None); rec.update(channel_info(r))
        rec.update(temporal)
        rec["search_mode"] = "ml" if clf is not None else "untrained_signal"
        if not temporal["confirmed"]:
            rejected.append(rec)
            continue
        hits = prior_art(r.ra, r.dec)
        ident = f"VarWISE local {r.source_id}" if r.source_id < 0 else f"Gaia {r.source_id}"
        print(f"  {ident}  P={r.period:.5f} d  score={r.score:.3f}  prior-art hits: {hits or 'none'}")
        rec["prior_art"] = ";".join(hits)
        if any("ERROR" in x for x in hits):
            rec["status"] = "UNVERIFIED - prior-art service unavailable"
            unverified.append(rec)
        elif not hits:
            rec["status"] = "CANDIDATE - manual imaging and literature checks required"
            keep.append(rec)
    out = pd.DataFrame(keep).drop(columns=["Index"], errors="ignore")
    out.to_csv(os.path.join(args.out, "candidates.csv"), index=False)
    pd.DataFrame(unverified).to_csv(os.path.join(args.out, "unverified.csv"), index=False)
    pd.DataFrame(rejected).to_csv(os.path.join(args.out, "failed_temporal_check.csv"), index=False)
    anomalies = collect_anomaly_leads(unk, args, cache, args.out,
                                     already_handled=[x["source_id"] for x in keep + unverified])
    pd.DataFrame(anomalies).to_csv(os.path.join(args.out, "anomaly_leads.csv"), index=False)
    print(f"additional unverified anomaly leads: {len(anomalies)}")
    for r in out.itertuples():
        fold_plot(r.ra, r.dec, r.period, cache, os.path.join(args.out, f"fold_{r.source_id}.png"))
    print(f"\n{len(out)} candidates written to {args.out}/candidates.csv")
    print("MANUAL STEPS STILL REQUIRED before calling any of them new: check the folded PNG by eye, "
          "PanSTARRS/Legacy imaging for blends and a nearby bright star, a second period search "
          "(PDM / double the period for eclipsing binaries), MPChecker (asteroids), "
          "and a final VSX name search on the web site.")
    return 0


def smoke(args):
    """Run every network stage on a tiny box. Prints PASS/FAIL per stage with the error."""
    import traceback
    args.size = 2.0; args.max_vw = 300; args.n_train = 25
    state = {}
    fails, warnings_seen = [], []

    def stage(name, fn, optional=False):
        try:
            r = fn(); print(f"[PASS] {name}"); return r
        except Exception as ex:
            if optional:
                warnings_seen.append(name)
            else:
                fails.append(name)
            print(f"[{'WARN' if optional else 'FAIL'}] {name}: {type(ex).__name__}: {ex}")
            print("       " + traceback.format_exc().strip().splitlines()[-3].strip())
            return None
    vw = stage("1 VarWISE query (IRSA TAP)", lambda: get_varwise(args))
    if vw is not None:
        vw = vw.reset_index(drop=True); vw["uid"] = np.arange(len(vw))
        print("       columns:", list(vw.columns)[:15])
    if args.skip_gaia:
        g = None
        print("[SKIP] 2 Gaia DR3 cross-match (not needed with --skip-gaia)")
    else:
        g = stage("2 Gaia DR3 cross-match + vari_summary join",
                  lambda: gaia_match(vw)) if vw is not None else None
    if args.search_mode == "signal":
        tr = None
        print("[SKIP] 3 Gaia training-star query (label-free signal search)")
    else:
        tr = stage("3 Gaia training-star query", lambda: gaia_training_stars(args, args.n_train),
                   optional=args.search_mode != "ml")
    def z():
        d = ztf_lc(150.0, 30.0, args.out or ".")
        return d
    os.makedirs(args.out, exist_ok=True)
    def z2():
        row = tr.iloc[0] if tr is not None and len(tr) else None
        ra, de = (float(row["ra"]), float(row["dec"])) if row is not None else (args.ra0, args.dec0)
        d = ztf_lc(ra, de, args.out)
        if d is None: raise RuntimeError("ZTF API returned no rows for this star (try another)")
        print(f"       ZTF rows: {len(d)}"); return d
    d = stage("4 ZTF light-curve API", z2)
    if d is not None:
        stage("5 Lomb-Scargle features", lambda: lc_features(d["mjd"], d["mag"], d["magerr"]))
    def pa():
        h = prior_art(150.0, 2.2)
        errs = [x for x in h if "ERROR" in x]
        print("       hits:", h)
        if errs:
            raise RuntimeError("services unreachable or failing: " + "; ".join(errs))
    # A transient SIMBAD outage cannot authorize a candidate, but should not
    # prevent searching: each candidate will instead enter unverified.csv.
    stage("6 Prior-art services (VizieR + SIMBAD)", pa, optional=True)
    if fails:
        print("\nPaste the FAIL lines back to Claude and that stage gets fixed.")
    else:
        print("\nAll required stages PASS." +
              (" Prior-art outage: any affected results remain UNVERIFIED." if warnings_seen else ""))
    return len(fails)


def train_only(args):
    """Download training stars, featurize, train 80/20, save model.joblib. No scoring."""
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "ztf_cache"); os.makedirs(cache, exist_ok=True)
    tr = gaia_training_stars(args, args.n_train)
    tr = tr.rename(columns={"ra": "g_ra", "dec": "g_dec"})
    ft = featurize(tr, cache, label_col="label")
    ft.to_csv(os.path.join(args.out, "training_features.csv"), index=False)
    if ft.empty or "label" not in ft:
        print("NO VALIDATED MODEL: no usable labelled ZTF light curves")
        return 2
    print(f"usable Gaia proxy labels: {len(ft)} (known variables {int(ft['label'].sum())}, "
          f"not Gaia-flagged {int((1 - ft['label']).sum())})")
    try:
        clf, auc, auc0 = train_eval(ft)
    except RuntimeError as ex:
        print(f"NO VALIDATED MODEL: {ex}")
        return 2
    if auc < 0.85 or abs(auc0 - 0.5) > 0.12:
        print(f"NO VALIDATED MODEL: AUC={auc:.3f}, null={auc0:.3f}. "
              "Not saving a model that the search could mistake for validated.")
        return 2
    import joblib
    joblib.dump(dict(model=clf, features=FEATURES, auc=auc, auc_control=auc0),
                os.path.join(args.out, "model.joblib"))
    print(f"saved {args.out}/model.joblib   (use `run` to score VarWISE sources with it)")
    return 0


# ------------------------------------------------------------------- auto ---
class Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1); self.o = sys.__stdout__
    def write(self, x):
        self.o.write(x); self.f.write(x)
    def flush(self):
        self.o.flush(); self.f.flush()


def clone_args(a, **kw):
    d = dict(vars(a)); d.update(kw)
    return argparse.Namespace(**d)


def generate_boxes(size, bmin=15.0, dec_min=-25.0, dec_max=80.0, seed=1):
    """Sky boxes covered by ZTF, away from the Galactic plane, in a fixed random order."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    boxes = []
    dec = dec_min + size / 2
    while dec < dec_max:
        step = size / max(np.cos(np.radians(dec)), 0.1)
        ra = step / 2
        while ra < 360:
            boxes.append((round(ra, 3), round(dec, 3)))
            ra += step
        dec += size
    arr = np.array(boxes)
    b = SkyCoord(arr[:, 0] * u.deg, arr[:, 1] * u.deg).galactic.b.deg
    arr = arr[np.abs(b) >= bmin]
    rng = np.random.default_rng(seed); rng.shuffle(arr)
    return [(float(x), float(y)) for x, y in arr]


SEARCH_CUTS = dict(score=0.9, log_fap=-6.0, amp_over_err=5.0)


def passes_cuts(df, require_model=True):
    """A conservative optical signal gate; classifier confidence is optional.

    This enables an UNTRAINED signal search on a target without labelled stars.
    Such results must never be represented as calibrated ML discoveries.
    """
    signal = ((df["log_fap"] <= SEARCH_CUTS["log_fap"])
              & (~df["alias"].astype(bool))
              & (df["amp_over_err"] >= SEARCH_CUTS["amp_over_err"]))
    return signal & (df["score"] >= SEARCH_CUTS["score"]) if require_model else signal


def temporal_consistency_lc(lc, period, tolerance=0.03):
    """Test whether independent temporal halves both recover the proposed period."""
    if lc is None or len(lc) < 80:
        return dict(confirmed=False, early_period=None, late_period=None,
                    reason="fewer than 80 observations")
    lc = lc.sort_values("mjd")
    first, second = np.array_split(np.arange(len(lc)), 2)
    periods = []
    for idx in (first, second):
        q = lc.iloc[idx]
        ft = lc_features(q["mjd"], q["mag"], q["magerr"])
        periods.append(ft["period"] if ft is not None else None)
    def near(p, ref):
        return p is not None and any(abs(p / (ref * k) - 1) <= tolerance
                                     for k in (0.5, 1.0, 2.0))
    ok = all(near(p, period) for p in periods)
    return dict(confirmed=bool(ok), early_period=periods[0], late_period=periods[1],
                reason="period independently seen in both halves" if ok else
                       "period absent or inconsistent in a temporal half")


def split_epoch_confirmation(ra, dec, period, cache, tolerance=0.03, channel="periodic"):
    """Independent first-half / second-half check on actual ZTF observations."""
    lc = ztf_lc(float(ra), float(dec), cache)
    if channel == "eclipse":
        return temporal_consistency_ecl(lc, period, tolerance)
    return temporal_consistency_lc(lc, period, tolerance)


def injection_recovery(neg, cache, clf, n=160, seed=7):
    """Inject sinusoids into REAL constant-star ZTF light curves (their own sampling and noise)
    and measure how often the whole search chain recovers them."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in neg.sample(min(n, len(neg)), random_state=seed).itertuples():
        lc = ztf_lc(float(r.ra), float(r.dec), cache)
        if lc is None or len(lc) < 40:
            continue
        for A in (0.05, 0.1, 0.2, 0.4):
            P = 10 ** rng.uniform(np.log10(0.3), np.log10(20))
            m = lc["mag"].values + A * np.sin(2 * np.pi * lc["mjd"].values / P + rng.uniform(0, 6.28))
            f = lc_features(lc["mjd"].values, m, lc["magerr"].values)
            if f is None:
                continue
            f["score"] = float(clf.predict_proba(pd.DataFrame([f])[FEATURES])[:, 1][0])
            ratio = f["period"] / P
            good_p = any(abs(ratio - k) < 0.01 * k for k in (1.0, 0.5, 2.0))
            rows.append(dict(amp=A, period=P, ok=bool(good_p and passes_cuts(pd.DataFrame([f])).iloc[0])))
    d = pd.DataFrame(rows)
    return d


def train_and_validate(args, boxes, out):
    """Download real training stars, build features, train 80/20, validate. Returns pack or None."""
    cache = os.path.join(out, "ztf_cache"); os.makedirs(cache, exist_ok=True)
    frames, used = [], 0
    for ra0, dec0 in boxes[:16]:
        a = clone_args(args, ra0=ra0, dec0=dec0, size=args.box_size)
        try:
            tr = gaia_training_stars(a, args.train_per_box)
        except Exception as ex:
            print(f"  training box ({ra0:.1f},{dec0:.1f}) skipped: {type(ex).__name__}: {str(ex)[:120]}")
            continue
        tr = tr.rename(columns={"ra": "g_ra", "dec": "g_dec"})
        ft = featurize(tr, cache, label_col="label")
        if len(ft):
            frames.append(ft)
        used += 1
        tot = sum(len(f) for f in frames)
        n1 = int(sum(f["label"].sum() for f in frames))
        print(f"  training pool: {tot} usable stars ({n1} variables, {tot - n1} constants) from {used} boxes")
        if tot >= args.min_train and min(n1, tot - n1) >= 0.3 * tot:
            break
    if not frames:
        print("NO MODEL: no training light curves could be downloaded."); return None
    ft = pd.concat(frames, ignore_index=True)
    n1 = int(ft["label"].sum())
    if len(ft) < args.min_train or min(n1, len(ft) - n1) < 0.25 * len(ft):
        print(f"NO MODEL: only {len(ft)} usable training stars ({n1} variables); need >= {args.min_train} "
              f"with each class >= 25%. Nothing was trained.")
        return None
    try:
        clf, auc, auc0 = train_eval(ft)
    except Exception as ex:
        print(f"NO MODEL: {ex}"); return None
    if auc < 0.85 or abs(auc0 - 0.5) > 0.12:
        print(f"NO MODEL: validation failed (held-out AUC {auc:.3f}, shuffled control {auc0:.3f}). "
              f"Not using this model.")
        return None
    test = spatial_test_mask(ft)
    te = ft[test].copy()
    te["score"] = clf.predict_proba(te[FEATURES])[:, 1]
    sel = passes_cuts(te)
    tp = int(((te["label"] == 1) & sel).sum()); fp = int(((te["label"] == 0) & sel).sum())
    fn = int(((te["label"] == 1) & ~sel).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    inj = injection_recovery(ft[ft["label"] == 0], cache, clf)
    rep = dict(n_train=int((~test).sum()), n_heldout=int(test.sum()),
               split_strategy="source-balanced ~80/20 with spatially disjoint sky cells",
               labels="known Gaia variables vs sources lacking Gaia variability solutions (proxy negatives)",
               auc_heldout=float(auc),
               auc_shuffled_control=float(auc0), precision_at_search_cuts=float(prec),
               recall_at_search_cuts=float(rec), false_positives_heldout=fp,
               injection_recovery_by_amp={str(k): float(v) for k, v in
                                          inj.groupby("amp")["ok"].mean().items()} if len(inj) else {},
               injection_n=int(len(inj)))
    json.dump(rep, open(os.path.join(out, "validation_report.json"), "w"), indent=2)
    ft.to_csv(os.path.join(out, "training_features.csv"), index=False)
    import joblib
    pack = dict(model=clf, features=FEATURES, report=rep)
    joblib.dump(pack, os.path.join(out, "model.joblib"))
    print("\n=== VALIDATION (real data, held out by sky cell) ===")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print("====================================================\n")
    return pack


def process_box(args, clf, ra0, dec0, out, cache):
    a = clone_args(args, ra0=ra0, dec0=dec0, size=args.box_size)
    vw = get_varwise(a)
    if vw is None or len(vw) == 0:
        return [], [], [], 0
    vw = vw.reset_index(drop=True); vw["uid"] = np.arange(len(vw))
    if args.skip_gaia:
        g_unk = varwise_targets_without_gaia(vw)
    else:
        g = gaia_match(vw)
        g = attach_varwise_provenance(g[g["sep"] <= MATCH_ARCSEC], vw)
        g_unk = g[g["vari_id"].isna() & (g["n_gaia"] == 1)]
    unk = featurize(g_unk, cache, eclipse=True)
    if len(unk) == 0:
        return [], [], [], len(vw)
    unk["score"] = clf.predict_proba(unk[FEATURES])[:, 1] if clf is not None else np.nan
    unk = unk.merge(g_unk[gaia_export_columns(g_unk)], on="source_id", how="left")
    sel = unk[passes_cuts(unk, require_model=clf is not None)]
    sel = sel.sort_values("score" if clf is not None else "log_fap",
                          ascending=clf is None).head(args.max_prior)
    sel = add_eclipse_channel(unk, sel, args.max_prior)
    good, unverified = [], []
    for r in sel.itertuples():
        temporal = split_epoch_confirmation(r.ra, r.dec, r.period, cache, channel=getattr(r, "channel", "periodic"))
        if not temporal["confirmed"]:
            print(f"   Gaia {r.source_id} rejected by independent temporal check: "
                  f"{temporal['reason']}")
            continue
        hits = prior_art(r.ra, r.dec)
        real = [h for h in hits if "ERROR" not in h]
        errs = [h for h in hits if "ERROR" in h]
        rec = r._asdict(); rec.pop("Index", None); rec.update(channel_info(r))
        rec.update(box_ra=ra0, box_dec=dec0, prior_art=";".join(hits),
                   search_mode="ml" if clf is not None else "untrained_signal", **temporal)
        if real:
            continue
        if errs:
            rec["status"] = "UNVERIFIED - prior-art service unavailable"
            unverified.append(rec)
        else:
            rec["status"] = "CANDIDATE - manual imaging and literature checks required"
            good.append(rec)
            fold_plot(r.ra, r.dec, r.period, cache, os.path.join(out, f"fold_{r.source_id}.png"))
    anomalies = collect_anomaly_leads(unk, args, cache, out,
                                     already_handled=[x["source_id"] for x in good + unverified])
    return good, unverified, anomalies, len(vw)


def auto(args):
    os.makedirs(args.out, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.out, "auto.log"))
    t0 = time.time()
    print(f"=== AUTO START {time.strftime('%Y-%m-%d %H:%M:%S')}  out={args.out} ===")
    print("Data source: real archives only (IRSA VarWISE, Gaia DR3, ZTF). No synthetic data is used.")
    # 1. preflight: every stage must pass before anything is trained
    print("\n[1/3] PREFLIGHT (every archive stage on a tiny box)")
    pre = clone_args(args, out=os.path.join(args.out, "preflight"))
    nfail = smoke(pre)
    if nfail:
        print(f"\nPREFLIGHT FAILED ({nfail} stage(s)). Nothing was trained, nothing was searched.")
        print("Fix the FAIL lines above (or paste them to Claude) and rerun the same command.")
        return 1
    boxes = generate_boxes(args.box_size, args.bmin)
    print(f"sky boxes available: {len(boxes)}  (box {args.box_size} deg, |b| >= {args.bmin}, Dec > -25)")
    # 2. train + validate
    mp = os.path.join(args.out, "model.joblib")
    pack = None
    if args.search_mode == "signal":
        print("\n[2/3] UNTRAINED SIGNAL SEARCH requested; ML scores disabled.")
    elif os.path.exists(mp) and not args.retrain:
        import joblib; pack = joblib.load(mp)
        rep = pack.get("report", {})
        auc = rep.get("auc_heldout", pack.get("auc", 0.0))
        null = rep.get("auc_shuffled_control", pack.get("auc_control", 999.0))
        if auc < 0.85 or abs(null - 0.5) > 0.12:
            print(f"saved model fails validation: AUC={auc}, null={null}; retraining")
            pack = train_and_validate(args, boxes, args.out)
        else:
            print(f"\n[2/3] using saved validated model {mp}")
            print(json.dumps(rep or {"auc": auc, "auc_control": null}, indent=2))
    else:
        print("\n[2/3] DOWNLOAD TRAINING DATA, TRAIN, VALIDATE")
        pack = train_and_validate(args, boxes, args.out)
    if pack is None and args.search_mode == "ml":
        print("\nSTOPPED: strict ML mode requires a validated model.")
        return 2
    if pack is None:
        print("SEARCH CONTINUES WITHOUT MODEL: no 80/20 ML validation available. "
              "Signal candidates must meet the temporal and prior-art checks.")
    clf = pack["model"] if pack is not None else None
    # 3. search, resumable
    print("\n[3/3] SEARCH")
    sp = os.path.join(args.out, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else dict(done=[], n_candidates=0, scanned=0)
    cache = os.path.join(args.out, "ztf_cache"); os.makedirs(cache, exist_ok=True)
    cpath = os.path.join(args.out, "candidates.csv"); upath = os.path.join(args.out, "unverified.csv")
    apath = os.path.join(args.out, "anomaly_leads.csv")
    try:
        # Reserve the training region entirely: no training-region object is
        # eligible as a discovery target (avoids contaminated Gaia negatives).
        search_boxes = boxes[16:] if pack is not None else boxes
        for ra0, dec0 in search_boxes:
            key = f"{ra0:.3f},{dec0:.3f}"
            if key in state["done"]:
                continue
            if state["n_candidates"] >= args.target:
                break
            if time.time() - t0 > args.hours * 3600:
                print("time limit reached"); break
            print(f"\n-- box {key}  (done {len(state['done'])}, candidates {state['n_candidates']}/{args.target})")
            try:
                good, unver, anomal, nvw = process_box(args, clf, ra0, dec0, args.out, cache)
            except Exception as ex:
                print(f"   box failed: {type(ex).__name__}: {str(ex)[:200]}  (will retry next run)")
                continue
            for lst, p in ((good, cpath), (unver, upath), (anomal, apath)):
                if lst:
                    pd.DataFrame(lst).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
            state["done"].append(key); state["n_candidates"] += len(good); state["scanned"] += nvw
            json.dump(state, open(sp, "w"))
            print(f"   VarWISE objects {nvw}; new candidates {len(good)}; "
                  f"unverified {len(unver)}; anomaly leads {len(anomal)}")
    except KeyboardInterrupt:
        print("\ninterrupted - progress saved; rerun the same command to resume")
    print(f"\n=== DONE: {state['n_candidates']} candidates in {cpath} "
          f"after {len(state['done'])} boxes, {state['scanned']} VarWISE objects ===")
    print("These are CANDIDATES. Still required before calling any new: eyeball the fold_*.png, "
          "imaging for blends, double-period check for eclipsing binaries, MPChecker, VSX website search.")
    return 0


def brutetest(args):
    """Adversarial, reproducible SIMULATION: train on sinusoids; hide new morphologies.

    This is a falsification test, NOT a measurement of real-sky precision or
    proof that any astrophysical object is previously undiscovered.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(20261027)
    os.makedirs(args.out, exist_ok=True)
    n = args.brutetest_n
    pool = []
    print("[BRUTE] generating known sine and constant stars in independent sky cells", flush=True)
    for k in range(8 * n):
        label = int(k % 2 == 0)
        t, m, e, _ = synth_lc(rng, bool(label), n=150)
        f = lc_features(t, m, e, spp=4)
        f["label"] = label
        f["cell"] = sky_cell(rng.uniform(0, 30), rng.uniform(0, 30), cell=3)
        f["source_id"] = k
        pool.append(f)
    df = pd.DataFrame(pool)
    clf, auc, auc0 = train_eval(df)
    test = spatial_test_mask(df)
    te = df[test].copy()
    te["score"] = clf.predict_proba(te[FEATURES])[:, 1]
    te["found"] = passes_cuts(te)
    rows = []
    for kind in ("sine", "eclipse", "period_drift", "single_event", "constant"):
        print(f"[BRUTE] unseen target morphology: {kind}", flush=True)
        for _ in range(n):
            t = np.sort(rng.uniform(0, 1000, 160))
            e = np.full(len(t), rng.uniform(0.022, 0.038))
            p = 10 ** rng.uniform(np.log10(0.6), np.log10(15))
            amp = rng.uniform(0.18, 0.45)
            phase = rng.uniform(0, 1)
            m = np.full(len(t), rng.uniform(13.5, 17.5))
            if kind == "sine":
                m += amp * np.sin(2 * np.pi * (t / p + phase))
            elif kind == "eclipse":
                ph = (t / p + phase) % 1
                m += amp * (ph < 0.12)
            elif kind == "period_drift":
                m += amp * np.sin(2 * np.pi * (t / p + 0.6 * (t / 1000) ** 2 + phase))
            elif kind == "single_event":
                center = rng.uniform(200, 800)
                m += amp * np.exp(-0.5 * ((t - center) / 20) ** 2)
            m += rng.normal(0, e)
            f = lc_features(t, m, e, spp=4)
            if f is None:
                continue
            f["score"] = float(clf.predict_proba(pd.DataFrame([f])[FEATURES])[0, 1])
            ft = pd.DataFrame([f])
            phys = bool(passes_cuts(ft, require_model=False).iloc[0])
            ml = bool(passes_cuts(ft, require_model=True).iloc[0])
            temp = temporal_consistency_lc(pd.DataFrame({"mjd": t, "mag": m,
                                                          "magerr": e}), f["period"])["confirmed"]
            anomaly = outlier_lead_features(t, m, e)
            ef = eclipse_features(t, m, e)
            eg = ecl_pass(ef)
            et = bool(eg and temporal_consistency_ecl(pd.DataFrame({"mjd": t, "mag": m, "magerr": e}),
                                                      ef["ecl_period"])["confirmed"])
            pgood = any(abs(f["period"] / p / k - 1) < 0.03 for k in (0.5, 1, 2))
            rows.append(dict(kind=kind, target_label_available_to_model=False,
                             score=f["score"], signal_gate=phys, ml_gate=ml,
                             temporal_gate=bool(temp), signal_and_temporal=bool(phys and temp),
                             ml_and_temporal=bool(ml and temp),
                             anomaly_lead=anomaly["flag"], ecl_gate=eg, ecl_and_temporal=et,
                             recovered_period=bool(pgood) if kind in
                             ("sine", "eclipse", "period_drift") else None))
    d = pd.DataFrame(rows)
    summary = {}
    for kind, g in d.groupby("kind", sort=False):
        summary[kind] = dict(n=len(g),
                             mean_model_score=round(float(g["score"].mean()), 4),
                             signal_fraction=round(float(g["signal_and_temporal"].mean()), 4),
                             ml_fraction=round(float(g["ml_and_temporal"].mean()), 4),
                             unverified_anomaly_fraction=round(float(g["anomaly_lead"].mean()), 4),
                             eclipse_channel_fraction=round(float(g["ecl_and_temporal"].mean()), 4))
    report = dict(provenance="SYNTHETIC - not real-sky performance",
                  training_morphologies=["sine", "constant"],
                  withheld_target_morphologies=["eclipse", "period_drift", "single_event"],
                  training_sources=int((~test).sum()), validation_sources=int(test.sum()),
                  validation_auc=float(auc), shuffled_auc=float(auc0),
                  heldout_positive_signal_recall=float(te.loc[te.label == 1, "found"].mean()),
                  heldout_negative_false_positive_rate=float(te.loc[te.label == 0, "found"].mean()),
                  target_classes=summary)
    d.to_csv(os.path.join(args.out, "brutetest_individual_trials.csv"), index=False)
    with open(os.path.join(args.out, "brutetest_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== BRUTE RESULTS: synthetic only ===")
    print(json.dumps(report, indent=2))
    print("This does not validate detection of unknown physical classes or prove catalog novelty.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "brutetest", "probe", "smoke", "train", "run", "auto"])
    ap.add_argument("--ra0", type=float, default=150.0)
    ap.add_argument("--dec0", type=float, default=30.0)
    ap.add_argument("--size", type=float, default=4.0, help="box side, degrees")
    ap.add_argument("--out", default="out")
    ap.add_argument("--table", default=None)
    ap.add_argument("--ra-col", default=None); ap.add_argument("--dec-col", default=None)
    ap.add_argument("--max-vw", type=int, default=20000)
    ap.add_argument("--n-train", type=int, default=400, help="stars per class")
    ap.add_argument("--max-prior", type=int, default=60, help="max survivors sent to prior-art")
    ap.add_argument("--target", type=int, default=5, help="auto: stop after this many verified candidates")
    ap.add_argument("--hours", type=float, default=12.0, help="auto: wall-clock limit for the search")
    ap.add_argument("--box-size", type=float, default=6.0, help="auto: box side in degrees")
    ap.add_argument("--bmin", type=float, default=15.0, help="auto: min |galactic latitude|")
    ap.add_argument("--train-per-box", type=int, default=150)
    ap.add_argument("--min-train", type=int, default=400, help="auto: usable training stars required")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--search-mode", choices=["auto", "ml", "signal"], default="auto",
                    help="auto: validated ML if possible, otherwise untrained optical signal search; "
                         "ml: require validated ML; signal: never require target training labels")
    ap.add_argument("--skip-gaia", action="store_true",
                    help="search every VarWISE sky position, even without a Gaia match; "
                         "uses negative local IDs, never pretends they are Gaia IDs")
    ap.add_argument("--include-anomalies", action="store_true",
                    help="also queue nonperiodic/eclipsing outlier patterns as unverified "
                         "anomaly_leads.csv (not confirmed discoveries)")
    ap.add_argument("--brutetest-n", type=int, default=50,
                    help="trials per target morphology in the synthetic adversarial test")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(selftest())
    if a.cmd == "brutetest":
        sys.exit(brutetest(a))
    if a.cmd == "probe":
        probe(); return
    if a.cmd == "smoke":
        sys.exit(1 if smoke(a) else 0)
    if a.cmd == "auto":
        sys.exit(auto(a))
    if a.cmd == "train":
        sys.exit(train_only(a))
    sys.exit(run(a))


if __name__ == "__main__":
    main()
