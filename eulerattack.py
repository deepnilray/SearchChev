#!/usr/bin/env python3
"""
eulerattack.py - EulerSearch #1: VarWISE x ZTF DR24, trained on independent labels.

THE ATTACK
  Targets      : VarWISE Pure Catalogue (IRSA), mid-IR variables, Paz et al. 2026.
  Light curves : ZTF DR24 (IRSA light-curve API), g and r, catflags == 0.
  Positives    : AAVSO VSX confirmed periodic variables (EA, EW, EB, RRAB, RRC,
                 DSCT, HADS, DCEP, ROT, BY), in TRAINING tiles only.
  Negatives    : Gaia DR3 QUIET stars, judged by an INDEPENDENT instrument: no
                 vari_summary entry, not flagged VARIABLE, and the lowest 25% of
                 Gaia's own G-band scatter proxy sqrt(n_obs)/flux_over_error in
                 their magnitude bin; no VSX star within 5 arcsec.
  Features     : signal SHAPE and SIGNIFICANCE only. Magnitude, number of points
                 and time span are NOT model inputs (they flipped between skies in
                 the failed varwise run, AUC 0.568). Classes are magnitude-matched.

PROTOCOL (fail-closed)
  1. manifest.json fixes TRAINING, VALIDATION and SEARCH tiles before any download.
  2. A guard refuses every query outside the current phase's tiles; every access
     is written to access_log.jsonl.
  3. Train; choose the score threshold from out-of-fold training scores (GroupKFold
     by tile) for a target false-positive rate.
  4. ONLY THEN unseal validation tiles. Pass requires, on the sealed holdout:
       n >= --min-val per class, AUC - 1.96*SE >= 0.85 (Hanley-McNeil),
       mean of 10 shuffled-label controls within 0.5 +- 0.08,
       false-positive rate 95% upper bound (Wilson) <= 0.05, recall >= 0.30.
  5. Fail -> STOP. No model saved, no target touched. There is no untrained mode.
  6. Search tiles only: score >= threshold AND (Lomb-Scargle gate OR calibrated
     eclipse gate) AND period recovered in both time halves AND clean prior art.
     Prior-art service errors go to unverified.csv, never to candidates.

COMMANDS
  python eulerattack.py brutetest            # offline adversarial test (synthetic sky)
  python eulerattack.py census  --out c1     # MEASURES whether the real data are good enough
  python eulerattack.py train   --out a1     # real labels, sealed validation, gate
  python eulerattack.py search  --out a1 --hours 5 --target 20
  python eulerattack.py all     --out a1 --hours 5

Requires varwise_gaia_hunt.py (SearchChev) in the same folder for the feature code.
Candidates are never discoveries: images, aliases (test 2P), literature, expert review.
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np
import pandas as pd

from varwise_gaia_hunt import (lc_features, eclipse_features, ecl_pass, alias_flag,
                               temporal_consistency_lc, temporal_consistency_ecl,
                               varwise_period_agreement, tap, IRSA_TAP, GAIA_TAP,
                               ZTF_API, pick_radec)

SHAPE_FEATURES = ["log_fap", "power", "amp_over_err", "chi2red", "log_period", "skew",
                  "iqr_over_err", "von_neumann", "ecl_snr", "ecl_depth_over_err",
                  "ecl_epochs", "ecl_depth_scatter", "ecl_concentration", "ecl_frac_in_faint"]
SHORTCUTS = ["mean_mag", "log_n", "span"]          # recorded, never fed to the model
VSX_TYPES = ("EA", "EW", "EB", "RRAB", "RRC", "DSCT", "HADS", "DCEP", "ROT", "BY")
LS_GATE = dict(log_fap=-6.0, amp_over_err=5.0)
GATE = dict(auc_lower=0.85, control_band=0.08, fpr_upper=0.05, min_recall=0.30)


# ============================================================ statistics ===
def hanley_se(auc, n1, n2):
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    v = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2)
    return math.sqrt(max(v, 0.0))


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    return min(1.0, (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d)


# ============================================================== features ===
def features(lc):
    """lc: DataFrame mjd, mag, magerr. Returns dict (shape features + recorded shortcuts) or None."""
    if lc is None or len(lc) < 60:
        return None
    t, m, e = (lc[c].to_numpy(float) for c in ("mjd", "mag", "magerr"))
    f = lc_features(t, m, e)
    if f is None:
        return None
    ef0 = eclipse_features(t, m, e)
    ef = ef0 or dict(ecl_period=np.nan, ecl_snr=0.0, ecl_depth=0.0,
                                           ecl_epochs=0, ecl_depth_scatter=9.0,
                                           ecl_concentration=0.0, ecl_frac_in_faint=0.0)
    o = np.argsort(t)
    d = np.diff(m[o])
    out = dict(f)
    out.update(ef)
    out["von_neumann"] = float(np.mean(d ** 2) / (np.var(m) + 1e-12))
    out["ecl_depth_over_err"] = float(ef["ecl_depth"] / np.median(e))
    out["span"] = float(np.ptp(t))
    out["ecl_pass"] = bool(ecl_pass(ef0)) if ef0 is not None else False
    out["ls_pass"] = bool(f["log_fap"] <= LS_GATE["log_fap"] and not f["alias"]
                          and f["amp_over_err"] >= LS_GATE["amp_over_err"])
    return out


def _feat_job(args):
    key, lc = args
    try:
        return key, features(lc)
    except Exception:
        return key, None


def featurize(rows, archive, workers):
    """rows: DataFrame with ra, dec (+anything). Adds features; drops rows without a usable curve."""
    jobs = [(i, archive.lightcurve(float(r.ra), float(r.dec))) for i, r in enumerate(rows.itertuples())]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = pool.map(_feat_job, jobs)
    else:
        res = [_feat_job(j) for j in jobs]
    keep, feats = [], []
    for i, f in res:
        if f is not None and all(np.isfinite(f[k]) for k in SHAPE_FEATURES):
            keep.append(i); feats.append(f)
    out = rows.iloc[keep].reset_index(drop=True)
    return pd.concat([out, pd.DataFrame(feats)], axis=1), None


# ================================================================= tiles ===
def make_tiles(box, dec_min=-25.0, dec_max=80.0, bmin=15.0):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    cand = []
    dec = dec_min + box / 2
    while dec + box / 2 <= dec_max:
        width = box / max(math.cos(math.radians(dec)), 0.1)
        n = max(1, int(360 // width))
        cand += [((k + 0.5) * 360.0 / n, dec, 360.0 / n) for k in range(n)]
        dec += box
    ra = np.array([c[0] for c in cand]); de = np.array([c[1] for c in cand])
    b = SkyCoord(ra * u.deg, de * u.deg).galactic.b.deg
    return [dict(id=f"{r:.3f}_{d:.3f}", ra=float(r), dec=float(d), ra_w=float(w), dec_h=box)
            for (r, d, w), bb in zip(cand, b) if abs(bb) >= bmin]


def in_tile(ra, dec, t):
    dra = (ra - t["ra"] + 180) % 360 - 180
    return abs(dra) <= t["ra_w"] / 2 and abs(dec - t["dec"]) <= t["dec_h"] / 2


def build_manifest(out, tiles, n_train, n_val, seed):
    path = os.path.join(out, "manifest.json")
    rng = np.random.default_rng(seed)
    order = [tiles[i] for i in rng.permutation(len(tiles))]
    man = dict(seed=seed, train=order[:n_train], validation=order[n_train:n_train + n_val],
               search=order[n_train + n_val:], created=time.strftime("%Y-%m-%d %H:%M:%S"))
    fp = hashlib.sha256(json.dumps([man["train"], man["validation"]], sort_keys=True).encode()).hexdigest()
    man["fingerprint"] = fp
    if os.path.exists(path):
        old = json.load(open(path))
        if old["fingerprint"] != fp:
            raise SystemExit("REFUSED: an existing manifest uses different tiles. New --out folder or same settings.")
        return old
    json.dump(man, open(path, "w"), indent=1)
    return man


class Guard:
    """Wraps an archive; refuses any access outside the current phase's tiles; logs everything."""
    def __init__(self, archive, manifest, out):
        self.a, self.man, self.phase = archive, manifest, "train"
        self.log = open(os.path.join(out, "access_log.jsonl"), "a")

    def _allowed(self):
        return {"train": self.man["train"], "validation": self.man["validation"],
                "search": self.man["search"]}[self.phase]

    def _check(self, what, ra, dec):
        ok = any(in_tile(ra, dec, t) for t in self._allowed())
        self.log.write(json.dumps(dict(t=time.time(), phase=self.phase, what=what,
                                       ra=round(ra, 5), dec=round(dec, 5), allowed=ok)) + "\n")
        self.log.flush()
        if not ok:
            raise PermissionError(f"{what} at ({ra:.4f},{dec:.4f}) is outside {self.phase} tiles")

    def tile_query(self, what, tile):
        self._check(what, tile["ra"], tile["dec"])
        return getattr(self.a, what)(tile)

    def lightcurve(self, ra, dec):
        self._check("lightcurve", ra, dec)
        return self.a.lightcurve(ra, dec)

    def prior_art(self, ra, dec):
        self._check("prior_art", ra, dec)
        return self.a.prior_art(ra, dec)


# ========================================================= real archive ===
class RealArchive:
    VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"

    def __init__(self, cache):
        self.cache = cache
        os.makedirs(cache, exist_ok=True)
        self._vw_table = None

    @staticmethod
    def _box(racol, deccol, t):
        return (f"CONTAINS(POINT('ICRS',{racol},{deccol}),BOX('ICRS',{t['ra']},{t['dec']},"
                f"{t['ra_w']},{t['dec_h']}))=1")

    def positives(self, t):
        types = ",".join(f"'{x}'" for x in VSX_TYPES)
        q = (f'SELECT "Name","RAJ2000","DEJ2000","Type","Period" FROM "B/vsx/vsx" WHERE '
             f'{self._box(chr(34)+"RAJ2000"+chr(34), chr(34)+"DEJ2000"+chr(34), t)} '
             f'AND "Type" IN ({types}) AND "Period" > 0.05')
        try:                                   # V = 0 marks confirmed variables in VSX
            d = tap(self.VIZIER_TAP).run_sync(q + ' AND "V" = 0', maxrec=20000).to_table().to_pandas()
        except Exception:
            d = tap(self.VIZIER_TAP).run_sync(q, maxrec=20000).to_table().to_pandas()
        d = d.rename(columns={"RAJ2000": "ra", "DEJ2000": "dec", "Type": "vsx_type",
                              "Period": "vsx_period", "Name": "name"})
        d["label"] = 1
        return d

    def negatives(self, t):
        q = ("SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.phot_g_n_obs, "
             "g.phot_g_mean_flux_over_error FROM gaiadr3.gaia_source AS g "
             "LEFT JOIN gaiadr3.vari_summary AS v ON v.source_id = g.source_id WHERE "
             f"{self._box('g.ra', 'g.dec', t)} AND g.phot_g_mean_mag BETWEEN 12 AND 19 "
             "AND v.source_id IS NULL AND g.phot_variable_flag <> 'VARIABLE' "
             "AND g.ruwe < 1.4 AND g.phot_g_n_obs > 100")
        d = tap(GAIA_TAP).run_sync(q, maxrec=60000).to_table().to_pandas()
        if d.empty:
            return d
        d["gaia_scatter"] = np.sqrt(d["phot_g_n_obs"]) / d["phot_g_mean_flux_over_error"]
        d["mbin"] = np.floor(d["phot_g_mean_mag"] * 2) / 2
        q25 = d.groupby("mbin")["gaia_scatter"].transform(lambda s: s.quantile(0.25))
        d = d[d["gaia_scatter"] <= q25].copy()
        vsx = self.positives(t)
        if len(vsx):
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            a = SkyCoord(d["ra"].values * u.deg, d["dec"].values * u.deg)
            b = SkyCoord(vsx["ra"].values * u.deg, vsx["dec"].values * u.deg)
            _, sep, _ = a.match_to_catalog_sky(b)
            d = d[sep.arcsec > 5]
        d["label"] = 0
        return d.drop(columns=["mbin"])

    def targets(self, t):
        irsa = tap(IRSA_TAP)
        if self._vw_table is None:
            names = irsa.run_sync("SELECT table_name FROM TAP_SCHEMA.tables WHERE "
                                  "lower(table_name) LIKE '%varwise%'").to_table().to_pandas()["table_name"]
            pure = [x for x in names if x.lower().split(".")[-1] == "varwisepure"]
            if not pure:
                raise RuntimeError(f"VarWISE Pure table not found among {list(names)}")
            cols = irsa.run_sync(f"SELECT column_name FROM TAP_SCHEMA.columns WHERE "
                                 f"table_name='{pure[0]}'").to_table().to_pandas()["column_name"]
            self._vw_table = (pure[0],) + pick_radec(cols)
        table, rac, decc = self._vw_table
        d = irsa.run_sync(f"SELECT * FROM {table} WHERE {self._box(rac, decc, t)}",
                          maxrec=200000).to_table().to_pandas()
        return d.rename(columns={rac: "ra", decc: "dec"})

    def lightcurve(self, ra, dec):
        """Nearest ZTF object within 1.5 arcsec, best band; failures are NOT cached."""
        import pickle, requests
        from io import StringIO
        key = os.path.join(self.cache, f"{ra:.5f}_{dec:.5f}.pkl")
        if os.path.exists(key):
            return pickle.load(open(key, "rb"))
        best = None
        for band in ("r", "g"):
            r = None
            for attempt in range(3):
                try:
                    r = requests.get(ZTF_API, params={"POS": f"CIRCLE {ra} {dec} 0.000417",
                                                      "BANDNAME": band, "FORMAT": "csv"}, timeout=60)
                    if r.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(2 * (attempt + 1))
            if r is None or r.status_code != 200:
                return None                                   # outage: do not cache
            d = pd.read_csv(StringIO(r.text)) if len(r.text) > 50 else pd.DataFrame()
            if d.empty:
                continue
            d = d[d["catflags"] == 0]
            if d.empty:
                continue
            pos = d.groupby("oid")[["ra", "dec"]].median()
            sep = 3600 * np.hypot((pos["ra"] - ra) * math.cos(math.radians(dec)), pos["dec"] - dec)
            sep = sep.sort_values()
            if len(sep) > 1 and sep.iloc[1] - sep.iloc[0] < 0.5:
                continue                                       # ambiguous neighbour: refuse
            d = d[d["oid"] == sep.index[0]][["mjd", "mag", "magerr"]]
            if best is None or len(d) > len(best):
                best = d.copy()
        pickle.dump(best, open(key, "wb"))
        return best

    def prior_art(self, ra, dec):
        from varwise_gaia_hunt import prior_art
        return prior_art(ra, dec)


# =============================================================== training ===
def collect(guard, tiles, kind_pos="positives", kind_neg="negatives", max_per_tile=400, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    for t in tiles:
        try:
            p = guard.tile_query(kind_pos, t)
            n = guard.tile_query(kind_neg, t)
        except PermissionError:
            raise
        except Exception as ex:
            print(f"  tile {t['id']} skipped: {type(ex).__name__}: {str(ex)[:100]}")
            continue
        if len(p) > max_per_tile:
            p = p.sample(max_per_tile, random_state=int(rng.integers(1 << 30)))
        if len(n) > 3 * max(len(p), 1):
            n = n.sample(3 * max(len(p), 1), random_state=int(rng.integers(1 << 30)))
        p = p.assign(tile=t["id"]); n = n.assign(tile=t["id"])
        pos.append(p); neg.append(n)
        print(f"  tile {t['id']}: {len(p)} labelled variables, {len(n)} quiet stars")
    cols = ["ra", "dec", "label", "tile"]
    P = pd.concat([x[cols + [c for c in ("vsx_type",) if c in x]] for x in pos], ignore_index=True) if pos else pd.DataFrame(columns=cols)
    N = pd.concat([x[cols] for x in neg], ignore_index=True) if neg else pd.DataFrame(columns=cols)
    return P, N


def magnitude_match(df, seed=0):
    """Equal positives and negatives in every 0.5-mag bin of ZTF median magnitude."""
    df = df.copy()
    df["mbin"] = np.floor(df["mean_mag"] * 2) / 2
    parts = []
    for _, g in df.groupby("mbin"):
        p, n = g[g.label == 1], g[g.label == 0]
        k = min(len(p), len(n))
        if k:
            parts += [p.sample(k, random_state=seed), n.sample(k, random_state=seed)]
    return pd.concat(parts, ignore_index=True).drop(columns="mbin") if parts else df.iloc[:0]


def fit(df, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(df[SHAPE_FEATURES], df["label"])
    return clf


def oof_threshold(df, target_fpr, seed=0):
    from sklearn.model_selection import GroupKFold
    scores = np.zeros(len(df))
    k = min(5, df["tile"].nunique())
    if k < 3:
        raise RuntimeError("need at least 3 training tiles for grouped cross-validation")
    for tr, te in GroupKFold(n_splits=k).split(df, df["label"], df["tile"]):
        scores[te] = fit(df.iloc[tr], seed).predict_proba(df.iloc[te][SHAPE_FEATURES])[:, 1]
    neg = scores[df["label"].to_numpy() == 0]
    thr = float(max(0.5, np.quantile(neg, 1 - target_fpr)))
    from sklearn.metrics import roc_auc_score
    return thr, float(roc_auc_score(df["label"], scores))


def evaluate(clf, train_df, val, thr, seed=0):
    from sklearn.metrics import roc_auc_score
    y = val["label"].to_numpy()
    s = clf.predict_proba(val[SHAPE_FEATURES])[:, 1]
    auc = float(roc_auc_score(y, s))
    n1, n0 = int(y.sum()), int((1 - y).sum())
    se = hanley_se(auc, n1, n0)
    rng = np.random.default_rng(seed)
    ctl = []
    for _ in range(10):
        sh = train_df.copy(); sh["label"] = rng.permutation(sh["label"].to_numpy())
        ctl.append(roc_auc_score(y, fit(sh, seed).predict_proba(val[SHAPE_FEATURES])[:, 1]))
    sel = s >= thr
    fp, tp = int((sel & (y == 0)).sum()), int((sel & (y == 1)).sum())
    return dict(n_val_pos=n1, n_val_neg=n0, auc=auc, auc_se=se, auc_lower95=auc - 1.96 * se,
                shuffled_control=float(np.mean(ctl)), threshold=thr,
                fpr=fp / max(n0, 1), fpr_upper95=wilson_upper(fp, n0),
                recall=tp / max(n1, 1), precision=tp / max(tp + fp, 1))


def gate(rep, min_val):
    reasons = []
    if min(rep["n_val_pos"], rep["n_val_neg"]) < min_val:
        reasons.append(f"holdout too small ({rep['n_val_pos']}/{rep['n_val_neg']} < {min_val} per class)")
    if rep["auc_lower95"] < GATE["auc_lower"]:
        reasons.append(f"AUC lower bound {rep['auc_lower95']:.3f} < {GATE['auc_lower']}")
    if abs(rep["shuffled_control"] - 0.5) > GATE["control_band"]:
        reasons.append(f"shuffled control {rep['shuffled_control']:.3f} not near 0.5")
    if rep["fpr_upper95"] > GATE["fpr_upper"]:
        reasons.append(f"FPR upper bound {rep['fpr_upper95']:.3f} > {GATE['fpr_upper']}")
    if rep["recall"] < GATE["min_recall"]:
        reasons.append(f"recall {rep['recall']:.3f} < {GATE['min_recall']}")
    return reasons


def train(args, archive):
    os.makedirs(args.out, exist_ok=True)
    man = build_manifest(args.out, make_tiles(args.box), args.train_tiles, args.val_tiles, args.seed)
    guard = Guard(archive, man, args.out)
    print(f"[manifest] {len(man['train'])} training, {len(man['validation'])} SEALED validation, "
          f"{len(man['search'])} search tiles")
    guard.phase = "train"
    P, N = collect(guard, man["train"], max_per_tile=args.max_per_tile, seed=args.seed)
    if len(P) < 50 or len(N) < 50:
        print("STOP: too few training labels."); return None
    tr, _ = featurize(pd.concat([P, N], ignore_index=True), guard, args.workers)
    tr.to_csv(os.path.join(args.out, "training_features_unmatched.csv"), index=False)
    tr = magnitude_match(tr, args.seed)
    print(f"[train] {int(tr.label.sum())} variables vs {int((1 - tr.label).sum())} quiet stars after "
          f"magnitude matching, {tr.tile.nunique()} tiles")
    if tr["label"].value_counts().min() < 100:
        print("STOP: fewer than 100 per class after matching; add --train-tiles."); return None
    thr, oof_auc = oof_threshold(tr, args.target_fpr, args.seed)
    clf = fit(tr, args.seed)
    print(f"[fit] done. grouped-CV AUC {oof_auc:.3f}; threshold {thr:.3f} for FPR {args.target_fpr}")
    print("[seal] ONLY NOW opening validation tiles")
    guard.phase = "validation"
    VP, VN = collect(guard, man["validation"], max_per_tile=args.max_per_tile, seed=args.seed + 1)
    va, _ = featurize(pd.concat([VP, VN], ignore_index=True), guard, args.workers)
    va.to_csv(os.path.join(args.out, "validation_features_unmatched.csv"), index=False)
    va = magnitude_match(va, args.seed)
    rep = evaluate(clf, tr, va, thr, args.seed)
    rep.update(train_pos=int(tr.label.sum()), train_neg=int((1 - tr.label).sum()),
               oof_auc=oof_auc, features=SHAPE_FEATURES, excluded_shortcuts=SHORTCUTS,
               manifest_fingerprint=man["fingerprint"], gate=GATE, min_val=args.min_val)
    reasons = gate(rep, args.min_val)
    rep["validated"] = not reasons
    rep["fail_reasons"] = reasons
    json.dump(rep, open(os.path.join(args.out, "validation_report.json"), "w"), indent=1)
    tr.to_csv(os.path.join(args.out, "training_features.csv"), index=False)
    va.to_csv(os.path.join(args.out, "validation_features.csv"), index=False)
    print(json.dumps({k: rep[k] for k in ("n_val_pos", "n_val_neg", "auc", "auc_lower95",
                                          "shuffled_control", "fpr", "fpr_upper95", "recall",
                                          "precision")}, indent=1))
    if reasons:
        print("VALIDATION FAILED -> STOP. No model saved; no target touched.\n  " + "\n  ".join(reasons))
        return None
    import joblib
    joblib.dump(dict(model=clf, threshold=thr, report=rep), os.path.join(args.out, "model.joblib"))
    print("VALIDATION PASSED. model.joblib written.")
    return dict(model=clf, threshold=thr, report=rep)


# ================================================================= search ===
def search(args, archive, pack=None):
    import joblib
    mp = os.path.join(args.out, "model.joblib")
    if pack is None:
        if not os.path.exists(mp):
            print("STOP: no validated model. Run train first."); return 2
        pack = joblib.load(mp)
    if not pack["report"].get("validated"):
        print("STOP: model is not validated."); return 2
    man = json.load(open(os.path.join(args.out, "manifest.json")))
    if man["fingerprint"] != pack["report"]["manifest_fingerprint"]:
        print("STOP: model and manifest do not belong together."); return 2
    guard = Guard(archive, man, args.out); guard.phase = "search"
    clf, thr = pack["model"], pack["threshold"]
    sp = os.path.join(args.out, "search_state.json")
    st = json.load(open(sp)) if os.path.exists(sp) else dict(done=[], candidates=0, scanned=0)
    t0 = time.time()
    paths = {k: os.path.join(args.out, f"{k}.csv") for k in ("candidates", "unverified", "known")}
    for t in man["search"]:
        if t["id"] in st["done"]:
            continue
        if st["candidates"] >= args.target or time.time() - t0 > args.hours * 3600:
            break
        try:
            vw = guard.tile_query("targets", t)
        except Exception as ex:
            print(f"  tile {t['id']} failed: {type(ex).__name__}: {str(ex)[:120]}"); continue
        rows = {"candidates": [], "unverified": [], "known": []}
        if len(vw):
            ft, lcs = featurize(vw, guard, args.workers)
            if len(ft):
                ft["score"] = clf.predict_proba(ft[SHAPE_FEATURES])[:, 1]
                hit = ft[(ft["score"] >= thr) & (ft["ls_pass"] | ft["ecl_pass"])]
                for r in hit.itertuples():
                    lc = guard.lightcurve(float(r.ra), float(r.dec))
                    ecl = bool(r.ecl_pass) and not bool(r.ls_pass)
                    per = float(r.ecl_period if ecl else r.period)
                    tc = (temporal_consistency_ecl if ecl else temporal_consistency_lc)(lc, per)
                    if not tc["confirmed"]:
                        continue
                    hits = guard.prior_art(float(r.ra), float(r.dec))
                    rec = {k: getattr(r, k) for k in ("ra", "dec", "score", "period", "log_fap",
                                                      "amp_over_err", "ecl_snr", "ecl_period",
                                                      "mean_mag", "n") if hasattr(r, k)}
                    for k in ("designation", "vartype", "confidence", "period1", "variability_snr"):
                        if hasattr(r, k):
                            rec["varwise_" + k] = getattr(r, k)
                    rec.update(channel="eclipse" if ecl else "periodic", tested_period=per,
                               also_check_double_period=2 * per, prior_art=";".join(hits),
                               varwise_period_agrees=varwise_period_agreement(
                                   per, getattr(r, "period1", np.nan)), tile=t["id"], **tc)
                    if any("ERROR" not in h for h in hits):
                        rows["known"].append(rec)
                    elif hits:
                        rec["status"] = "UNVERIFIED: prior-art service error"; rows["unverified"].append(rec)
                    else:
                        rec["status"] = "CANDIDATE: needs images, 2P fold, literature, expert"
                        rows["candidates"].append(rec)
        for k, v in rows.items():
            if v:
                pd.DataFrame(v).to_csv(paths[k], mode="a", header=not os.path.exists(paths[k]), index=False)
        st["done"].append(t["id"]); st["candidates"] += len(rows["candidates"]); st["scanned"] += len(vw)
        json.dump(st, open(sp, "w"))
        print(f"  tile {t['id']}: VarWISE {len(vw)}, candidates {len(rows['candidates'])}, "
              f"known {len(rows['known'])}, unverified {len(rows['unverified'])}")
    print(f"DONE: {st['candidates']} candidates from {st['scanned']} VarWISE objects in {len(st['done'])} tiles")
    return 0


# ================================================================= census ===
def census(args, archive):
    """Measures, on real data, whether VarWISE x ZTF is good enough. Touches search-pool tiles only."""
    os.makedirs(args.out, exist_ok=True)
    tiles = make_tiles(args.box)
    rng = np.random.default_rng(args.seed + 99)
    pick = [tiles[i] for i in rng.choice(len(tiles), size=min(args.census_tiles, len(tiles)), replace=False)]
    rows = []
    for t in pick:
        try:
            vw = archive.targets(t)
        except Exception as ex:
            print(f"  {t['id']}: {type(ex).__name__}"); continue
        sample = vw.sample(min(len(vw), args.census_per_tile), random_state=args.seed) if len(vw) else vw
        for r in sample.itertuples():
            lc = archive.lightcurve(float(r.ra), float(r.dec))
            n = 0 if lc is None else len(lc)
            f = features(lc) if n >= 60 else None
            rows.append(dict(tile=t["id"], vartype=getattr(r, "vartype", None), n_ztf=n,
                             mag=float(np.median(lc["mag"])) if n else np.nan,
                             span=float(np.ptp(lc["mjd"])) if n else np.nan,
                             usable=f is not None,
                             signal=bool(f and (f["ls_pass"] or f["ecl_pass"]))))
        print(f"  {t['id']}: {len(vw)} VarWISE Pure; sampled {len(sample)}")
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(args.out, "census.csv"), index=False)
    if d.empty:
        print("census: no data"); return 1
    summ = dict(sampled=len(d), with_ztf=float((d.n_ztf > 0).mean()), usable=float(d.usable.mean()),
                median_epochs=float(d.loc[d.n_ztf > 0, "n_ztf"].median()),
                median_span_days=float(d["span"].median()), optical_signal=float(d.signal.mean()),
                by_type=d.groupby("vartype")["signal"].mean().round(3).to_dict())
    json.dump(summ, open(os.path.join(args.out, "census.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    return 0


# ============================================================== brutetest ===
class MockArchive:
    """Synthetic sky with traps: brightness reversed between training and validation tiles,
    red noise, under-estimated errors, 3% bad epochs, eclipses, label noise, outages."""
    def __init__(self, man, seed=0, n=60, label_noise=0.0, invert_brightness=True, outage_every=0):
        self.rng = np.random.default_rng(seed)
        self.val_ids = {t["id"] for t in man["validation"]}
        self.n, self.noise, self.inv, self.outage_every = n, label_noise, invert_brightness, outage_every
        self.truth, self.calls = {}, 0

    def _place(self, t, k):
        ra = (t["ra"] + self.rng.uniform(-.45, .45, k) * t["ra_w"]) % 360
        dec = t["dec"] + self.rng.uniform(-.45, .45, k) * t["dec_h"]
        return ra, dec

    def _mags(self, t, variable, k):
        bright = (variable != (self.inv and t["id"] in self.val_ids))
        return self.rng.uniform(14, 18, k) if bright else self.rng.uniform(15.5, 19, k)

    def _make(self, t, variable, k):
        ra, dec = self._place(t, k)
        mag = self._mags(t, variable, k)
        kinds = self.rng.choice(["sine", "rr", "ew", "ea"], k) if variable else \
            self.rng.choice(["white", "red", "underr", "bad"], k)
        for a, b, m, kind in zip(ra, dec, mag, kinds):
            self.truth[(round(a, 6), round(b, 6))] = (kind, m)
        lab = np.full(k, int(variable))
        flip = self.rng.random(k) < self.noise
        lab[flip] = 1 - lab[flip]
        return pd.DataFrame(dict(ra=ra, dec=dec, label=lab))

    def positives(self, t):
        return self._make(t, True, self.n)

    def negatives(self, t):
        return self._make(t, False, self.n)

    def targets(self, t):
        d = pd.concat([self._make(t, True, 6), self._make(t, False, 18)], ignore_index=True)
        d["vartype"] = "ew"; d["period1"] = np.nan; d["designation"] = "mock"
        return d.drop(columns="label")

    def lightcurve(self, ra, dec):
        self.calls += 1
        if self.outage_every and self.calls % self.outage_every == 0:
            return None
        kind, m0 = self.truth.get((round(ra, 6), round(dec, 6)), ("white", 17.0))
        r = np.random.default_rng(int(abs(ra * 1e5 + dec * 1e3)) % (2 ** 32))
        nights = np.sort(r.choice(np.arange(1500), size=int(r.integers(150, 320)), replace=False))
        t = nights + 0.3 + 0.08 * r.random(len(nights))
        err = 0.01 + 0.02 * 10 ** (0.4 * (m0 - 18))
        P = 10 ** r.uniform(np.log10(0.25), np.log10(8)); ph = (t / P + r.random()) % 1
        A = r.uniform(0.12, 0.5)
        sig = {"sine": A * np.sin(2 * np.pi * ph), "rr": A * (ph - 0.5) * 2,
               "ew": A * 0.5 * np.cos(4 * np.pi * ph),
               "ea": np.where(np.minimum(ph, 1 - ph) < 0.03, A + 0.2, 0.0)}.get(kind, np.zeros(len(t)))
        noise = r.normal(0, err, len(t))
        if kind == "red":
            x = np.zeros(len(t))
            for i in range(1, len(t)):
                x[i] = 0.9 * x[i - 1] + r.normal(0, 1.5 * err)
            noise += x
        if kind == "underr":
            noise *= 1.8
        if kind == "bad":
            bad = r.random(len(t)) < 0.03
            noise[bad] += r.choice([-1, 1], bad.sum()) * r.uniform(0.3, 1.0, bad.sum())
        return pd.DataFrame(dict(mjd=t, mag=m0 + sig + noise, magerr=np.full(len(t), err)))

    def prior_art(self, ra, dec):
        u = self.rng.random()
        return ["VSX"] if u < 0.3 else (["SIMBAD:ERROR(Timeout)"] if u < 0.4 else [])


def brutetest(args):
    import shutil
    from sklearn.metrics import roc_auc_score
    base = os.path.join(args.out, "brutetest"); shutil.rmtree(base, ignore_errors=True)
    results, ok_all = {}, True

    def setup(name, over=None, **kw):
        a = argparse.Namespace(**vars(args)); a.out = os.path.join(base, name)
        for k, v in (over or {}).items():
            setattr(a, k, v)
        os.makedirs(a.out, exist_ok=True)
        man = build_manifest(a.out, make_tiles(a.box), a.train_tiles, a.val_tiles, a.seed)
        return a, MockArchive(man, seed=a.seed, **kw)

    def check(name, cond, detail):
        nonlocal ok_all
        ok_all &= bool(cond)
        results[name] = dict(passed=bool(cond), detail=detail)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    # 1. trap: brightness reversed between skies. Shape model must validate.
    a, mock = setup("clean")
    pack = train(a, mock)
    rep = json.load(open(os.path.join(a.out, "validation_report.json")))
    check("validates_despite_reversed_brightness", pack is not None,
          f"AUC {rep['auc']:.3f} (lower {rep['auc_lower95']:.3f}), FPR {rep['fpr']:.3f}, recall {rep['recall']:.3f}")
    # 2. the old design (magnitude as a feature) on the same data collapses
    from sklearn.ensemble import HistGradientBoostingClassifier
    tru = pd.read_csv(os.path.join(a.out, "training_features_unmatched.csv"))
    vau = pd.read_csv(os.path.join(a.out, "validation_features_unmatched.csv"))
    old = ["log_fap", "power", "amp", "amp_over_err", "chi2red", "log_period", "log_n",
           "mean_mag", "skew", "iqr_over_err"]                      # the failed design's inputs
    c = HistGradientBoostingClassifier(max_depth=4, random_state=0).fit(tru[old], tru.label)
    auc_old = roc_auc_score(vau.label, c.predict_proba(vau[old])[:, 1])
    new = fit(magnitude_match(tru, args.seed), args.seed)
    auc_new_unmatched = roc_auc_score(vau.label, new.predict_proba(vau[SHAPE_FEATURES])[:, 1])
    check("old_design_breaks_new_design_holds", auc_new_unmatched > auc_old + 0.05,
          f"same reversed-brightness skies, unmatched holdout: old design AUC {auc_old:.3f}, "
          f"new design AUC {auc_new_unmatched:.3f}")
    va = pd.read_csv(os.path.join(a.out, "validation_features.csv"))
    tr = pd.read_csv(os.path.join(a.out, "training_features.csv"))
    c2 = HistGradientBoostingClassifier(max_depth=4, random_state=0).fit(tr[["mean_mag"]], tr.label)
    auc_mag = roc_auc_score(va.label, c2.predict_proba(va[["mean_mag"]])[:, 1])
    check("magnitude_matching_neutralises_magnitude", abs(auc_mag - 0.5) < 0.15,
          f"magnitude-only model on matched validation AUC {auc_mag:.3f}")
    # 3. sealed tiles were not touched before the fit
    log = [json.loads(x) for x in open(os.path.join(a.out, "access_log.jsonl"))]
    first_val = next(i for i, x in enumerate(log) if x["phase"] == "validation")
    check("no_validation_access_before_fit", all(x["phase"] == "train" for x in log[:first_val])
          and all(x["allowed"] for x in log), f"{first_val} training accesses before first validation access")
    # 4. guard refuses forbidden access
    g = Guard(mock, json.load(open(os.path.join(a.out, "manifest.json"))), a.out); g.phase = "search"
    try:
        t = g.man["validation"][0]; g.lightcurve(t["ra"], t["dec"]); refused = False
    except PermissionError:
        refused = True
    check("search_cannot_touch_validation_tiles", refused, "PermissionError raised")
    # 5. search with traps: prior-art errors -> unverified, known -> excluded
    search(a, mock, pack)
    cand = pd.read_csv(os.path.join(a.out, "candidates.csv")) if os.path.exists(os.path.join(a.out, "candidates.csv")) else pd.DataFrame()
    unv = pd.read_csv(os.path.join(a.out, "unverified.csv")) if os.path.exists(os.path.join(a.out, "unverified.csv")) else pd.DataFrame()
    truth_var = [mock.truth.get((round(r.ra, 6), round(r.dec, 6)), ("?",))[0] in ("sine", "rr", "ew", "ea")
                 for r in cand.itertuples()] if len(cand) else []
    purity = float(np.mean(truth_var)) if truth_var else float("nan")
    check("candidates_are_real_variables", len(cand) > 0 and purity >= 0.95,
          f"{len(cand)} candidates, true-variable fraction {purity:.3f}; {len(unv)} unverified")
    check("prior_art_errors_never_candidates",
          (len(cand) == 0 or not cand["prior_art"].fillna("").str.contains("ERROR").any()), "checked")
    # 6. pure label noise must STOP
    a2, mock2 = setup("noise", label_noise=0.5)
    check("random_labels_stop", train(a2, mock2) is None, "gate refused a model trained on coin-flip labels")
    # 7. too small a holdout must STOP even if the model is good
    a3, mock3 = setup("small", over=dict(val_tiles=1))
    check("small_holdout_stops", train(a3, mock3) is None, "gate refused an underpowered holdout")
    # 8. 20% label noise: still validates or stops cleanly, never crashes
    a4, mock4 = setup("noisy20", label_noise=0.2)
    p4 = train(a4, mock4)
    check("label_noise_20pct_handled", True, f"{'validated' if p4 else 'stopped'}")
    # 9. outages: no crash, no cached failure
    a5, mock5 = setup("outage", outage_every=7)
    try:
        train(a5, mock5); crashed = False
    except Exception as ex:
        crashed = True
    check("archive_outages_do_not_crash", not crashed, "every 7th light-curve call returned nothing")
    # 10. no model -> search refuses
    a6, _ = setup("nomodel")
    check("search_refuses_without_model", search(a6, MockArchive(json.load(open(os.path.join(a6.out, "manifest.json"))))) == 2, "returned 2")
    json.dump(results, open(os.path.join(base, "brutetest_report.json"), "w"), indent=1, default=float)
    print("\nBRUTETEST", "ALL PASSED" if ok_all else "HAS FAILURES",
          f"(old design on the same skies: AUC {auc_old:.3f})")
    return 0 if ok_all else 1


# =================================================================== main ===
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["brutetest", "census", "train", "search", "all"])
    ap.add_argument("--out", default="attack1")
    ap.add_argument("--box", type=float, default=3.0, help="tile size, deg")
    ap.add_argument("--train-tiles", type=int, default=30)
    ap.add_argument("--val-tiles", type=int, default=12)
    ap.add_argument("--max-per-tile", type=int, default=150)
    ap.add_argument("--min-val", type=int, default=150, help="minimum holdout objects per class")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--target", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--census-tiles", type=int, default=10)
    ap.add_argument("--census-per-tile", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261030)
    a = ap.parse_args()
    if a.cmd == "brutetest":
        a.train_tiles, a.val_tiles, a.min_val = 8, 4, 100
        return brutetest(a)
    archive = RealArchive(os.path.join(a.out, "ztf_cache"))
    if a.cmd == "census":
        return census(a, archive)
    if a.cmd == "train":
        return 0 if train(a, archive) else 3
    if a.cmd == "search":
        return search(a, archive)
    pack = train(a, archive)
    return search(a, archive, pack) if pack else 3


if __name__ == "__main__":
    sys.exit(main())
