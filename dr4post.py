"""dr4post.py - the steps between and after the dr4engine passes.

train      fit the 'missed variable' classifier on samples from sealed TRAINING ranges (Gaia-flagged variables vs
           unflagged stars, magnitude-matched), set the threshold at a 1% false-positive rate from out-of-fold scores,
           then open the sealed VALIDATION ranges once and apply the pre-registered gate. Writes model/missed.joblib
           and model/validation_report.json; refuses to write a model that fails the gate.
collect    merge every range's candidate lists into catalogue files, ranked, with run statistics.
followup   ZTF and NEOWISE checks (followup.py) plus prior-art for the top dips / impact / anomaly candidates.
predict    score the pre-registered SearchChev list against DR4: is each star flagged variable in DR4, and does
           DR4's own light curve show our ZTF period.

usage: python dr4post.py train   <results dir> <model dir>
       python dr4post.py collect <results dir> <catalogue dir>
       python dr4post.py followup <catalogue dir> [n_top]
       python dr4post.py predict <results dir> <prediction.csv> <out.csv>
"""
import glob, json, math, os, sys
import numpy as np, pandas as pd
import dr4engine as E

GATE = dict(auc_lower95=0.85, shuffled_within=0.08, fpr_upper95=0.05, recall=0.30, min_per_class=150)


def _samples(res, phase):
    fs = [f for f in glob.glob(os.path.join(res, "ranges", "*", "sample.parquet"))
          if json.load(open(os.path.join(os.path.dirname(f), "summary.json"))).get("phase") == phase]
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def _match(d, seed=0):
    """Equal numbers of positives and negatives in every 0.5-mag bin, so brightness carries no information."""
    d = d[np.isfinite(d.phot_g_mean_mag)].copy(); d["bin"] = (d.phot_g_mean_mag / 0.5).astype(int)
    out = []
    for _, g in d.groupby("bin"):
        p, n = g[g.y == 1], g[g.y == 0]; k = min(len(p), len(n))
        if k:
            out += [p.sample(k, random_state=seed), n.sample(k, random_state=seed)]
    return pd.concat(out, ignore_index=True) if out else d.iloc[:0]


def _wilson_upper(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def train(res, outdir, seed=20261202):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    import joblib
    os.makedirs(outdir, exist_ok=True)
    tr = _samples(res, "train"); va = _samples(res, "validation")
    for d in (tr, va):
        if len(d):
            d["y"] = d.flagged_variable.fillna(False).astype(int)
    tr = _match(tr, seed)
    if len(tr) == 0 or tr.y.sum() < GATE["min_per_class"]:
        raise SystemExit(f"STOP: too few training stars ({len(tr)})")
    X, y = E.missed_matrix(tr), tr.y.values
    groups = tr.source_id.values // (2 ** 35) // (4 ** 7)          # HEALPix level-5 cell: neighbours stay together
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(5).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=seed).fit(X.iloc[a], y[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    thr = float(np.quantile(oof[y == 0], 0.99))                   # 1% false-positive rate on training negatives
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=seed).fit(X, y)
    shuf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=seed).fit(
        X, np.random.default_rng(seed).permutation(y))
    # ---- sealed validation, opened once ----
    va = _match(va, seed)
    Xv, yv = E.missed_matrix(va), va.y.values
    s = model.predict_proba(Xv)[:, 1]; auc = float(roc_auc_score(yv, s))
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    npos, nneg = int(yv.sum()), int((1 - yv).sum())
    se = math.sqrt((auc * (1 - auc) + (npos - 1) * (q1 - auc ** 2) + (nneg - 1) * (q2 - auc ** 2)) / max(npos * nneg, 1))
    fp = int(((s >= thr) & (yv == 0)).sum()); tp = int(((s >= thr) & (yv == 1)).sum())
    rep = dict(n_train=len(tr), n_val_pos=npos, n_val_neg=nneg, auc=auc, auc_lower95=auc - 1.96 * se,
               shuffled_control=float(roc_auc_score(yv, shuf.predict_proba(Xv)[:, 1])), threshold=thr,
               fpr=fp / max(nneg, 1), fpr_upper95=_wilson_upper(fp, nneg), recall=tp / max(npos, 1),
               features=E.MISSED_FEATURES, note="negatives are unflagged stars, some of them real variables, "
                                                "so the false-positive rate is an upper bound")
    fails = []
    if rep["auc_lower95"] < GATE["auc_lower95"]: fails.append("auc")
    if abs(rep["shuffled_control"] - 0.5) > GATE["shuffled_within"]: fails.append("shuffled control")
    if rep["fpr_upper95"] > GATE["fpr_upper95"]: fails.append("fpr")
    if rep["recall"] < GATE["recall"]: fails.append("recall")
    if min(npos, nneg) < GATE["min_per_class"]: fails.append("too few validation stars")
    rep["gate_failures"] = fails; rep["validated"] = not fails
    json.dump(rep, open(os.path.join(outdir, "validation_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    if fails:
        raise SystemExit("STOP: gate failed: " + ", ".join(fails))
    joblib.dump(dict(model=model, threshold=thr, features=E.MISSED_FEATURES), os.path.join(outdir, "missed.joblib"))


def collect(res, outdir):
    os.makedirs(outdir, exist_ok=True)
    summ = [json.load(open(f)) for f in glob.glob(os.path.join(res, "ranges", "*", "summary.json"))]
    stats = dict(ranges=len(summ), stars=int(sum(s.get("n", 0) for s in summ)),
                 by_phase=pd.Series([s.get("phase") for s in summ]).value_counts().to_dict())
    for name in ("dips", "impact", "step", "anomaly", "missed"):
        fs = glob.glob(os.path.join(res, "ranges", "*", f"{name}.csv*"))
        parts = []
        for f in fs:
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if len(d):
                parts.append(d.assign(range=os.path.basename(os.path.dirname(f))))
        if not parts:
            stats[name] = 0; continue
        d = pd.concat(parts, ignore_index=True).drop_duplicates("source_id")
        if "known_variable" in d:
            d = d.sort_values(["known_variable", "score"], ascending=[True, False])
        else:
            d = d.sort_values("score", ascending=False)
        d.to_csv(os.path.join(outdir, f"{name}.csv.gz" if name == "missed" else f"{name}.csv"), index=False)
        stats[name] = len(d)
        if "known_variable" in d:
            stats[name + "_unflagged"] = int((~d.known_variable.astype(bool)).sum())
    json.dump(stats, open(os.path.join(outdir, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


def followup(catdir, n_top=300):
    import followup as FU
    from concurrent.futures import ThreadPoolExecutor
    from prior_art import prior_art
    for name in ("impact", "dips", "step", "anomaly"):
        p = os.path.join(catdir, f"{name}.csv")
        if not os.path.exists(p):
            continue
        C = pd.read_csv(p)
        C = C[~C.known_variable.astype(bool)].head(n_top) if "known_variable" in C else C.head(n_top)
        if C.empty:
            continue
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(FU.assess, C.itertuples()))

        def pa(r):
            try:
                return ";".join(sorted(prior_art(float(r.ra), float(r.dec), 5)))
            except Exception as e:
                return "ERROR"
        with ThreadPoolExecutor(6) as ex:
            C["prior_art"] = list(ex.map(pa, C.itertuples()))
        C = C.merge(pd.DataFrame(res), on="source_id", how="left")
        if name == "impact":
            C["priority"] = C.ir_brightening.fillna(False).astype(int) * 2 + C.ztf_confirms.fillna(False).astype(int)
        else:
            C["priority"] = C.ztf_confirms.fillna(False).astype(int)
        C.sort_values(["priority", "score"], ascending=False).to_csv(os.path.join(catdir, f"{name}_followup.csv"), index=False)
        print(name, len(C), "checked;", int((C.priority > 0).sum()), "with independent support")


def predict(res, pred_csv, out_csv):
    """Score the dated SearchChev prediction: DR4 flag for each star, and DR4's own light curve folded on our period."""
    P = pd.read_csv(pred_csv)
    P["source_id"] = P.source_id.astype("int64")
    P["hp8"] = P.source_id.map(E.healpix8)
    ranges = E.discover("gdr4")
    rows = []
    for rng, files in ranges.items():
        a, b = map(int, rng.split("-"))
        sel = P[(P.hp8 >= a) & (P.hp8 <= b)]
        if sel.empty:
            continue
        tmp = os.path.join(res, "tmp_predict"); os.makedirs(tmp, exist_ok=True)
        local = {k: E.fetch(v, os.path.join(tmp, os.path.basename(v))) for k, v in files.items()}
        src = E.source_table(local, "gdr4").set_index("source_id")
        want = set(sel.source_id)
        lcs = {sid: lc for sid, lc in E.light_curves(local["epoch"]) if sid in want}
        for r in sel.itertuples():
            d = dict(source_id=r.source_id, dr4_flagged=bool(src.flagged_variable.get(r.source_id, False)),
                     dr4_class=str(src.gaia_class.get(r.source_id, "")), dr4_lc=r.source_id in lcs)
            if r.source_id in lcs and np.isfinite(getattr(r, "period", np.nan)):
                from astropy.timeseries import LombScargle
                t, m, e = lcs[r.source_id]["G"]; ls = LombScargle(t, m, e)
                best, fb = 0.0, 1 / r.period
                for mult in (0.5, 1.0, 2.0):
                    f0 = 1 / (r.period * mult)
                    fz = np.linspace(f0 * 0.999, f0 * 1.001, 41); pz = float(ls.power(fz).max())
                    if pz > best:
                        best, fb = pz, f0
                d["dr4_power_at_ztf_period"] = best
                d["dr4_fap_at_ztf_period"] = float(ls.false_alarm_probability(best, minimum_frequency=fb * 0.999,
                                                                              maximum_frequency=fb * 1.001))
            rows.append(d)
        for p in local.values():
            os.remove(p)
    out = P.merge(pd.DataFrame(rows), on="source_id", how="left"); out.to_csv(out_csv, index=False)
    s = dict(n=len(out), dr4_flagged=float(out.dr4_flagged.mean()),
             period_confirmed=float((out.dr4_fap_at_ztf_period < 1e-3).mean()))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(sys.argv[2], sys.argv[3])
    elif cmd == "collect":
        collect(sys.argv[2], sys.argv[3])
    elif cmd == "followup":
        followup(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300)
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3], sys.argv[4])
