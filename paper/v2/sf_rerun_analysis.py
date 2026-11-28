"""Selection function from the rerun (branch sf-rerun; paper checklist A1-A3, A7, B1-B3).
Reads shard*/sf_rows.csv and sf_state.json from a local copy of branch sf-rerun (argv[1]) and the SCoPe lookup
(scope_sf.parquet). Writes sf_rerun_summary.json and sf_rerun_bins.csv.
C = N_flagged / (N_flagged + N_quiet) among detections, for each detector variant and each way of counting."""
import glob, json, sys
import numpy as np, pandas as pd
from scipy.stats import norm
D = sys.argv[1]
R = pd.concat([pd.read_csv(f) for f in glob.glob(f'{D}/shard*/sf_rows.csv')], ignore_index=True)
ST = [json.load(open(f)) for f in glob.glob(f'{D}/shard*/sf_state.json')]
cnt = {k: v for s in ST for k, v in s.get('counts', {}).items()}
done = sorted(set(str(t) for s in ST for t in s['done'])); skipped = sorted(set(str(t) for s in ST for t in s.get('skipped', [])))
R = R[~R.gaia_class.fillna('').isin(['AGN', 'GALAXY'])]                       # non-stellar Gaia classes removed (A3)
R['tile'] = R.tile.astype(str)
try:
    SC = pd.read_parquet('scope_sf.parquet'); vs = set(SC[SC.scope_var].gaia_id); seen = set(SC.gaia_id)
except Exception:
    SC = None
def comp(f, q): return f / (f + q) if f + q else np.nan
def wilson(k, n, z=norm.ppf(0.975)):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d; return (c - h, c + h)
rng = np.random.default_rng(20261128)
tiles = sorted(R.tile.unique() if len(R) else [])
out = dict(tiles_done=len(done), tiles_skipped=len(skipped), counts={g: {k: sum(c[g][k] for c in cnt.values()) for k in next(iter(cnt.values()))[g]} for g in ('flagged', 'quiet')} if cnt else {})
bins_rows = []
for det in ('det_orig', 'det_clf', 'det_free'):
    d = R[R[det]]
    F, Q = d[d.group == 'flagged'], d[d.group == 'quiet']
    res = dict(F=len(F), Q=len(Q), C=comp(len(F), len(Q)))
    fc = F.groupby('tile').size().reindex(done, fill_value=0).values; qc = Q.groupby('tile').size().reindex(done, fill_value=0).values
    bs = [comp(fc[i].sum(), qc[i].sum()) for i in (rng.integers(0, len(done), len(done)) for _ in range(2000))]
    res['C_lo'], res['C_hi'] = np.percentile(bs, [2.5, 97.5])
    if SC is not None:
        fs, qs = F[F.source_id.isin(vs)], Q[Q.source_id.isin(vs)]
        res.update(F_in_scope=int(F.source_id.isin(seen).sum()), Q_in_scope=int(Q.source_id.isin(seen).sum()),
                   F_sc=len(fs), Q_sc=len(qs), C_sc=comp(len(fs), len(qs)))
    # flagged fraction against ZTF S/N, period, amplitude, G and Gaia observation count (A7)
    for var, edges in (('amp_over_err', [5, 8, 12, 20, 1e9]), ('period', list(np.logspace(np.log10(0.05), np.log10(500), 9))),
                       ('amp', [0, 0.05, 0.1, 0.2, 0.4, 10]), ('G', list(np.arange(13, 19.0, 1.0)) + [18.5]),
                       ('phot_g_n_obs', [0, 200, 300, 400, 600, 1e5])):
        for lo, hi in zip(edges[:-1], edges[1:]):
            f = ((F[var] >= lo) & (F[var] < hi)).sum(); q = ((Q[var] >= lo) & (Q[var] < hi)).sum()
            if f + q == 0: continue
            l, h = wilson(f, f + q)
            row = dict(det=det, var=var, lo=lo, hi=hi, F=int(f), Q=int(q), C=comp(f, q), C_lo=l, C_hi=h)
            if SC is not None:
                a = ((fs[var] >= lo) & (fs[var] < hi)).sum(); b = ((qs[var] >= lo) & (qs[var] < hi)).sum()
                row.update(F_sc=int(a), Q_sc=int(b), C_sc=comp(a, b))
            bins_rows.append(row)
    out[det] = res
# Gaia class composition of flagged detections (classifier vs classifier-free)
for det in ('det_clf', 'det_free'):
    out[f'{det}_flagged_classes'] = R[R[det] & (R.group == 'flagged')].gaia_class.fillna('none').value_counts().head(12).to_dict()
pd.DataFrame(bins_rows).to_csv('sf_rerun_bins.csv', index=False)
json.dump(out, open('sf_rerun_summary.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
