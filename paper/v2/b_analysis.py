"""Paper checklist section B (B1-B4): selection-function tables from the 280 denominator partitions.
Inputs: /home/claude/sf/F.pkl, Q.pkl (detections among Gaia-flagged and Gaia-quiet stars), /home/claude/sf/shard*/denom_rows.csv
(all pre-filtered flagged stars, for recall by Gaia class), scope_sf.parquet (SCoPe lookup, 24/24 bundles).
Outputs: b_tables.json, b_bins.csv, numbers_b.tex (macros prefixed \\nb...), table_sn.tex, table_pa.tex, table_class.tex.
C = N_flagged / (N_flagged + N_quiet); three ways of counting the quiet side (all detections, supported, SCoPe-confirmed on
both sides). Intervals: 2.5-97.5 percentiles of a bootstrap over partitions (2,000 resamples, seed 20261128)."""
import json, glob
import numpy as np, pandas as pd
SF = '/home/claude/sf'
F = pd.read_pickle(f'{SF}/F.pkl'); Q = pd.read_pickle(f'{SF}/Q.pkl')
for d in (F, Q):
    d['tile'] = d.tile.astype(str)
    for c in ('amp', 'period', 'amp_over_err', 'G', 'score'): d[c] = d[c].astype(float)
SC = pd.read_parquet('scope_sf.parquet'); vs = set(SC[SC.scope_var].gaia_id); seen = set(SC.gaia_id)
F['sc'] = F.source_id.isin(vs); Q['sc'] = Q.source_id.isin(vs); F['sin'] = F.source_id.isin(seen); Q['sin'] = Q.source_id.isin(seen)
F = F[~F.gaia_class.fillna('').isin(['AGN', 'GALAXY'])]                       # non-stellar Gaia classes removed (A3)
tiles = sorted(set(str(t) for t in json.load(open('/home/claude/searchchev/freeze/denominator_tiles.json'))))
rng = np.random.default_rng(20261128); BOOT = [rng.integers(0, len(tiles), len(tiles)) for _ in range(2000)]

def C_with_ci(f, q):
    """f, q: DataFrames of flagged and quiet detections in a bin -> (C, lo, hi), tile bootstrap."""
    fc = f.groupby('tile').size().reindex(tiles, fill_value=0).values; qc = q.groupby('tile').size().reindex(tiles, fill_value=0).values
    if fc.sum() + qc.sum() == 0: return (np.nan, np.nan, np.nan)
    c = fc.sum() / (fc.sum() + qc.sum())
    bs = [fc[i].sum() / max(fc[i].sum() + qc[i].sum(), 1) for i in BOOT]
    return (c, *np.percentile(bs, [2.5, 97.5]))

ways = {'all': (F, Q), 'sup': (F, Q[Q.conf]), 'sc': (F[F.sc], Q[Q.sc])}
out, rows = {}, []
for w, (f, q) in ways.items():
    out[f'C_{w}'] = C_with_ci(f, q); out[f'N_{w}'] = (len(f), len(q))
edges = {'amp_over_err': [5, 8, 12, 20, 1e9], 'period': list(np.logspace(np.log10(0.05), np.log10(500), 9)),
         'amp': [0, 0.05, 0.1, 0.2, 0.4, 10], 'G': [13, 14, 15, 16, 17, 18, 18.5]}
for var, e in edges.items():
    for lo, hi in zip(e[:-1], e[1:]):
        r = dict(var=var, lo=lo, hi=hi)
        for w, (f, q) in ways.items():
            fb, qb = f[(f[var] >= lo) & (f[var] < hi)], q[(q[var] >= lo) & (q[var] < hi)]
            r[f'F_{w}'], r[f'Q_{w}'] = len(fb), len(qb)
            r[f'C_{w}'], r[f'lo_{w}'], r[f'hi_{w}'] = C_with_ci(fb, qb) if len(fb) + len(qb) >= 30 else (np.nan,) * 3
        rows.append(r)
B = pd.DataFrame(rows); B.to_csv('b_bins.csv', index=False)

# period x amplitude (B2): SCoPe-confirmed on both sides and all detections
pe = [0.05, 0.5, 1.6, 5, 16, 160]; ae = [0.05, 0.1, 0.2, 0.4, 10]
PA = []
for plo, phi in zip(pe[:-1], pe[1:]):
    for alo, ahi in zip(ae[:-1], ae[1:]):
        sel = lambda d: d[(d.period >= plo) & (d.period < phi) & (d.amp >= alo) & (d.amp < ahi)]
        fa, qa = sel(F), sel(Q); fs, qs = fa[fa.sc], qa[qa.sc]
        PA.append(dict(plo=plo, phi=phi, alo=alo, ahi=ahi, F=len(fa), Q=len(qa),
                       C_all=len(fa) / (len(fa) + len(qa)) if len(fa) + len(qa) >= 20 else np.nan,
                       C_sc=len(fs) / (len(fs) + len(qs)) if len(fs) + len(qs) >= 20 else np.nan))
PA = pd.DataFrame(PA)

# recall by Gaia class among flagged stars that passed the pre-filter (B2)
D = pd.concat([pd.read_csv(f) for f in glob.glob(f'{SF}/shard*/denom_rows.csv')], ignore_index=True)
D['gaia_class'] = D.gaia_class.fillna('none').replace('', 'none')
cls = D.groupby('gaia_class').agg(n=('detected', 'size'), det=('detected', 'sum')).sort_values('n', ascending=False)
cls['recall'] = cls.det / cls.n
# purity (B4): SCoPe-variable fraction of detections inside SCoPe coverage
out['purity_Q'] = (int(Q.sc.sum()), int(Q.sin.sum())); out['purity_F'] = (int(F.sc.sum()), int(F.sin.sum()))
out['purity_Q_by_sn'] = {f'{lo:g}-{hi:g}': (int(Q[(Q.amp_over_err >= lo) & (Q.amp_over_err < hi)].sc.sum()),
                         int(Q[(Q.amp_over_err >= lo) & (Q.amp_over_err < hi)].sin.sum())) for lo, hi in zip([5, 8, 12, 20], [8, 12, 20, 1e9])}
json.dump(dict(out=out, classes=cls.head(15).reset_index().to_dict('records'), pa=PA.to_dict('records')),
          open('b_tables.json', 'w'), indent=1, default=float)

# ---- LaTeX: macros and tables
M = {}
def p(x): return f"{100 * x:.0f}"
for w in ways:
    c, lo, hi = out[f'C_{w}']; M[f'C{w}'] = p(c); M[f'C{w}lo'] = p(lo); M[f'C{w}hi'] = p(hi); M[f'M{w}'] = p(1 - c)
M['purQ'] = p(out['purity_Q'][0] / out['purity_Q'][1]); M['purF'] = p(out['purity_F'][0] / out['purity_F'][1])
pq = out['purity_Q_by_sn']; M['purQlow'] = p(pq['5-8'][0] / pq['5-8'][1]); M['purQhigh'] = p(pq['20-1e+09'][0] / pq['20-1e+09'][1])
sn = B[B['var'] == 'amp_over_err']
M['Csnlow'] = p(sn.iloc[0].C_sc); M['Csnhigh'] = p(sn.iloc[-1].C_sc); M['Csnlowall'] = p(sn.iloc[0].C_all); M['Csnhighall'] = p(sn.iloc[-1].C_all)
for k in ('ECL', 'RR', 'RS', 'SOLAR_LIKE', 'none', 'AGN'):
    if k in cls.index: M[f'rec{k.replace("_", "").replace("none", "noclass")}'] = p(cls.loc[k, 'recall']); M[f'n{k.replace("_", "").replace("none", "noclass")}'] = f"{int(cls.loc[k, 'n']):,}"
with open('numbers_b.tex', 'w') as fh:
    for k, v in sorted(M.items()):
        fh.write(f"\\newcommand{{\\nb{k}}}{{{v}}}\n")

def cell(r, w):
    return '--' if pd.isna(r[f'C_{w}']) else f"{100 * r[f'C_{w}']:.0f} ({100 * r[f'lo_{w}']:.0f}--{100 * r[f'hi_{w}']:.0f})"
lab = {'amp_over_err': 'ZTF S/N (amplitude / median error)', 'period': 'Period (d)', 'amp': 'Amplitude (mag)', 'G': '$G$ (mag)'}
def rng_(var, lo, hi):
    if hi > 1e8 or (var == 'amp' and hi > 5): return f"$\\ge{lo:g}$"
    if var == 'period': return f"{lo:.2f}--{hi:.2f}" if hi < 1 else (f"{lo:.1f}--{hi:.1f}" if hi < 10 else f"{lo:.0f}--{hi:.0f}")
    return f"{lo:g}--{hi:g}"
L = [r"\begin{deluxetable}{lrrccc}", r"\tablecaption{Fraction of ZTF-detected periodic variables for which Gaia DR3 published a variability result, $C$ (\%), in the 280 random partitions. $N_{\rm f}$, $N_{\rm q}$: detections among Gaia-flagged and Gaia-quiet stars. $C_{\rm all}$: every detection; $C_{\rm sup}$: quiet detections counted only if catalogued within $5''$ or passing vetting; $C_{\rm SC}$: SCoPe-confirmed variables on both sides (headline). In brackets: 95\% interval from a bootstrap over partitions. Bins with fewer than 30 stars are left out.\label{tab:sf}}",
     r"\tablehead{\colhead{Bin} & \colhead{$N_{\rm f}$} & \colhead{$N_{\rm q}$} & \colhead{$C_{\rm all}$} & \colhead{$C_{\rm sup}$} & \colhead{$C_{\rm SC}$}}", r"\startdata"]
for var in edges:
    L.append(r"\multicolumn{6}{l}{\textit{" + lab[var] + r"}} \\")
    for r in B[B['var'] == var].itertuples():
        r = r._asdict()
        if r['F_all'] + r['Q_all'] < 30: continue
        L.append(f"{rng_(var, r['lo'], r['hi'])} & {r['F_all']:,} & {r['Q_all']:,} & {cell(r, 'all')} & {cell(r, 'sup')} & {cell(r, 'sc')} \\\\")
L += [r"\enddata", r"\end{deluxetable}"]; open('table_sn.tex', 'w').write("\n".join(L) + "\n")
L = [r"\begin{deluxetable}{lcccc}", r"\tablecaption{$C$ (\%) by period and amplitude, SCoPe-confirmed variables on both sides (all detections in brackets). Cells with fewer than 20 stars are left out.\label{tab:pa}}",
     r"\tablehead{\colhead{Period (d)} & \colhead{0.05--0.1 mag} & \colhead{0.1--0.2 mag} & \colhead{0.2--0.4 mag} & \colhead{$\ge0.4$ mag}}", r"\startdata"]
for (plo, phi), g in PA.groupby(['plo', 'phi']):
    cells = ['--' if pd.isna(r.C_sc) else f"{100 * r.C_sc:.0f} ({100 * r.C_all:.0f})" for r in g.itertuples()]
    L.append(f"{plo:g}--{phi:g} & " + " & ".join(cells) + r" \\")
L += [r"\enddata", r"\end{deluxetable}"]; open('table_pa.tex', 'w').write("\n".join(L) + "\n")
L = [r"\begin{deluxetable}{lrr}", r"\tablecaption{Detection rate of the frozen detector among Gaia-flagged stars that pass the pre-filter, by Gaia DR3 class (classes with at least 300 stars).\label{tab:class}}",
     r"\tablehead{\colhead{Gaia DR3 class} & \colhead{Stars} & \colhead{Detected (\%)}}", r"\startdata"]
for k, r in cls[cls.n >= 300].iterrows():
    L.append(f"{k.replace('_', ' ').replace('|', '/')} & {int(r.n):,} & {100 * r.recall:.1f} \\\\")
L += [r"\enddata", r"\end{deluxetable}"]; open('table_class.tex', 'w').write("\n".join(L) + "\n")
print(json.dumps({k: (v if not isinstance(v, tuple) else [round(x, 3) if isinstance(x, float) else x for x in v]) for k, v in out.items()}, indent=1, default=float))
print(B[['var', 'lo', 'hi', 'F_all', 'Q_all', 'C_all', 'C_sup', 'C_sc', 'lo_sc', 'hi_sc']].round(3).to_string())
print(PA.round(2).to_string()); print(cls.head(12).round(3).to_string())
