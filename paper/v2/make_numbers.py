"""Every number quoted in the paper, computed from the frozen data. Writes numbers.tex (LaTeX macros) and numbers.json.
Inputs: freeze/ and zenodo_v1/ (frozen catalogue), the freeze-commit search output (/home/claude/sf/hits_freeze.pkl,
search_state files), the denominator run (/home/claude/sf/F.pkl, Q.pkl) and the SCoPe lookup (scope_sf.parquet)."""
import json, glob, os
import numpy as np, pandas as pd
from scipy.stats import norm, wilcoxon

R = '/home/claude/searchchev'; SF = '/home/claude/sf'
N = {}
def put(k, v): N[k] = v
def fmt(x, d=0):
    return f"{x:,.{d}f}"

# ---- freeze -----------------------------------------------------------------------------------------------
meta = json.load(open(f'{R}/freeze/freeze_meta.json'))
done = set(); sc = sn = 0
for f in glob.glob(f'{SF}/fz/**/search_state.json', recursive=True):
    s = json.load(open(f)); done |= set(map(str, s['done'])); sc += s.get('scored', 0); sn += s.get('scanned', 0)
assert len(done) == meta['tiles_done'] and sc == meta['stars_scored'] and sn == meta['gaia_quiet_targets']
put('tiles', fmt(len(done))); put('area', fmt(round(meta['sky_deg2'], -2)))
put('quietM', f"{sn/1e6:.1f}"); put('scoredM', f"{sc/1e6:.1f}"); put('scored', fmt(sc)); put('quiet', fmt(sn))

dens = sn/meta['sky_deg2']; put('isodens', fmt(round(dens, -1))); put('isofrac', f"{100*(1-np.exp(-dens*np.pi*(3/3600)**2)):.1f}")
# ---- validation -------------------------------------------------------------------------------------------
v = json.load(open(f'{R}/results/realsky1/validation_report.json'))
put('nvalpos', fmt(v['n_val_pos'])); put('nvalneg', fmt(v['n_val_neg']))
put('auc', f"{v['auc']:.3f}"); put('auclo', f"{v['auc_lower95']:.3f}"); put('aucshuf', f"{v['shuffled_control']:.3f}")
put('rulefp', str(v['rule_fp'])); put('rulefpr', f"{100*v['rule_fpr']:.2f}"); put('rulefprhi', f"{100*v['rule_fpr_upper95']:.2f}")
put('rulerecall', f"{100*v['rule_recall']:.0f}"); put('ruletp', str(v['rule_tp']))
rc = v['recall_by_class']
for k, nm in [('ECL', 'Ecl'), ('RR', 'RR'), ('RS', 'RS'), ('LPV', 'LPV')]:
    put(f'rec{nm}', f"{100*rc[k]['rule_recall']:.0f}"); put(f'n{nm}', str(rc[k]['n']))

# ---- funnel at the freeze ---------------------------------------------------------------------------------
H = pd.read_pickle(f'{SF}/hits_freeze.pkl')
assert H.source_id.is_unique
main = H[H.channel != 'short']; short = H[H.channel == 'short']
gaia = {'Gaia_DR3_vari', 'GaiaDR3_rotation', 'GaiaDR3_EB'}; weak = {'GaiaDR2_LAV_Mowlavi2021', 'TESS_10k_unvetted', 'YSO_Delfini2025'}
def split(d):
    s = d.prior_art.fillna('').str.split(';').apply(lambda x: set(x) - {''})
    gnb = s.apply(lambda x: bool(x & gaia)); ext = s.apply(lambda x: len(x - gaia - weak) > 0) & ~gnb
    return gnb, ext
put('hits', fmt(len(H))); put('hitsmain', fmt(len(main))); put('hitsshort', fmt(len(short)))
kn = H[H.status == 'KNOWN']; gnb, ext = split(kn)
put('hitsknown', fmt(len(kn))); put('hitsknownext', fmt(int(ext.sum()))); put('hitsknowngaianb', fmt(int(gnb.sum())))
put('hitsknownother', fmt(int(len(kn) - ext.sum() - gnb.sum())))
tovet = H[H.status != 'KNOWN']; put('hitstovet', fmt(len(tovet)))
put('hitsunver', fmt(int((H.status == 'UNVERIFIED').sum())))
put('pass', fmt(sum(meta['grades'].values()))); put('gradeC', fmt(meta['grades']['C']))
c = pd.read_csv(f'{R}/zenodo_v1/catalogue_v1.csv', low_memory=False)
assert len(c) == meta['R2_candidates'] and c.source_id.isin(H.source_id).all()
put('nAB', fmt(len(c))); put('nA', fmt(int((c.auto_grade == 'A').sum()))); put('nB', fmt(int((c.auto_grade == 'B').sum())))
put('nABshort', str(int((c.channel == 'short').sum()))); put('nABecl', str(int((c.channel == 'eclipse').sum())))
put('vetpassfrac', f"{100*len(c)/len(tovet):.0f}")
# external catalogue counts among known-at-search hits (no Gaia neighbour)
from collections import Counter
cnt = Counter()
for x in kn[ext].prior_art.fillna('').str.split(';'):
    cnt.update(set(x) - gaia - weak - {''})
put('extVSX', fmt(cnt['VSX'])); put('extATLAS', fmt(cnt['ATLAS_Heinze2018'])); put('extChen', fmt(cnt['ZTF_Chen2020']))
put('extASASSN', fmt(cnt['ASASSN']))
# all Gaia-quiet periodic variables found (main channels + short channel), known-at-search with external match + vetted A/B
put('missedlo', fmt(int(ext.sum() + len(c))))

# ---- SCoPe on the catalogue --------------------------------------------------------------------------------
ss = c.scope_status.value_counts()
inside = len(c) - ss.get('NO_SCOPE_FIELD', 0)
put('scopein', fmt(inside)); put('scopeknown', fmt(ss['KNOWN'])); put('scopeknownpct', f"{100*ss['KNOWN']/inside:.1f}")
below = ss['SCOPE_CANDIDATE'] + ss['KNOWN_PERIOD_MATCH_LOWVNV'] + ss['PRESENT_LOW']
put('scopebelow', fmt(below)); put('scopebelowpct', f"{100*below/inside:.1f}")
put('scopeabsent', fmt(ss['ABSENT'])); put('scopeabsentpct', f"{100*ss['ABSENT']/inside:.1f}"); put('scopenofield', fmt(ss['NO_SCOPE_FIELD']))
known_any = (c.scope_status == 'KNOWN') | c.known_nonscope.fillna(False).astype(bool)
put('knownany', fmt(int(known_any.sum())))
img = pd.read_csv(f'{R}/freeze/image_check.csv'); nv = pd.read_csv(f'{R}/zenodo_v1/new_variables_v1.csv')
lowamp = pd.read_csv(f'{R}/zenodo_v1/low_amplitude_candidates_v1.csv')
put('nocat', str(len(img))); put('drop', str(int((img.image_verdict == 'DROP').sum())))
put('uncertain', str(int((img.image_verdict == 'UNCERTAIN').sum()))); put('weak', str(len(lowamp))); put('new', str(len(nv)))
assert len(img) == (img.image_verdict == 'DROP').sum() + (img.image_verdict == 'UNCERTAIN').sum() + len(lowamp) + len(nv)
put('notknownnoimg', fmt(int((~known_any).sum() - len(img))))
put('Pmed', f"{c.P.median():.1f}"); put('ampmed', f"{c.amp.median():.2f}")
put('Pmin', f"{c.P.min():.3f}"); put('Pminmain', f"{c[c.channel != 'short'].P.min():.2f}"); put('Pmax', f"{c.P.max():.0f}")

# ---- population ----------------------------------------------------------------------------------------------
g = pd.read_csv('gaia_params.csv'); cg = c.merge(g[['source_id', 'parallax', 'parallax_over_error', 'phot_g_n_obs']], on='source_id', how='left')
m = cg.parallax_over_error > 5; MG = cg.G + 5*np.log10(cg.parallax/100.)
put('ncmd', fmt(int(m.sum()))); put('redfrac', f"{100*(cg.bp_rp[m] > 1.8).mean():.0f}"); put('MGmed', f"{MG[m].median():.1f}")
put('bprpunk', f"{cg.bp_rp[m & ~known_any].median():.2f}"); put('bprpknown', f"{cg.bp_rp[m & known_any].median():.2f}")
put('ampunk', f"{cg.amp[~known_any].median():.3f}"); put('ampknown', f"{cg.amp[known_any].median():.3f}")
put('Punk', f"{cg.P[~known_any].median():.1f}"); put('Pknown', f"{cg.P[known_any].median():.1f}")
k = pd.read_csv(f'{R}/dr4_prediction/control_list.csv')
k = k.merge(g[['source_id', 'phot_g_n_obs']].rename(columns={'source_id': 'control_source_id', 'phot_g_n_obs': 'nc'}), on='control_source_id') \
     .merge(g[['source_id', 'phot_g_n_obs']].rename(columns={'source_id': 'paired_candidate', 'phot_g_n_obs': 'nk'}), on='paired_candidate')
put('npairs', str(len(k))); put('nobsdiff', f"{np.median(k.nk - k.nc):.0f}"); put('nobsp', f"{wilcoxon(k.nk, k.nc).pvalue:.2f}")
put('ncontrol', str(len(pd.read_csv(f'{R}/dr4_prediction/control_list.csv'))))

# ---- selection function (denominator) ------------------------------------------------------------------------
F = pd.read_pickle(f'{SF}/F.pkl'); Q = pd.read_pickle(f'{SF}/Q.pkl')
tiles = sorted(set(str(t) for t in json.load(open(f'{R}/freeze/denominator_tiles.json'))))
st = [json.load(open(f)) for f in glob.glob(f'{SF}/shard*/denom_state.json')]
cn = {k: v for s in st for k, v in s.get('counts', {}).items()}
assert len(cn) == len(tiles) == 280
put('dtiles', str(len(tiles))); put('dflag', fmt(sum(x['flagged'] for x in cn.values())))
put('dmatch', fmt(sum(x['ztf_matched'] for x in cn.values()))); put('dfeat', fmt(sum(x['prefilter_kept'] for x in cn.values())))
put('dF', fmt(len(F))); put('dQ', fmt(len(Q))); put('dQext', fmt(int(Q.ext.sum()))); put('dQvet', fmt(int(Q.vet.sum())))
put('dQconf', fmt(int(Q.conf.sum()))); put('dQgnb', fmt(int(Q.gaia_nb.sum())))
put('dFfrac', f"{100*len(F)/sum(x['prefilter_kept'] for x in cn.values()):.1f}")
def comp(f, q): return f / (f + q)
put('Craw', f"{100*comp(len(F), len(Q)):.0f}"); put('Cconf', f"{100*comp(len(F), Q.conf.sum()):.0f}")
put('Mraw', f"{100-100*comp(len(F), len(Q)):.0f}"); put('Mconf', f"{100-100*comp(len(F), Q.conf.sum()):.0f}")
rng = np.random.default_rng(20261127)
fc = F.groupby('tile').size().reindex(tiles, fill_value=0).values
qr = Q.groupby('tile').size().reindex(tiles, fill_value=0).values
qc = Q[Q.conf].groupby('tile').size().reindex(tiles, fill_value=0).values
SC = None
if os.path.exists('scope_sf.parquet'):
    SC = pd.read_parquet('scope_sf.parquet')      # one row per Gaia id: scope_var (bool)
    vs = set(SC[SC.scope_var].gaia_id); seen = set(SC.gaia_id)
    F['sc'] = F.source_id.isin(vs); Q['sc'] = Q.source_id.isin(vs); F['sin'] = F.source_id.isin(seen); Q['sin'] = Q.source_id.isin(seen)
    fs = F.groupby('tile').sc.sum().reindex(tiles, fill_value=0).values; qs = Q.groupby('tile').sc.sum().reindex(tiles, fill_value=0).values
    put('scFin', fmt(int(F.sin.sum()))); put('scQin', fmt(int(Q.sin.sum())))
    put('scF', fmt(int(F.sc.sum()))); put('scQ', fmt(int(Q.sc.sum())))
    put('scFpct', f"{100*F.sc.sum()/F.sin.sum():.0f}"); put('scQpct', f"{100*Q.sc.sum()/Q.sin.sum():.0f}")
    put('Csc', f"{100*comp(F.sc.sum(), Q.sc.sum()):.0f}"); put('Msc', f"{100-100*comp(F.sc.sum(), Q.sc.sum()):.0f}")
B = []
for _ in range(4000):
    i = rng.integers(0, len(tiles), len(tiles)); row = [comp(fc[i].sum(), qr[i].sum()), comp(fc[i].sum(), qc[i].sum())]
    if SC is not None: row.append(comp(fs[i].sum(), qs[i].sum()))
    B.append(row)
B = 100*np.array(B)
for j, nm in enumerate(['raw', 'conf', 'sc'][:B.shape[1]]):
    lo, hi = np.percentile(B[:, j], [2.5, 97.5]); put(f'C{nm}lo', f"{lo:.0f}"); put(f'C{nm}hi', f"{hi:.0f}")

def wilson(k, n, z=norm.ppf(0.975)):
    if n == 0: return np.nan, np.nan
    p = k/n; d = 1 + z*z/n; cc = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d; return cc - h, cc + h
edges = {'period': np.logspace(np.log10(0.05), np.log10(500), 9), 'amp': np.array([0, 0.05, 0.1, 0.2, 0.4, 10.]), 'G': np.arange(13, 19.5, 1.0)}
rows = []
for var, e in edges.items():
    for lo, hi in zip(e[:-1], e[1:]):
        sel = lambda d: d[(d[var].astype(float) >= lo) & (d[var].astype(float) < hi)]
        f, q = sel(F), sel(Q); qcf = q[q.conf]
        r = dict(var=var, lo=lo, hi=hi, F=len(f), Q=len(q), Qconf=len(qcf))
        if len(f) + len(q) == 0: continue
        r['C_raw'] = comp(len(f), len(q)); r['C_raw_lo'], r['C_raw_hi'] = wilson(len(f), len(f)+len(q))
        r['C_conf'] = comp(len(f), len(qcf)); r['C_conf_lo'], r['C_conf_hi'] = wilson(len(f), len(f)+len(qcf))
        if SC is not None:
            a, b = int(f.sc.sum()), int(q.sc.sum()); r.update(Fsc=a, Qsc=b)
            if a + b: r['C_sc'] = comp(a, b); r['C_sc_lo'], r['C_sc_hi'] = wilson(a, a+b)
        rows.append(r)
T = pd.DataFrame(rows); T.to_csv('selection_function_bins.csv', index=False)
def binval(var, lo, col): return T[(T['var'] == var) & np.isclose(T.lo, lo)][col].iloc[0]
pe = edges['period']
put('CshortP', f"{100*binval('period', pe[0], 'C_raw'):.0f}--{100*binval('period', pe[1], 'C_conf'):.0f}")
long = T[(T['var'] == 'period') & (T.lo >= pe[3] - 1e-9) & (T.hi <= pe[7] + 1e-9)]
put('ClongPlo', f"{100*long.C_raw.min():.0f}"); put('ClongPhi', f"{100*long.C_conf.max():.0f}")
ae = edges['amp']
put('Clowamp', f"{100*binval('amp', ae[1], 'C_raw'):.0f}--{100*binval('amp', ae[1], 'C_conf'):.0f}")
put('Chighamp', f"{100*binval('amp', ae[4], 'C_raw'):.0f}--{100*binval('amp', ae[4], 'C_conf'):.0f}")
fp = F.period.astype(float); fs_ = F[fp < 0.5]; fl = F[(fp >= pe[3]) & (fp < pe[7])]
put('FshortN', fmt(len(fs_))); put('FshortECL', fmt(int((fs_.gaia_class == 'ECL').sum()))); put('FshortRR', fmt(int((fs_.gaia_class == 'RR').sum())))
put('FlongN', fmt(len(fl))); put('FlongRS', fmt(int((fl.gaia_class == 'RS').sum()))); put('FlongSOL', fmt(int((fl.gaia_class == 'SOLAR_LIKE').sum())))
gb = lambda lo, col: binval('G', lo, col)
put('GdropRaw', f"{100*(max(gb(13,'C_raw'),gb(14,'C_raw')) - min(gb(16,'C_raw'),gb(17,'C_raw'))):.0f}")
put('GdropConf', f"{100*(max(gb(13,'C_conf'),gb(14,'C_conf')) - min(gb(16,'C_conf'),gb(17,'C_conf'))):.0f}")
# LaTeX table of the bins
lab = {'period': 'P (d)', 'amp': 'Amp. (mag)', 'G': '$G$'}
L = [r'\begin{deluxetable}{lrrcc' + ('c' if SC is not None else '') + '}',
     r'\tablecaption{Fraction of ZTF-detected periodic variables flagged by Gaia DR3, in the registered bins. $N_{\rm f}$, $N_{\rm q}$: detections among Gaia-flagged and Gaia-quiet stars. $C_{\rm all}$, $C_{\rm sup}$: flagged fraction (\%) counting every quiet detection, or supported ones only' + (r'; $C_{\rm SC}$: SCoPe-confirmed on both sides' if SC is not None else '') + r'. Bins with fewer than 30 stars are omitted.\label{tab:sf}}',
     r'\tablehead{\colhead{Bin} & \colhead{$N_{\rm f}$} & \colhead{$N_{\rm q}$} & \colhead{$C_{\rm all}$} & \colhead{$C_{\rm sup}$}' + (r' & \colhead{$C_{\rm SC}$}' if SC is not None else '') + '}', r'\startdata']
def rng_(lo, hi, var):
    if var == 'amp' and hi > 5: return f'$\\ge{lo:g}$'
    if var == 'period': return f'{lo:.2f}--{hi:.2f}' if hi < 1 else (f'{lo:.1f}--{hi:.1f}' if hi < 10 else f'{lo:.0f}--{hi:.0f}')
    return f'{lo:g}--{hi:g}'
cur = None
for r in T.itertuples():
    if r.F + r.Q < 30: continue
    if r.var != cur:
        L.append(r'\multicolumn{' + ('6' if SC is not None else '5') + r'}{l}{\textit{' + {'period': 'Period (d)', 'amp': 'Amplitude (mag)', 'G': '$G$ (mag)'}[r.var] + r'}} \\'); cur = r.var
    cells = [rng_(r.lo, r.hi, r.var), f'{r.F:,}', f'{r.Q:,}', f'{100*r.C_raw:.0f} ({100*r.C_raw_lo:.0f}--{100*r.C_raw_hi:.0f})', f'{100*r.C_conf:.0f} ({100*r.C_conf_lo:.0f}--{100*r.C_conf_hi:.0f})']
    if SC is not None:
        cells.append(f'{100*r.C_sc:.0f} ({100*r.C_sc_lo:.0f}--{100*r.C_sc_hi:.0f})' if pd.notna(getattr(r, 'C_sc', np.nan)) else '--')
    L.append(' & '.join(cells) + r' \\')
L += [r'\enddata', r'\end{deluxetable}']
open('sf_table.tex', 'w').write('\n'.join(L) + '\n')
F.to_pickle('F_sc.pkl'); Q.to_pickle('Q_sc.pkl')

with open('numbers.tex', 'w') as fh:
    for k2, v2 in sorted(N.items()):
        assert k2.isalpha(), k2
        fh.write(f"\\newcommand{{\\n{k2}}}{{{v2}}}\n")
json.dump(N, open('numbers.json', 'w'), indent=1)
print(json.dumps(N, indent=0))
print(T.round(3).to_string())
