"""A6 injection test -> numbers_inj.tex (macros \\ni...). Efficiency = fraction of injections detected (det_clf)."""
import pandas as pd, numpy as np
I = pd.read_csv('data/injection_sf.csv'); M = {}
p = lambda x: f"{100 * x:.0f}"
M['N'] = f"{len(I):,}"
s = I[I.kind == 'sin']; e = I[I.kind == 'ecl']
s = s.assign(gb=(s.G * 2).astype(int))
def matched(d):
    """Detection efficiency of flagged and quiet injections, quiet reweighted to the flagged G distribution (0.5-mag bins)."""
    f = d[d.group == 'flagged']; q = d[d.group == 'quiet']
    w = f.gb.value_counts(normalize=True); eq = q.groupby('gb').det_clf.mean(); c = w.index.intersection(eq.index); w = w[c] / w[c].sum()
    return f.groupby('gb').det_clf.mean().reindex(c).mul(w).sum(), eq[c].mul(w).sum(), len(f), len(q)
a, b, nf, nq = matched(s[(s.A >= 0.01) & (s.A < 0.02)]); M['lowF'] = p(a); M['lowQ'] = p(b); M['lowNF'] = str(nf); M['lowNQ'] = str(nq)
a, b, _, _ = matched(s[s.A >= 0.02]); M['hiF'] = p(a); M['hiQ'] = p(b)
M['clf'] = p(s.det_clf.mean()); M['free'] = p(s.det_free.mean())
M['ecl'] = p(e.det_clf.mean())
with open('numbers_inj.tex', 'w') as fh:
    for k, v in sorted(M.items()): fh.write(f"\\newcommand{{\\ni{k}}}{{{v}}}\n")
print(M)
