"""inject_test.py - completeness and false-alarm rate of the dips and impact searches on real Gaia sampling.

Takes real Gaia transit times and errors from one epoch-photometry file, builds a constant star at the source's own
magnitude scatter, and injects
  (a) single dimming episodes (duration 1-300 d, depth 0.05-1.5 mag, smooth trapezoid), the signal the search wants,
  (b) detached eclipsing binaries (P 0.3-30 d, eclipse width 3-8% of P, depth 0.1-1 mag), the main impostor,
  (c) nothing (pure noise),
then runs the same features and rules used on the sky. Rates are what a paper quotes for completeness and purity.

usage: python inject_test.py <EpochPhotometry file> [n_per_kind]
"""
import sys
import numpy as np, pandas as pd
import dr4engine as E


def trapezoid(t, t0, dur, depth, ramp=0.2):
    x = np.abs(t - t0) / (dur / 2)
    return depth * np.clip((1 - x) / ramp, 0, 1)


def eclipses(t, P, w, d1, d2, phase0):
    ph = ((t / P) + phase0) % 1
    a = np.clip(1 - np.abs(ph - 0.0) / (w / 2), 0, 1) + np.clip(1 - np.abs(ph - 1.0) / (w / 2), 0, 1)
    b = np.clip(1 - np.abs(ph - 0.5) / (w / 2), 0, 1)
    return d1 * a + d2 * b


def star_row(med, plx=5.0):
    return dict(parallax=plx, parallax_error=0.1, pmra=10.0, pmdec=0.0, gaia_class="", phot_g_mean_mag=med, bp_rp=1.0,
                flagged_variable=False)


def run(path, n=400, seed=1):
    rng = np.random.default_rng(seed)
    cad = []
    for sid, lc in E.light_curves(path):
        t, m, e = lc["G"]
        if t.size >= 25:
            cad.append((t, e, np.median(m)))
        if len(cad) >= 3000:
            break
    rows = []
    for kind in ("dip", "step", "eb", "noise"):
        for i in range(n):
            t, e, med = cad[rng.integers(len(cad))]
            noise = rng.normal(0, 1, t.size) * np.sqrt(e ** 2 + 0.003 ** 2)
            truth = dict(kind=kind)
            if kind == "dip":
                dur = float(np.exp(rng.uniform(np.log(1), np.log(300)))); depth = float(np.exp(rng.uniform(np.log(0.05), np.log(1.5))))
                t0 = rng.uniform(t.min() + dur / 2, max(t.min() + dur / 2 + 1, t.max() - dur / 2))
                sig = trapezoid(t, t0, dur, depth); truth.update(dur=dur, depth=depth)
                truth["n_in_dip"] = int((sig > 0.5 * depth).sum())
            elif kind == "step":
                size = float(np.exp(rng.uniform(np.log(0.1), np.log(4.0)))) * rng.choice([-1, 1])
                ts = rng.uniform(np.percentile(t, 15), np.percentile(t, 85))
                ramp = float(np.exp(rng.uniform(np.log(1), np.log(200))))
                sig = size * np.clip((t - ts) / ramp + 0.5, 0, 1); truth.update(depth=abs(size), dur=ramp)
            elif kind == "eb":
                P = float(np.exp(rng.uniform(np.log(0.3), np.log(30)))); w = rng.uniform(0.03, 0.08)
                d1 = float(np.exp(rng.uniform(np.log(0.1), np.log(1.0)))); d2 = d1 * rng.uniform(0, 1)
                sig = eclipses(t, P, w, d1, d2, rng.uniform()); truth.update(P=P, depth=d1)
            else:
                sig = 0 * t
            m = med + noise + sig
            f = E.features({"G": (t, m, e)}); f.update(star_row(med)); f.update(truth); f["source_id"] = len(rows)
            rows.append(f)
    F = pd.DataFrame(rows)
    hit_d = set(E.search_dips(F).source_id); hit_i = set(E.search_impact(F).source_id)
    hit_s = set(E.search_step(F).source_id)
    F["in_dips"] = F.source_id.isin(hit_d); F["in_impact"] = F.source_id.isin(hit_i); F["in_step"] = F.source_id.isin(hit_s)
    out = {}
    for kind, g in F.groupby("kind"):
        out[kind] = dict(n=len(g), dips=round(g.in_dips.mean(), 4), impact=round(g.in_impact.mean(), 4),
                         step=round(g.in_step.mean(), 4))
    d = F[F.kind == "dip"].copy()
    d["dur_bin"] = pd.cut(d.dur, [1, 3, 10, 30, 100, 300]); d["depth_bin"] = pd.cut(d.depth, [0.05, 0.1, 0.2, 0.5, 1.5])
    comp = d[d.n_in_dip >= 3].pivot_table(index="dur_bin", columns="depth_bin", values="in_dips", aggfunc="mean", observed=False)
    st = F[F.kind == "step"].copy(); st["size_bin"] = pd.cut(st.depth, [0.1, 0.3, 1, 4])
    out["step_completeness_by_size"] = st.groupby("size_bin", observed=False).in_step.mean().round(2).to_dict()
    return out, comp, F


if __name__ == "__main__":
    out, comp, F = run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
    print("selection rates by injected kind:", out)
    print("dips completeness (episodes sampled by >= 3 transits), rows duration [d], columns depth [mag]:")
    print(comp.round(2).to_string())
    F.to_csv("inject_results.csv", index=False)
