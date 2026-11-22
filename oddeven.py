import sys, numpy as np, pandas as pd
from vet import partition_rows, plot_candidate
from realsky import order5_tiles
tiles = {t["id"]: t for t in order5_tiles("/home/claude/work/cache", 0, 90, -90)}
for sid, ra, dec, P, tile in [(1006043282467183104, 94.18084066667916, 61.262807603892, 0.901217, 1787),
                              (1006101831461460224, 93.0919265740759, 61.33263496349847, 0.644249, 1787)]:
    rows = [r for r in partition_rows(tiles[tile], [ra], [dec])[0] if len(r["t"]) > 40 and r["band"] in "gr"]
    for b in "gr":
        rb = [r for r in rows if r["band"] == b]
        t = np.concatenate([r["t"] for r in rb]); m = np.concatenate([r["m"] - np.median(r["m"]) for r in rb])
        ph = ((t - 58000) / P) % 1; j0 = np.argmax(np.bincount((ph * 50).astype(int), weights=m, minlength=50) /
                                                  np.maximum(np.bincount((ph * 50).astype(int), minlength=50), 1))
        c = (j0 + 0.5) / 50
        cyc = np.floor((t - 58000) / P - c + 0.5)
        inecl = np.abs(((t - 58000) / P - c + 0.5) % 1 - 0.5) < 0.015
        sec = np.abs(((t - 58000) / P - c) % 1 - 0.5) < 0.015
        od, ev = m[inecl & (cyc % 2 == 1)], m[inecl & (cyc % 2 == 0)]
        print(sid, b, "P", P, "odd depth %.3f±%.3f (n=%d)  even %.3f±%.3f (n=%d)  at phase 0.5: %.3f±%.3f (n=%d)" % (
            np.median(od), 1.25*np.std(od)/np.sqrt(len(od)), len(od), np.median(ev), 1.25*np.std(ev)/np.sqrt(len(ev)), len(ev),
            np.median(m[sec]), 1.25*np.std(m[sec])/np.sqrt(sec.sum()), sec.sum()))
    plot_candidate(rows, (P, 2 * P), f"Gaia DR3 {sid}: P and 2P", f"runs/realsky1/oddeven_{sid}.png")
