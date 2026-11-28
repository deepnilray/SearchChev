"""Vet the detections in order-4 and order-6 partitions that the search could not vet (paper checklist A4).

vet.py's partition reader always opened Norder=5 files, so the 920 search hits (at the 27 Nov freeze) that needed
vetting in the 101 order-4/6 partitions were never vetted. This runs the fixed vet.py (from dr4-engine) on them,
partition by partition, with the same arguments interface as the denominator workflow.
Since 28 Nov the input list also holds the 1,843 hits marked known only through Arévalo et al. (J/A+A/705/A247) entries
classed non-variable (column why=arevalo_nonvar; checklist A8). Original order-4/6 list: freeze/o46_only.csv.
Inputs fetched from dr4-engine (FETCH_HEAD in the workflow): vet.py, freeze/o46_tovet.csv.
"""
import argparse, json, os, subprocess, sys, time
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--cache", default="cache"); ap.add_argument("--scratch", default="scratch")
ap.add_argument("--model", default=""); ap.add_argument("--tiles", default="")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--hours", type=float, default=5.0); ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
sys.path.insert(0, os.getcwd())
for f in ("vet.py", "freeze/o46_tovet.csv"):
    if not os.path.exists(f"dr4_{os.path.basename(f)}"):
        open(f"dr4_{os.path.basename(f)}", "wb").write(subprocess.run(["git", "show", f"FETCH_HEAD:{f}"], capture_output=True, check=True).stdout)
os.replace("dr4_vet.py", "vet.py") if os.path.exists("dr4_vet.py") else None
import vet as V
from realsky import order5_tiles, extra_order_tiles
os.makedirs(a.out, exist_ok=True); os.makedirs(a.cache, exist_ok=True)
tiles = {str(t["id"]): t for t in order5_tiles(a.cache, 0, 90, -90) + extra_order_tiles(a.cache, 0, 90, -90)}
H = pd.read_csv("dr4_o46_tovet.csv", dtype={"tile": str})
want = sorted(H.tile.unique()); want = [w for i, w in enumerate(want) if i % a.nshards == a.shard]
st_p = os.path.join(a.out, "vet_state.json"); st = json.load(open(st_p)) if os.path.exists(st_p) else dict(done=[])
out_p = os.path.join(a.out, "vet_rows.csv"); t0 = time.time()
for tid in want:
    if tid in st["done"] or time.time() - t0 > a.hours * 3600:
        continue
    d = os.path.join(a.out, f"vet_t{tid}"); os.makedirs(d, exist_ok=True)
    try:
        r = V.vet(H[H.tile == tid], tiles, d)
    except Exception as ex:
        att = st.setdefault("attempts", {}); att[tid] = att.get(tid, 0) + 1
        if att[tid] >= 2:
            st["done"].append(tid); st.setdefault("failed", []).append(tid)
        json.dump(st, open(st_p, "w")); print(f"vet tile {tid}: error {ex!r}", flush=True); continue
    r = r.assign(tile=tid)
    r.to_csv(out_p, mode="a", header=not os.path.exists(out_p), index=False)
    st["done"].append(tid); json.dump(st, open(st_p, "w"))
    print(f"vet tile {tid}: {len(r)} vetted, grades {r.get('auto_grade', pd.Series(dtype=str)).value_counts().to_dict()} ({time.time() - t0:.0f}s)", flush=True)
