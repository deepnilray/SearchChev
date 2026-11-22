import os as _os
OUT = _os.environ.get("OUT", "runs/realsky1")
import pandas as pd, os, sys, subprocess
if not os.path.exists(OUT + "/search_hits.csv"):
    sys.exit(0)
h = pd.read_csv(OUT + "/search_hits.csv")
for t, g in h[h.status == "NEW_CANDIDATE"].groupby("tile"):
    d = f"{OUT}/vet_t{t}"
    if os.path.exists(f"{d}/vetting.csv"):
        continue
    os.makedirs(d, exist_ok=True); g.to_csv(f"{d}/in.csv", index=False)
    subprocess.run([sys.executable, "vet.py", f"{d}/in.csv", d])
