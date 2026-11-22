import pandas as pd, os, sys, subprocess
h = pd.read_csv("runs/realsky1/search_hits.csv")
for t, g in h[h.status == "NEW_CANDIDATE"].groupby("tile"):
    d = f"runs/realsky1/vet_t{t}"
    if os.path.exists(f"{d}/vetting.csv"):
        continue
    os.makedirs(d, exist_ok=True); g.to_csv(f"{d}/in.csv", index=False)
    subprocess.run([sys.executable, "vet.py", f"{d}/in.csv", d])
