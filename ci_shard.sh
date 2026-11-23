#!/usr/bin/env bash
# One shard of the real-sky search on a GitHub runner: search one tile at a time, vet, re-check prior art,
# publish after every tile so nothing is lost at the job time limit.
set -u
export CACHE=cache
K=$1; N=$2; BUDGET_MIN=${3:-320}
OUT=work/shard$K; mkdir -p $OUT cache scratch
cp results/realsky1/model/{frozen.joblib,manifest.json,validation_report.json,val_features.parquet} $OUT/
cp results/realsky1/model/search_state_seed.json $OUT/search_state.json
# resume: mark every tile already finished by any earlier run as done, so repeated runs continue the sky
URL0="https://x-access-token:${TOKEN}@github.com/${REPO}.git"
if git clone -q --depth 1 --branch realsky-results "${URL_OVERRIDE:-$URL0}" prev 2>/dev/null; then
  python - "$OUT/search_state.json" <<'PY'
import json, glob, sys
p = sys.argv[1]; s = json.load(open(p)); done = set(map(str, s["done"]))
for f in glob.glob("prev/shards/*/*/search_state.json"):
    done |= set(map(str, json.load(open(f))["done"]))
s["done"] = [int(x) if x.isdigit() else x for x in sorted(done)]
json.dump(s, open(p, "w")); print("resume: tiles already done", len(done))
PY
fi
END=$(( $(date +%s) + BUDGET_MIN * 60 ))
publish () {
  D=pub/shards/run${GITHUB_RUN_ID:-local}/shard$K
  URL="${URL_OVERRIDE:-https://x-access-token:${TOKEN}@github.com/${REPO}.git}"
  for try in 1 2 3 4 5 6; do
    rm -rf pub; git clone -q --depth 1 --branch realsky-results "$URL" pub 2>/dev/null || { mkdir -p pub && cd pub && git init -q && git checkout -q -b realsky-results && git remote add origin "$URL" && cd ..; }
    mkdir -p $D
    cp $OUT/search_hits.csv $OUT/search_state.json $OUT/dossier.csv $D/ 2>/dev/null
    for v in $OUT/vet_t*; do [ -d "$v" ] && mkdir -p $D/$(basename $v) && cp $v/vetting.csv $v/*.png $v/*_lc.npz $D/$(basename $v)/ 2>/dev/null; done
    cd pub; git add -A; git -c user.name=searchchev-bot -c user.email=bot@users.noreply.github.com commit -qm "shard $K run ${GITHUB_RUN_ID:-local}" || { cd ..; return; }
    if git push -q origin HEAD:realsky-results; then cd ..; return; fi
    cd ..; sleep $(( RANDOM % 20 + 5 ))
  done
}
while [ $(date +%s) -lt $(( END - 45 * 60 )) ]; do
  before=$(python -c "import json;print(len(json.load(open('$OUT/search_state.json'))['done']))")
  python realsky.py search --out $OUT --cache cache --scratch scratch --shard $K --nshards $N --max-tiles 1 --hours 5 --workers 4 --extra-orders --short-channel --search-bmax 90 || break
  after=$(python -c "import json;print(len(json.load(open('$OUT/search_state.json'))['done']))")
  OUT=$OUT timeout 1200 python vet_pending.py
  OUT=$OUT timeout 1200 python dossier.py
  publish
  [ "$after" -gt "$before" ] || break
  rm -f cache/ztf_*.pkl scratch/*.pkl
done
