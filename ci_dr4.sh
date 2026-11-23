#!/usr/bin/env bash
# One shard of a dr4engine pass on a GitHub runner: process one HEALPix range at a time and publish after every range,
# so nothing is lost at the job time limit. Usage: ci_dr4.sh <shard> <nshards> <budget_min> <release> <phases>
set -u
K=$1; N=$2; BUDGET_MIN=$3; REL=$4; PHASES=$5
BR="dr4-results-$REL"
URL="https://x-access-token:${TOKEN}@github.com/${REPO}.git"
mkdir -p cache work
git clone -q --depth 1 --branch "$BR" "$URL" prev 2>/dev/null || mkdir -p prev/ranges
MODEL=prev/model/missed.joblib
END=$(( $(date +%s) + BUDGET_MIN * 60 ))
publish () {
  for try in 1 2 3 4 5 6; do
    rm -rf pub; git clone -q --depth 1 --branch "$BR" "$URL" pub 2>/dev/null || { mkdir -p pub && cd pub && git init -q && git checkout -q -b "$BR" && git remote add origin "$URL" && cd ..; }
    mkdir -p pub/ranges; cp -r work/ranges/* pub/ranges/ 2>/dev/null; [ -f work/SCHEMA_ALERT.txt ] && cp work/SCHEMA_ALERT.txt pub/
    cd pub; git add -A; git -c user.name=searchchev-bot -c user.email=bot@users.noreply.github.com commit -qm "dr4 $REL shard $K run ${GITHUB_RUN_ID:-local}" || { cd ..; return; }
    if git push -q origin HEAD:"$BR"; then cd ..; return; fi
    cd ..; sleep $(( RANDOM % 20 + 5 ))
  done
}
while [ $(date +%s) -lt $(( END - 50 * 60 )) ]; do
  before=$(ls work/ranges 2>/dev/null | wc -l)
  python dr4engine.py run --release "$REL" --cache cache --out work --shard $K --nshards $N --max-files 1 --hours 1 \
      --phases "$PHASES" --done-from prev/ranges --model "$MODEL"
  status=$?
  publish
  [ -f work/SCHEMA_ALERT.txt ] && { echo "unknown DR4 epoch-photometry schema: stopping"; exit 0; }
  after=$(ls work/ranges 2>/dev/null | wc -l)
  [ "$after" -gt "$before" ] || [ $status -ne 0 ] || break
done
