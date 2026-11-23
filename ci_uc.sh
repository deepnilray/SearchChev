#!/usr/bin/env bash
# One shard of the ultracompact white-dwarf search; publishes after every HATS partition.
set -u
K=$1; N=$2; BUDGET_MIN=$3; BR=uc-results
URL="https://x-access-token:${TOKEN}@github.com/${REPO}.git"
mkdir -p cache out
git clone -q --depth 1 --branch "$BR" "$URL" prev 2>/dev/null || mkdir -p prev
END=$(( $(date +%s) + BUDGET_MIN * 60 ))
publish () {
  for try in 1 2 3 4 5 6; do
    rm -rf pub; git clone -q --depth 1 --branch "$BR" "$URL" pub 2>/dev/null || { mkdir -p pub && cd pub && git init -q && git checkout -q -b "$BR" && git remote add origin "$URL" && cd ..; }
    cp out/uc_*.csv pub/ 2>/dev/null
    cd pub; git add -A; git -c user.name=searchchev-bot -c user.email=bot@users.noreply.github.com commit -qm "uc shard $K" || { cd ..; return; }
    if git push -q origin HEAD:"$BR"; then cd ..; return; fi
    cd ..; sleep $(( RANDOM % 20 + 5 ))
  done
}
python uc_wd.py control || { echo "positive control failed: not searching"; exit 1; }
while [ $(date +%s) -lt $(( END - 30 * 60 )) ]; do
  before=$(ls out | wc -l)
  python uc_wd.py run cache out --shard $K --nshards $N --hours 0.75 --done-from prev
  publish
  after=$(ls out | wc -l)
  [ "$after" -gt "$before" ] || break
done
