#!/usr/bin/env bash
OUT=runs/realsky1_shard7_local
while true; do
  python realsky.py search --out $OUT --cache /home/claude/work/cache --scratch /home/claude/work/scratch --shard 7 --nshards 20 --max-tiles 1 --hours 5 --workers 2 || break
  OUT=$OUT python vet_pending.py; OUT=$OUT python dossier.py
done
