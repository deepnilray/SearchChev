"""Print how many search tiles are still undone across all published shard states (used to chain CI runs)."""
import json, glob, os, sys
import realsky as R
man = json.load(open("results/realsky1/model/manifest.json"))
sealed = {t["id"] for t in man["train"] + man["validation"]}
have = {t["id"] for t in man["search"]}
extra = R.extra_order_tiles("cache") + R.high_latitude_tiles("cache")
pool = {str(t["id"]) for t in man["search"]} | {str(t["id"]) for t in extra if t["id"] not in have and t["id"] not in sealed}
done = set(map(str, json.load(open("results/realsky1/model/search_state_seed.json"))["done"]))
for f in glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "prev", "shards/*/*/search_state.json")):
    done |= set(map(str, json.load(open(f))["done"]))
print(len(pool - done))
