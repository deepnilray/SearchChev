"""Merge the SCoPe lookups (branch scope-sf, one parquet per bundle) into one row per Gaia source:
scope_var = True if any SCoPe light curve of the star has vnv_xgb >= 0.8 or vnv_dnn >= 0.8 (the rule used for the
catalogue check). Writes scope_sf.parquet."""
import glob, sys
import pandas as pd
fs = sorted(glob.glob(sys.argv[1] + '/idx_*.parquet'))
assert len(fs) == 24, len(fs)
d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
d['hi'] = (d.vnv_xgb >= 0.8) | (d.vnv_dnn >= 0.8)
o = d.groupby('gaia_id').hi.any().rename('scope_var').reset_index()
o.to_parquet('scope_sf.parquet', index=False)
print(len(fs), 'bundles,', len(d), 'rows,', len(o), 'stars,', int(o.scope_var.sum()), 'variable')
