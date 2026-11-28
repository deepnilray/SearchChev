"""Build dr4_xmatch.csv for score_dr4.py on Gaia DR4 release day (2 Dec 2026).

For every SearchChev candidate (prediction_candidates.csv, column source_id) and control star (control_list.csv,
column control_source_id) it finds the DR4 counterpart through gaiadr4.dr3_neighbourhood (columns, from the DR4 draft
data model section 7.3: source_id = DR4 id, dr3_source_id, angular_distance [mas], magnitude_difference,
proper_motion_propagation, probable_match), keeping probable_match = true and the smallest angular distance. Stars
with no neighbourhood row get a 1-arcsec position match (match = 'position_1arcsec'), as pre-registered.
It then reads phot_variable_flag and has_epoch_photometry from gaiadr4.gaia_source.

Usage:  python build_dr4_xmatch.py <TAP url> [out.csv]
TAP url examples: https://gea.esac.esa.int/tap-server/tap (ESA), https://gaia.aip.de/tap (AIP),
https://gaia.ari.uni-heidelberg.de/tap (ARI). Table and column names are checked first and the script stops with a
clear message if DR4 names differ from the draft data model, so the fix is a one-line edit.
"""
import sys, time
import numpy as np, pandas as pd
import pyvo
from astropy.table import Table

TAP = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "dr4_xmatch.csv"
svc = pyvo.dal.TAPService(TAP)
C = pd.read_csv("prediction_candidates.csv"); K = pd.read_csv("control_list.csv")
ids = pd.Series(pd.concat([C.source_id, K.control_source_id]).astype("int64").unique(), name="id")
print(len(ids), "DR3 source_ids to match")

def run(q, upload=None, tries=6):
    for k in range(tries):
        try:
            try:
                return svc.run_sync(q, uploads=upload, maxrec=10_000_000).to_table().to_pandas()
            except Exception:
                return svc.run_async(q, uploads=upload, maxrec=10_000_000).to_table().to_pandas()
        except Exception as ex:
            if k == tries - 1:
                raise
            print("retry:", repr(ex)[:200]); time.sleep(20 * (k + 1))

# 0. schema check against the draft data model
cols = run("SELECT TOP 1 source_id, dr3_source_id, angular_distance, magnitude_difference, probable_match "
           "FROM gaiadr4.dr3_neighbourhood")
cols = run("SELECT TOP 1 source_id, phot_variable_flag, has_epoch_photometry FROM gaiadr4.gaia_source")
print("schema OK")

# 1. neighbourhood match, in batches of 2,000 uploaded ids
rows = []
for i in range(0, len(ids), 2000):
    up = Table.from_pandas(ids.iloc[i:i + 2000].to_frame())
    rows.append(run("SELECT n.dr3_source_id, n.source_id AS dr4_source_id, n.angular_distance, n.magnitude_difference, "
                    "n.probable_match, g.phot_variable_flag, g.has_epoch_photometry "
                    "FROM gaiadr4.dr3_neighbourhood AS n JOIN TAP_UPLOAD.ids AS u ON u.id = n.dr3_source_id "
                    "JOIN gaiadr4.gaia_source AS g ON g.source_id = n.source_id", upload={"ids": up}))
N = pd.concat(rows, ignore_index=True)
N = N[N.probable_match.astype(bool)].sort_values("angular_distance").drop_duplicates("dr3_source_id")
N["match"] = "neighbourhood"
print(len(N), "matched through dr3_neighbourhood")

# 2. 1-arcsec position fallback for the rest (DR3 positions, J2016.0; DR4 positions propagated by pm to J2016.0)
miss = ids[~ids.isin(N.dr3_source_id)]
F = pd.DataFrame()
if len(miss):
    up = Table.from_pandas(miss.to_frame())
    p3 = run("SELECT g.source_id AS id, g.ra, g.dec FROM gaiadr3.gaia_source AS g JOIN TAP_UPLOAD.ids AS u ON u.id = g.source_id",
             upload={"ids": up})
    up2 = Table.from_pandas(p3)
    F = run("SELECT u.id AS dr3_source_id, g.source_id AS dr4_source_id, g.ra, g.dec, g.pmra, g.pmdec, "
            "g.phot_variable_flag, g.has_epoch_photometry "
            "FROM gaiadr4.gaia_source AS g JOIN TAP_UPLOAD.p AS u ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', u.ra, u.dec, 0.000556))",
            upload={"p": up2})
    # DR4 positions (J2017.5) propagated back 1.5 yr to J2016.0, as in the data-model example; separation in mas
    F = F.merge(p3.rename(columns={"id": "dr3_source_id", "ra": "ra3", "dec": "dec3"}), on="dr3_source_id")
    pmra, pmdec = F.pmra.fillna(0).astype(float), F.pmdec.fillna(0).astype(float)
    ra4 = F.ra - pmra * 1.5 / 3.6e6 / np.cos(np.radians(F.dec)); de4 = F.dec - pmdec * 1.5 / 3.6e6
    F["angular_distance"] = 3.6e6 * np.hypot((ra4 - F.ra3) * np.cos(np.radians(F.dec3)), de4 - F.dec3)
    F = F[F.angular_distance < 1000].sort_values("angular_distance").drop_duplicates("dr3_source_id"); F["match"] = "position_1arcsec"
    print(len(F), "matched by 1-arcsec position;", len(miss) - len(F), "unmatched")
X = pd.concat([N, F], ignore_index=True)[["dr3_source_id", "dr4_source_id", "match", "phot_variable_flag", "has_epoch_photometry", "angular_distance"]]
X.to_csv(OUT, index=False); print("wrote", OUT, len(X), "rows")
