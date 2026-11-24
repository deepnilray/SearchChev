# DR4 engine: whole-archive search of Gaia epoch photometry

Built and tested on Gaia DR3 bulk files (November 2026), ready for Gaia DR4 on 2 December 2026.
Branch `dr4-engine`. Results go to branch `dr4-results-<release>`.

## What it searches for, in one pass over every light curve
| Search | Signal | Main impostor and how it is removed |
|---|---|---|
| `dips` | one long, deep dimming on a quiet star: occulting disks, ring systems, dust clouds | eclipsing binaries under sparse sampling: no normal points within 3 d of faint ones; ZTF must show one episode, not recurring eclipses |
| `impact` | dimming of 30 days or longer on a main-sequence star: giant-impact debris | dippers, R CrB stars, long-period variables: giants and very red stars excluded; NEOWISE must show an infrared brightening |
| `step` | a large lasting change, flat on both sides: one edge of a dimming longer than the baseline (Gaia17bpp-like) | trends of ordinary variables: both sides must be flat |
| `anomaly` | light curves unlike anything else in their range (IsolationForest) | quasars and galaxies removed; every candidate inspected |
| `missed` | variables Gaia DR4 itself did not flag | sealed train/validation ranges, 1% false-positive threshold, pre-registered gate |
| `uc_wd.py` | white-dwarf binaries with periods of 3-60 minutes in ZTF | same frequency required in g and r; integer cycles per day rejected |

## Tests already passed
* **Reader**: wide (DR3) and long (one row per transit) layouts give identical features (50 of 50 sources).
* **Injection on real Gaia sampling** (`inject_test.py`): dimmings of at least 0.2 mag sampled by three or more transits are
  recovered 67-100% of the time; steps of at least 0.3 mag 78-86%; pure noise 0 false alarms; eclipsing binaries reach
  the dips list 4.6% of the time and are then removed by the ZTF check.
* **Giant-impact signature**: `followup.ir_brightening` on ASASSN-21qj (Kenworthy et al. 2023) finds the published
  infrared afterglow, W1 brighter by 0.40 mag at 19 sigma with W1-W2 redder by 0.39 mag, peaking in late 2019.
* **ZTF check**: known eclipsing binaries are classed as recurrent, never as single dimmings.
* **Ultracompact positive control**: ZTF J1539+5027 recovered at 6.913 min (log FAP -50 in r, -36 in g).
* **Classifier path**: train, out-of-fold threshold, sealed validation, shuffled control and gate run end to end; a
  model failing the gate is never written.

## Running
```
# release day
Actions -> dr4-search -> Run workflow: branch dr4-engine, release gdr4, stage A
# white dwarfs (ZTF only, no DR4 needed)
Actions -> uc-white-dwarfs -> Run workflow: branch dr4-engine
```
Stage A processes the sealed training and validation ranges, trains and gates the model, then starts stage C on every
search range. Stage C ends by merging candidates into `catalogue/` and running ZTF, NEOWISE and prior-art checks on the
top 300 of each rare-event list. Every step publishes after each range, so nothing is lost at the 6-hour job limit,
and each run restarts the next until no range is left.

## Things only release day can settle
* DR4 file names and columns: `discover()` matches any `<Product>_<first>-<last>` file under `Gaia/gdr4/`, and the
  reader accepts both layouts and several column names. An unknown layout stops the run and writes `SCHEMA_ALERT.txt`
  with the header, so the fix is one edit.
* Whether DR4 epoch photometry is on the CDN on day one or only in the archive.

## Checked against the official draft DR4 data model (24 Nov 2026)

Source: ESA, *Gaia DR4 draft data model* (pre-release, 26 June 2026),
`anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/gaia-dr4-prerelease-draft-data-model_2026-06-26.zip`,
and the DR4 content page (cosmos.esa.int/web/gaia/dr4). What the documents say and what changed in the engine:

| DR4 fact (document section) | Effect on the engine | Fixed in code |
|---|---|---|
| `epoch_photometry` (11.1) is **one row per FoV transit** with scalar `g_obs_time, g_flux, g_flux_error, g_flux_over_error, g_mag, g_n_obs, bp_*, rp_*` and flag columns; 121.8 billion rows, 7.2 TB | The old reader treated `g_obs_time` + `g_mag` as DR3 array columns and would have yielded **no light curves, silently** | `_per_transit()` detects the layout; `_per_transit_curves()` groups rows by source across chunks |
| Bulk files are **Parquet** in the `DATAMODEL_GAIA` serialisation (8.4, 12.x text: "Parquet files available from the bulk download repository") | Parquet header and chunked reading were not supported | `epoch_header()` and `read_table(..., chunksize)` read Parquet schema and row batches |
| No per-transit rejection flag in `epoch_photometry`; VariPipe flags are a separate DataLink product `vari_epoch_flags` (18.7) for processed variables only | Quiet stars would have kept bad transits | Transits need `g_n_obs >= 5` of 9 AF CCDs (`G_MIN_CCD`) |
| `vari_summary` (18.18) has **403,061,676 rows**: every source the variability pipeline processed, not only variables (DR3: 10.5 M variables) | Using membership as "known variable" would have marked almost every star as known and emptied all searches | Membership used only for `gdr3` |
| `phot_variable_flag` now takes `VARIABLE`, `POSSIBLE`, `NON_VARIABLE`, or empty; `phot_variable_score` (GVD) is set for every processed source | POSSIBLE stars counted as quiet | `possible_variable` column; POSSIBLE is excluded from "missed" negatives and searched stars and ranked as known |
| `vari_classifier_result` (18.4) has one row **per classifier** with `classifier_best_class_name`, `classifier_best_class_prob` (DR3: `best_class_name`, one row) | Class lookup would have found nothing, so quasars and galaxies would pass the stellar filter | Both schemas read; most probable class over classifiers kept |
| `vari_unsupervised` (18.19, 101.8 M sources) holds autoencoder latent coordinates (Huijse et al. 2025), **no anomaly score**; it covers processed variables | The anomaly search is not pre-empted, but anyone can mine these latents | Anomaly candidates are to be compared with their latent-space neighbours in follow-up (pre-mortem item) |
| No DPAC table for long dimmings, steps or impact-like brightenings; tables exist for eclipsing binaries, LPVs, rotation, YSOs, flaring stars, short-timescale, microlensing (1,360), planetary transits (1,007) | `dips`, `step`, `impact` stay open; prior art must include every `vari_*` table | Prior-art step for DR4 candidates lists all `vari_*` tables |
| `source_id` is **not guaranteed** to be the same in DR3 and DR4; DPAC supplies `dr3_neighbourhood` (7.3) | The DR4 prediction test must not join on `source_id` | Scoring script (26 Nov) joins through `dr3_neighbourhood`, falling back to a 1" position match |
| HEALPix level 12 = `source_id // 34359738368` (unchanged) | `healpix8()` still valid | — |
| Archive closed 25 Nov 12:00 CET to 2 Dec 12:00 CET; DataLink: at most 5,000 sources per request, 5 concurrent jobs when logged in | A full-sky pass through DataLink is impossible in the time; the bulk repository is the only route | If no bulk ranges are found, the engine writes `SCHEMA_ALERT.txt` with the first 400 keys it sees and stops cleanly |

Not stated anywhere in the published documents, so still unknown until 2 December: the bulk-file names and
grouping (HEALPix ranges, HATS partitions, or other), the CDN path, and whether partner centres (CDS, ASDC, ARI,
AIP, Flatiron) carry the full epoch photometry on day one. The 2 December task handles each case:
ranges found → run; `SCHEMA_ALERT.txt` → adapt `discover()` to the real names and re-dispatch.

Tests run on 24 Nov: a synthetic file written exactly in the draft-model columns (300 sources, Parquet, 7,000-row
groups so sources straddle chunks) gives 300/300 light curves with the right point counts after the `g_n_obs`
cut; the same data written in the DR3 array layout gives identical G light curves; `source_table()` returns 10
VARIABLE, 10 POSSIBLE and the AGN class from a two-classifier DR4 table; `process_range()` runs end to end.
