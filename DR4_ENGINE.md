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
