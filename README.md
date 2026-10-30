# SearchChev

Created with SearchChev. Author: Deepnil Ray.

A terminal pipeline that searches for optical variable stars that sit in the VarWISE
infrared-variable catalogue (Paz et al. 2026, ApJS 284, 41) but in no optical variable catalogue.
It downloads its own real data (IRSA VarWISE, Gaia DR3, ZTF), trains and validates a model on it,
then searches the sky box by box.

## Run

```
pip install -r requirements.txt
python varwise_gaia_hunt.py selftest            # offline logic test
python varwise_gaia_hunt.py brutetest           # adversarial test, SIMULATED data only
python varwise_gaia_hunt.py smoke --out smoke   # every archive stage, PASS/FAIL per stage
python varwise_gaia_hunt.py auto --out run1     # preflight, download, train, validate, search (resumable)
```

`auto` stops without training if any preflight stage fails, and uses a model only if its
held-out AUC is at least 0.85 and a shuffled-label control is near 0.5.

## What it does

1. VarWISE sources in a sky box (the "pure" table is preferred).
2. Gaia DR3 counterpart within 3 arcsec; drops sources that already have a Gaia variability solution.
3. ZTF light curves, Lomb-Scargle period search, plus an eclipse (box-fit) channel for narrow dips.
4. 80/20 model split by sky cell (Gaia-known variables vs unflagged sources; proxy negatives are
   not confirmed constant stars).
5. Cuts: signal significance, non-alias period, independent agreement of both temporal halves.
6. Prior-art cut: VSX, ASAS-SN, ZTF periodic (Chen 2020), Gaia DR3 classifier, SIMBAD.
7. Comparison of the optical period with VarWISE's own infrared period (`varwise_period_agrees`).

## Status (be honest about it)

- MEASURED on simulated data only: the offline tests pass; eclipse channel gave 0 false passes in
  600 noise-only curves and recovered 62.5% of simulated narrow eclipses.
- NOT YET MEASURED on the real sky: every network stage, real-sky precision, completeness.
- Output files are CANDIDATES. Before calling any object new: inspect the folded light curve,
  check imaging for blends, test double the period, run MPChecker, search VSX by position.
- Known limits: single events are not found by the periodic channels; the anomaly queue
  (`--include-anomalies`) flags every kind of variability and is a lead generator only.

## EulerSearch #1: `eulerattack.py` (sealed, gated attack)

VarWISE Pure targets x ZTF DR24 light curves, trained on VSX confirmed variables (positives) and
Gaia-quiet stars (negatives, judged by a different instrument). Shape features only; magnitude-matched
classes; sky tiles fixed in a manifest before any download; a guard refuses any access outside the
current phase; validation tiles open only after the fit; the gate requires AUC lower bound >= 0.85,
shuffled control near 0.5, false-positive upper bound <= 5%, >= 150 per class. Failure stops; there is
no untrained mode.

```
python eulerattack.py brutetest            # offline adversarial test: 10/10 checks passed (synthetic sky)
python eulerattack.py census --out c1      # measures data adequacy on real tiles
python eulerattack.py train  --out a1
python eulerattack.py search --out a1 --hours 5 --target 20
```
Real-sky performance is not yet measured.
