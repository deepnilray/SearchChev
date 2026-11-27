# SearchChev: advance prediction for Gaia DR4

Fixed on 27 November 2026, before Gaia DR4 (release 2 December 2026). The files in this folder, with their
SHA-256 hashes in `SHA256SUMS`, are the complete prediction. Nothing here is changed after DR4 appears.

## What was searched

ZTF DR23 light curves of Gaia DR3 stars that DR3 left without any variability result
(`phot_variable_flag = NOT_AVAILABLE` and no `vari_*` entry, RUWE < 1.4, 13 <= G <= 18.5), in 2,796 HATS sky partitions (orders 4-6) at
|b| >= 15 deg (about 10,700 deg2). The detector, its threshold and its validation were frozen on 22 November
before any search tile was opened (sealed validation with a shuffled control). A star is a candidate when it is
grade A or B under the automatic rules, its period is recovered in at least two independent ZTF light curves and
it carries no warning flag.

## The two lists

* `prediction_candidates.csv`: all 5,778 frozen candidates (Gaia DR3 source_id, position, G, BP-RP, ZTF period in
  days, r-band amplitude, grade, detection channel).
* `control_list.csv`: for a seeded random 20% of the candidates (pandas `sample(frac=0.2, random_state=20261127)`),
  one Gaia DR3 star chosen at random (seed = the candidate's source_id) among stars within 0.5 deg with
  |dG| < 0.25 mag, |d(BP-RP)| < 0.15 mag, `phot_variable_flag = NOT_AVAILABLE`, RUWE < 1.4, and not a candidate.
  Each control row carries the candidate it is paired with and that candidate's ZTF period. Controls were
  obtained from the AIP Gaia mirror for 550 of the 1,156 sampled candidates before the mirror stopped answering on
  27 November (HTTP 429/504); which pairs succeeded depends on server availability, not on the stars. For 37 sampled
  candidates no star met the cuts; one of these was retried with widened cuts (|dG| < 0.4, |d(BP-RP)| < 0.3, 1 deg)
  and is marked `widened_cuts`. The control list has 550 stars.

## Predictions

* **P1.** Among DR4-matched stars, the fraction DR4 flags `VARIABLE` is higher for the candidates than for the
  controls, at z > 3.3 (two-proportion test, p < 0.001). The `POSSIBLE` and `NON_VARIABLE` fractions are reported
  separately for both lists.
* **P2.** For candidates with DR4 epoch photometry, the DR4 G-band Lomb-Scargle periodogram has a peak at the ZTF
  period (0.5x, 1x or 2x, within 1%) with false-alarm probability < 1e-3 in a larger fraction than for the
  controls tested at their paired candidate's period.
* **P3 (direction only).** Among the candidates, the DR4 `VARIABLE` fraction rises with ZTF amplitude, and falls
  for faint stars and for periods longer than about 10 days. It is measured in the pre-registered bins:
  log P in 8 bins over 0.05-500 d, amplitude in 5 bins, G in 1-mag bins from 12 to 19, with Wilson intervals.

No target number is set for any fraction. The result is reported as it comes out, including a failure of P1 or P2.

## Matching DR3 to DR4

Through the DR4 `dr3_neighbourhood` table, never by equal source_id. Where it has no entry, a 1 arcsec position
match is used and reported as such.

## Scoring

`python score_dr4.py dr4_xmatch.csv [dr4_lc_periods.csv]` writes `score.json` with every number above and its
95% Wilson interval. Input formats are described at the top of the script.
