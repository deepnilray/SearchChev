# SearchChev v1: variable stars that Gaia DR3 did not flag, found in ZTF light curves

Deepnil Ray (ORCID 0000-0001-8779-0700). Frozen 27 November 2026, 03:45 UTC, before Gaia DR4 (2 December 2026).

## Method

1. **Targets.** Gaia DR3 stars with no variability result (`phot_variable_flag = NOT_AVAILABLE`, no `vari_*`
   entry) and RUWE < 1.4, in HATS order-5 sky tiles at Galactic latitude |b| >= 15 deg.
2. **Light curves.** ZTF DR23 g, r and i light curves, cross-matched to Gaia DR3.
3. **Detector.** A gradient-boosted classifier on light-curve shape features plus Lomb-Scargle and eclipse
   searches. Its threshold and training were frozen on 22 November 2026 before any search tile was
   opened; it was then validated once on sealed tiles with a shuffled-label control.
4. **Vetting rules (automatic, unchanged since the pilot).** Grade A or B only: the period must be recovered in
   at least two independent ZTF light curves (different fields, CCDs or filters), stay stable season by season,
   lie away from sampling aliases, and the star must have no brighter Gaia star within 6 arcsec. Grade C and any
   flagged star are never claimed.
5. **Prior art (rules fixed on 24 November, before the freeze).** A star is known if it appears in VSX, any of
   17 VizieR variable-star catalogues, SIMBAD with a variable type, ALeRCE with a periodic class at probability
   >= 0.8, or the latest ZTF SCoPe release (Healy et al. 2024, ApJS 272, 14; Zenodo 20057104) with variable
   probability >= 0.8 and either a class probability >= 0.8 or a matching period.
6. **Sky images.** Every star that survives prior art is inspected on Pan-STARRS1 or Legacy Survey images;
   galaxies, blends within 3 arcsec and brighter stars within 6 arcsec are dropped.

## Numbers at the freeze

| Quantity | Value |
|---|---|
| Sky tiles completed | 2,796 (about 9,400 deg2) |
| Gaia-DR3-quiet stars in those tiles | 40.4 million |
| Of these, with ZTF light curves and scored | 13.9 million |
| Grade A/B periodic variables (this catalogue) | 5,778 (A 3,210, B 2,568) |
| Known in VizieR, SIMBAD or ALeRCE | 529 |
| Checked against SCoPe so far | 788 |
| of which in SCoPe as variable with matching class or period | 688 (87.3%) |
| of which in SCoPe with a different period or class | 93 (11.8%) |
| of which absent from SCoPe | 7 (0.9%) |
| New variables that passed every check, including sky images | 3 |

The SCoPe check of the other 4,990 stars is in progress (the SCoPe release is 68 GB). Its results, and the new
variables among those stars, will be added as version 1.1 of this record before 2 December 2026. The search
itself continues on further sky tiles for version 2.

Ofek et al. (2020) ZTF DR1 periodic candidates were not checked (the catalogue could not be obtained); this is a
stated limitation.

## Files

* `catalogue_v1.csv`: the 5,778 frozen candidates with all prior-art columns.
* `new_variables_v1.csv`: the stars that passed every check, with light-curve plots in `lightcurves/`.
* `image_check.csv` and `image_check_v1_sheet.png`: the sky-image verdicts.
* `dr4_prediction/`: the advance prediction for Gaia DR4, its control list and scoring script.
* `code_snapshot.zip`: the code at the freeze commit.
* `SHA256SUMS`: hashes of every file.

Results branch: github.com/deepnilray/SearchChev, branch `realsky-results`, commit 4190e37.
