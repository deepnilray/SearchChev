```
   _____                      __      ________
  / ___/___  ____ ___________/ /_    / ____/ /_  ___ _   __
  \__ \/ _ \/ __ `/ ___/ ___/ __ \  / /   / __ \/ _ \ | / /
 ___/ /  __/ /_/ / /  / /__/ / / / / /___/ / / /  __/ |/ /
/____/\___/\__,_/_/   \___/_/ /_/  \____/_/ /_/\___/|___/

        hunting the variable stars that Gaia DR3 called quiet
```

**Author:** Deepnil Ray ([ORCID 0000-0001-8779-0700](https://orcid.org/0000-0001-8779-0700)) · The NoRCEL Institute
**Catalogue v1:** [doi:10.5281/zenodo.24579691](https://doi.org/10.5281/zenodo.24579691) (frozen 27 Nov 2026, 03:45 UTC)

---

## What it does

Gaia DR3 published variability results for about 10.5 million of its 1.8 billion sources. Every other star is
either constant or a variable that Gaia's pipeline did not report. SearchChev reads Zwicky Transient Facility (ZTF)
DR23 light curves for Gaia DR3 stars with **no** variability result and finds the periodic ones, with a detector whose
threshold was fixed and whose performance was measured on sky tiles sealed before the search (AUC 0.954).

```
                 .        *          .             *           .
    *       .        GAIA DR3: "nothing to see here"       .        *
        .        *          .        *        .       *          .
   ------------------------------------------------------------------
                            ZTF, phase-folded:

      mag  |  ** *  *** ** *         ** *  *** ** *
           |         *    *  *               *    *  *
           |              *    *  *                *    *
           |                  * ** *                  * ** *
           +--------------------------------------------------> phase
                  0              0.5               1
```

## The funnel (catalogue v1)

```
  40.4 M   Gaia DR3 stars with no variability result
           13 <= G <= 18.5, |b| >= 15 deg, dec >= -25 deg, 10,700 deg^2
     |
     v
  13.9 M   with an isolated ZTF r-band light curve, scored by the frozen detector
     |
     v
  31,089   confirmed periodic signals
     |
     +--->  most already in a published variable-star catalogue (within 5")
     |
     v
   5,778   vetted catalogue stars, grade A or B
           (period in two separate light curves, across seasons, alias-free)
     |
     v
      32   NEW VARIABLE STARS
           listed as variable in none of VSX, VizieR, SIMBAD, ALeRCE, SCoPe, SNAD or ZVAR;
           image, contamination and light-curve checks passed
```

The 32 include three detached eclipsing binaries, one with a 25.4-day orbit. ZVAR's automated periodograms recover
24 of their periods; the other 8 were found by SearchChev alone.

## Two more results

* **How much of the variable sky Gaia DR3 published.** The same frozen detector is run on Gaia-flagged and
  Gaia-quiet stars in the same random sky partitions, so both groups share every ZTF selection effect. The ratio
  gives the fraction of ZTF-detectable periodic variables for which DR3 published a result, as a function of
  signal strength, period, amplitude and magnitude. (Paper in preparation.)
* **A prediction for Gaia DR4 (2 December 2026).** What DR4 will report for the catalogue stars and for matched
  control stars was deposited publicly before the release, and will be scored on release day.

## Branches

```
  main                 this page + the workflow definitions
  realsky-validation   the search code (realsky.py, vet.py, prior_art.py)
  realsky-results      search output, partition by partition
  dr4-engine           catalogue freeze, paper, selection function, DR4 prediction and scoring
```

## Cite

> Ray, D. 2026, SearchChev catalogue v1: periodic variable stars among Gaia DR3 stars with no variability result,
> Zenodo, doi:10.5281/zenodo.24579691

```
            *        .        *        .        *
       .        the sky is not as quiet as it looks        .
            *        .        *        .        *
```
