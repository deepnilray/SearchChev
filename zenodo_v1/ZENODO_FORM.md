# Zenodo upload form, field by field (SearchChev v1)

Go to zenodo.org, log in with your ORCID, click **New upload**, then:

1. **Files:** drag in `searchchev_v1_zenodo.zip` (one file; Zenodo keeps it as is).
2. **Resource type:** Dataset.
3. **Title:** SearchChev v1: periodic variable stars missed by Gaia DR3, found in ZTF light curves, with an advance prediction for Gaia DR4
4. **Publication date:** 2026-11-27
5. **Creators:** Ray, Deepnil. ORCID 0000-0001-8779-0700. Affiliation: Ashoka University (add The NoRCEL Institute if you want it shown).
6. **Description** (paste):

   A search of Zwicky Transient Facility DR23 light curves for periodic variability in Gaia DR3 stars that DR3 left
   without any variability result. In 2,796 sky tiles (about 9,400 square degrees at Galactic latitude above 15
   degrees), 13.9 million such stars were scored by a detector whose threshold was frozen and validated on sealed
   tiles before the search. 5,778 stars pass every vetting rule (period recovered in at least two independent ZTF
   light curves, stable season to season, no brighter neighbour). The record contains the frozen catalogue with
   prior-art checks against VSX, 17 VizieR catalogues, SIMBAD, ALeRCE and the ZTF SCoPe classifications, the new
   variables that pass every check including sky images, and an advance prediction of what Gaia DR4 (2 December
   2026) will report for these stars, with a matched control list and the scoring script. The catalogue was frozen
   on 27 November 2026 at 03:45 UTC. METHOD.md describes the method and the numbers; SHA256SUMS lists file hashes.

7. **License:** Creative Commons Attribution 4.0 International.
8. **Keywords:** variable stars; Gaia DR3; Gaia DR4; ZTF; selection function; time-domain astronomy
9. **Version:** v1
10. **Related works:**
    * "Is supplemented by" → https://github.com/deepnilray/SearchChev (Software)
    * "References" → 10.5281/zenodo.20057104 (Dataset; ZTF SCoPe classifications)
    * "References" → 10.3847/1538-4365/ad33c6 (Journal article; Healy et al. 2024)
11. Leave access **Open**. Click **Publish**.

After publishing, send me the DOI (it looks like 10.5281/zenodo.NNNNNNNN). I add it to the RNAAS note and the plan.
