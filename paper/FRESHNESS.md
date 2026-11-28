# Freshness sweep (rule R1), 28 Nov 2026, 04:40-05:30 UTC

Purpose: find anything released on the same question, survey or objects that could change a SearchChev claim.

| # | Query / source | Result | Effect on claims |
|---|---|---|---|
| 1 | Web/arXiv: "Gaia DR3 variability completeness selection function missed variables ZTF 2026" | Forward model of the Gaia DR3 *source* selection function (arXiv:2610.32551); open-cluster selection function (A&A 706, A341); Gaia DR3 variable white dwarfs vetted by ZTF (A&A 2026). None measures the DR3 *variability* selection function against an independent survey. | Novelty statement stays, worded as "fraction against one independent detector as a function of period, amplitude and G". Cite Holl+2018 DR2 completeness estimates (80% for >=12 FoV transits, 51% for >=20) via Everall & Boubert 2022. |
| 2 | Web/arXiv: ZTF periodic-variable catalogues 2026 | ZVAR DR1 (Graham et al., arXiv:2611.16166, posted 12 Nov 2026, before our 27 Nov freeze); ultracool-dwarf ZTF variables (arXiv:2605.02216); Chen+2020 and SCoPe (already checked). | ZVAR checked 28 Nov (freeze/zvar_check_28nov.csv): periodograms recover our period for 24 of 32 new variables, 8 not. Paper must cite ZVAR as prior and state this. |
| 3 | Gaia DR4 content (cosmos.esa.int/web/gaia/dr4) and DR4 papers page | DR4 variability: 20 vari tables incl. vari_summary (403,061,676 processed sources per draft model), vari_rotation_modulation, vari_short_timescale; epoch photometry via DataLink; partner centres CDS, ASDC, ARI, AIP, Flatiron. Eyer+2023: DR3 general detection found ~400 M potentially variable sources but published results for 10.5 M. | Our C measures the fraction of variables for which DR3 *published* a variability result, not internal detection: say so explicitly. DR4 will publish results for far more stars, so P1 is expected to show a large rise; report DR4 rotation-class membership of our stars. |
| 4 | The 32 new variables by Gaia DR3 source_id (web search, three IDs) | No pages mention them. | None. |
| 5 | SNAD catalogue (snad.space/catalog/snad_catalog.csv, 374 objects incl. ZTF DR23 anomaly-search variables, arXiv:2507.06217) | Nearest SNAD object to any of the 32 is 22 arcmin away. | None. |
| 6 | All-VizieR cone, 5 arcsec, all 32 (28 Nov, idia mirror, 0 errors) | Only positional/photometric/spectroscopic surveys; one star (3212111109018082688) in young-star catalogues without variability or period. | None; mention the young-star membership. |
| 7 | VSX live, 30 arcsec, all 32 (28 Nov) | No entries. | None. |
| 8 | ZVAR DR1 cone search, 2 arcsec, all 32 (28 Nov, Deepnil's login) | 24 of 32 with a matching significant period (FAP < 1e-3, factor 0.5/1/2 within 1%). | Wording: 32 verified, not listed as variable in any catalogue; ZVAR periodograms recover 24; 8 recovered nowhere else. |

Still unchecked: Ofek et al. 2020 ZTF DR1 candidate list (not in VizieR; catsHTM format). Stated as a limitation.
