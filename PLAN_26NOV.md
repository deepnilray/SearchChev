# SearchChev: locked plan for Thursday 26 November 2026 (IST)

Written Tuesday 24 November, before the data are final, so the decision rules cannot be bent to the results.
The scheduled sessions on 26 November follow this file exactly. Anything it does not cover is reported to Deepnil
before it is decided.

## Goal of the day
Freeze a catalogue of new variable stars that survives every check below, publish it with a date before Gaia DR4
(2 December), and fix the advance DR4 prediction. Nothing leaves us that has not passed every step.

## Fixed rules (decided now)

**R1. When the catalogue freezes.** At the first of: (a) the search has no regions left, or (b) 12:00 IST on 26 Nov.
If (b), the catalogue covers the regions done by then, the paper states the sky fraction, and the search keeps
running for version 2. No waiting past 12:00.

**R2. Who counts as a candidate.** Grade A or B from the automatic rules (unchanged since the pilot), period
recovered in at least two independent ZTF light curves, no flag. Grade C is never claimed.

**R3. Prior art.** A star is **known**, and dropped, if any of these holds:
1. VSX (live API) has an entry within 5", or within 30" with the same period (x0.5, x1, x2, within 1%).
2. Any of the 17 VizieR catalogues of the live run has it within 5" with a period or type (the three flag-only
   lists, Mowlavi 2021, unvetted TESS 10k and Delfini 2025, make it "new first period", kept).
3. **SCoPe, latest version** (Healy et al., Zenodo 20057104: 1.65 billion light curves in 636 ZTF fields, i.e.
   almost the whole ZTF primary grid, so it applies to nearly every candidate): variable probability (vnv) >= 0.8
   in either model AND either a class probability >= 0.8 or a period matching ours (x0.5, x1, x2, within 1%) in any
   filter. A SCoPe row with vnv 0.5-0.8 and no matching period is kept but flagged "SCoPe candidate" and counted
   separately. Stars SCoPe already lists this way are reported in their own table as "independently recovered,
   first vetted period and type", never as discoveries.
4. **ALeRCE** light-curve classifier: a periodic class with probability >= 0.8 for the same ZTF object.
5. **Ofek et al. 2020** (ZTF DR1, 10 million candidates): a match within 2" with a period matching ours. If the
   catalogue cannot be downloaded by 12:00, this check is listed as a limitation in the paper, not skipped silently.
6. SIMBAD within 10" with any variable-star type.
A check that errors is retried; a star whose check still fails is "unverified" and not claimed.

**R4. Sky images.** Every remaining star is inspected on Legacy Survey DR10 or Pan-STARRS cutouts. Dropped: a
galaxy, a visible blend within 3", or a brighter star within 6". Kept with a flag: a fainter neighbour 3-6".

**R5. The claimed catalogue** is R2 + R3 + R4 survivors. Its size is reported as it is; no target number.

## Order of work on 26 November
| IST | Step | Output |
|---|---|---|
| 09:30 | Check the search. If finished or at 12:00 (R1): merge every shard into one list | `catalogue_v1_raw.csv` |
| at search end | Start the SCoPe index on GitHub (`scope.yml`, 24 bundle jobs, 69 GB, about 1-2 h); it waits for the search so it never slows it | branch `scope-index` |
| 09:30-12:00 | In the workspace: rerun the 17 VizieR catalogues, live VSX, SIMBAD, ALeRCE, Ofek 2020 | `prior_art_full.csv` |
| SCoPe done | Apply SCoPe (R3.3) | overlap numbers by vnv, class and filter |
| 12:00-13:30 | Sky-image sheets for all survivors; inspect every one (R4) | `image_check.csv` |
| 13:30 | Freeze `catalogue_v1.csv` (R5), with SHA-256 hash | frozen list |
| 13:30-15:00 | **Advance DR4 prediction:** the frozen stars with ZTF period, type and amplitude, plus a control list of Gaia-quiet stars the detector scored low; the scoring script; success measures fixed | `dr4_prediction/` |
| 14:00-17:00 | **Decisions (god mode):** the lead objects for the paper and spectroscopy (shortest-period binaries, possible white-dwarf systems, colour-magnitude outliers, deepest eclipses); the VSX order; the paper outline | decision note |
| 17:00-19:00 | **Zenodo package** for Deepnil to upload on Friday 27: catalogue (VizieR-ready), folded light curves, method note, DR4 prediction, code snapshot, hashes | `zenodo_v1/` |
| 19:00 | Draft email to Prof. Seshadri (results, ATel credential request), for Deepnil's approval | draft only |
| after SCoPe finishes | DR3 rehearsal of the DR4 engine on GitHub; then the white-dwarf search | rehearsal report |

## What Deepnil does
* **Friday 27 November:** upload the Zenodo package under his ORCID. That date is the priority date.
* **When VSX reviews ZTF J164226.77+392523.3:** submit the next draft in the agreed order.
* Approve or edit the Seshadri email.

## The paper (after 26 November, not rushed into it)
Standard set on 24 November: sealed validation; completeness by injection on real ZTF sampling; purity measured
with TESS and with Gaia DR4 (the advance prediction); every catalogue overlap counted; at least one physical result;
the rare objects; a VizieR-ready table and code with a DOI; five figures. A separate agent that has not seen the
drafting reviews it as a hostile referee, and every point is fixed before Deepnil receives it. Target AJ or MNRAS
(ApJS if the DR4 purity and the physics are strong). First full draft 30 November; DR4 results added from 2 December.

## Contingencies
* **Search not finished by 12:00:** R1(b).
* **SCoPe jobs fail or queue:** run `scope_index.py` in the workspace for the fields that contain catalogue stars only.
* **A check service is down:** retry; stars it could not check are "unverified" and not claimed (R3).
* **The count drops sharply after prior art (expected: in a first SCoPe field, 8 of 9 candidates were already
  listed):** report it honestly. The paper then leads with what only this work has: the DR4 advance test, measured
  completeness and purity, and an independent check of SCoPe itself on Gaia-quiet stars (how often its high-vnv
  periods are confirmed season by season in independent light curves, and which real variables it scores low).
  The discoveries are the stars that pass every check, however many that is.
* **Freeze time:** if SCoPe finishes after 12:00, the freeze moves to SCoPe completion + 1 h; the Zenodo package
  still goes out Friday 27.
* **Anything else that changes what is claimed:** stop and ask Deepnil.


## Amendment after reading the DR4 draft data model (24 Nov, before the freeze; no rule above is changed)

- The advance DR4 prediction is scored against `phot_variable_flag` in its three DR4 values (VARIABLE,
  POSSIBLE, NON_VARIABLE), `phot_variable_score`, the per-classifier `vari_classifier_result` rows, and the
  periods in the specific `vari_*` tables. Success measures are fixed for each of the three flag values separately.
- Stars are matched DR3 → DR4 through `dr3_neighbourhood`, never by equal `source_id`; a 1" position match is
  the fallback and is reported as such.
- Details and code changes: DR4_ENGINE.md, section "Checked against the official draft DR4 data model".
