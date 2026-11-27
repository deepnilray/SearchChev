# SearchChev: handoff for a new session (27 Nov 2026, 12:15 IST)

Owner: Deepnil Ray (ORCID 0000-0001-8779-0700), Ashoka University undergraduate. Repo: github.com/deepnilray/SearchChev.
Everything below is on branch `dr4-engine` unless stated.

## What the project is
A search of ZTF DR23 light curves for periodic variable stars among Gaia DR3 stars that DR3 left without any
variability result. Detector frozen and validated on sealed sky tiles (AUC 0.954, false-positive rate 0.23%,
recovers 63% of sealed Gaia variables). Prof. T. R. Seshadri (Ashoka) reviewed it and said the METHOD matters more
than individual stars: lead every write-up with the method and the "what Gaia misses" result.

## Results at the freeze (27 Nov 2026, 03:45 UTC; files in `freeze/`, `zenodo_v1/`)
- 2,796 HATS sky partitions (orders 4/5/6), 10,700 deg2, |b| >= 15, dec >= -25, 13 <= G <= 18.5.
- 40.4 million Gaia-quiet stars; 13.9 million scored by the detector.
- 5,778 grade A/B periodic variables (catalogue_v1). SCoPe check of all: 86.8% known as variable, 11.7% in SCoPe
  below its threshold, 1.5% absent, 18 outside SCoPe coverage.
- 90 listed as variable in no catalogue -> 2 dropped (blends), 32 held back (blend/contamination/magnitude mismatch),
  24 too weak, **32 new variable stars** (`zenodo_v1/new_variables_v1.csv`): incl. detached eclipsing binaries at
  1.89, 3.02 and 25.4 d (Gaia DR3 1198491283713804928, 1052374297278903936, 346276760375634816) and contact
  binaries at 0.248 and 0.290 d.
- Population result: missed variables are mostly K/M dwarfs (62% with BP-RP > 1.8), amplitude ~0.1 mag, and Gaia
  observed them as often as matched quiet stars (391 pairs, Wilcoxon p = 0.33).
- Pre-registered DR4 prediction: `dr4_prediction/` (5,778 candidates, 550 matched controls, score_dr4.py, P1-P3).

## Deliverables ready
- `searchchev_v1_zenodo.zip` (rebuilt 27 Nov after the area correction; SHA-256 in `freeze/ZENODO_ZIP_SHA256`) and
  `zenodo_v1/ZENODO_FORM.md` (field-by-field). Deepnil uploads it himself under his ORCID. NOT yet uploaded.
- Preprint: `paper/preprint/searchchev_preprint.tex/.pdf` (8 pages, AASTeX701 twocolumn; needs aastex701.cls,
  epsf.sty, ulem.sty in the same folder). Zenodo DOI reserved: 10.5281/zenodo.24579691 (draft record zenodo.org/uploads/24579691; NOT yet published) and already in the preprint and RNAAS note. Post on Preprints.org
  (Physical Sciences -> Astronomy and Astrophysics) after the DOI is in. One preprint server only.
- RNAAS note for 2 Dec: `paper/searchchev_rnaas.tex` (red \dr{} placeholders = DR4 numbers; free, no APC).
- Two-page summary for Seshadri + email: draft in Deepnil's Ashoka Gmail (not sent; he sends it).

## Running now (never stop or slow these)
- `realsky-search` (branch realsky-validation, results to realsky-results): self-restarting, more sky for v2.
- `denominator` (workflow on main, script denom.py on dr4-engine, results to branch denom-results), run
  45952407890 started 27 Nov 06:31 UTC: frozen detector on Gaia-FLAGGED variables in a seeded 10% of partitions
  (freeze/denominator_tiles.json, 280 partitions) -> Gaia DR3's catch rate (selection function). Self-restarts.

## Scheduled sessions (they fire into the ORIGINAL session, not a new one)
Fri 15:30 denominator check; Sat 09:15 freeze loose ends; Sat 10:45 selection-function analysis + preprint v2;
Sat 10:00 pre-mortem; Sat 12:30 paper decisions; Sat 16:00 DR4-engine rehearsal; Mon 30 Nov 10:15 final paper +
independent hostile-referee review; Wed 2 Dec 09:00 DR4 launch; Wed 2 Dec 22:00 RNAAS fill.

## Journals (checked 27 Nov)
MNRAS charges up to GBP 2,310 (waiver policy exists); Ashoka has no OUP deal found (its CUP deal is humanities
only); ONOS APC scheme is for government institutions. Free options: Astronomy and Computing (subscription route,
best fit for a methods paper), Astrophysics and Space Science, Astronomische Nachrichten, RMxAA, Open Journal of
Astrophysics (free, needs arXiv; Seshadri can endorse/co-author), JAAVSO (for the 32 stars; free for AAVSO members).
Email sent to library@ashoka.edu.in asking about APC support; await reply.

## Rules Deepnil set
- He uploads to Zenodo and submits to TNS/VSX himself; never enter his passwords; he may log in and let you fill forms,
  final Submit only after his "yes" in chat.
- Important, time-sensitive actions: just do them (standing permission, 27 Nov).
- Never invent facts; check every number against files before it goes in anything public.
- Emails: personal Gmail by default; Ashoka account when he asks.
- Pending on him: Zenodo upload + DOI; send Seshadri draft; correct the pending VSX remark to cite SCoPe; do not
  submit the five VSX drafts until re-checked.
