"""Checklist C4: verify every reference against Crossref (the DOI registry behind ADS records).
For each expected (first author, year, journal, volume, page/article) the script queries Crossref and records the DOI,
the registered title, volume, page and year, and whether they match. Output: refs_checked.csv / refs_checked.md."""
import json, time, urllib.parse, urllib.request
import pandas as pd

REFS = [  # key, query, expected year, volume, page/article
 ("Gaia2016", "Gaia Collaboration Prusti The Gaia mission", 2016, "595", "A1"),
 ("Gaia2023", "Gaia Collaboration Vallenari Gaia Data Release 3 Summary of the content and survey properties", 2023, "674", "A1"),
 ("Eyer2023", "Eyer Gaia Data Release 3 Summary of the variability processing and analysis", 2023, "674", "A13"),
 ("Rimoldini2023", "Rimoldini Gaia Data Release 3 All-sky classification of 12.4 million variable sources", 2023, "674", "A14"),
 ("Lebzelter2023", "Lebzelter Gaia Data Release 3 Second Gaia catalogue of long-period variable candidates", 2023, "674", "A15"),
 ("Mowlavi2023", "Mowlavi Gaia Data Release 3 The first Gaia catalogue of eclipsing-binary candidates", 2023, "674", "A16"),
 ("Clementini2023", "Clementini Gaia Data Release 3 Specific processing and validation of all-sky RR Lyrae and Cepheid stars", 2023, "674", "A18"),
 ("Distefano2023", "Distefano Gaia Data Release 3 Rotational modulation and patterns of colour variation in solar-like variables", 2023, "674", "A20"),
 ("Marton2023", "Marton Gaia Data Release 3 Gaia catalogue of young stellar object candidates", 2023, "674", "A21"),
 ("Gavras2023", "Gavras Gaia Data Release 3 Cross-match of Gaia sources with variable objects from the literature", 2023, "674", "A22"),
 ("Holl2018", "Holl Gaia Data Release 2 Summary of the variability processing and analysis results", 2018, "618", "A30"),
 ("Everall2022", "Everall Boubert Completeness of the Gaia-verse V astrometry and radial velocity sample selection functions", 2022, "509", "6205"),
 ("Bellm2019", "Bellm The Zwicky Transient Facility System Overview Performance and First Results", 2019, "131", "018002"),
 ("Masci2019", "Masci The Zwicky Transient Facility Data Processing Products and Archive", 2019, "131", "018003"),
 ("Chen2020", "Chen The Zwicky Transient Facility Catalog of Periodic Variable Stars", 2020, "249", "18"),
 ("Healy2024", "Healy The ZTF Source Classification Project III A Catalog of Variable Sources", 2024, "272", "14"),
 ("Ofek2020", "Ofek A catalogue of over 10 million variable source candidates in ZTF Data Release 1", 2020, "499", "5782"),
 ("Heinze2018", "Heinze A First Catalog of Variable Stars Measured by the Asteroid Terrestrial-impact Last Alert System ATLAS", 2018, "156", "241"),
 ("Jayasinghe2021", "Jayasinghe The ASAS-SN catalogue of variable stars IX The spectroscopic properties of Galactic variable stars", 2021, "503", "200"),
 ("Drake2014", "Drake The Catalina Surveys Periodic Variable Star Catalog", 2014, "213", "9"),
 ("Chen2018", "Chen The WISE Catalog of Periodic Variable Stars", 2018, "237", "28"),
 ("Prsa2022", "Prsa TESS Eclipsing Binary Stars I Short-cadence Observations of 4584 Eclipsing Binaries in Sectors 1-26", 2022, "258", "16"),
 ("Kounkel2022", "Kounkel Untangling the Galaxy IV Empirical constraints on angular momentum evolution and gyrochronology for young stars", 2022, "164", "137"),
 ("Mowlavi2021", "Mowlavi Large-amplitude variables in Gaia Data Release 2 Multi-band variability characterization", 2021, "648", "A44"),
 ("SanchezSaez2021", "Sanchez-Saez Alert Classification for the ALeRCE Broker System The Light Curve Classifier", 2021, "161", "141"),
 ("Forster2021", "Forster The Automatic Learning for the Rapid Classification of Events ALeRCE Alert Broker", 2021, "161", "242"),
 ("Watson2006", "Watson The International Variable Star Index VSX", 2006, "25", "47"),
 ("Chambers2016", "Chambers The Pan-STARRS1 Surveys", 2016, None, None),
 ("Lomb1976", "Lomb Least-squares frequency analysis of unequally spaced data", 1976, "39", "447"),
 ("Scargle1982", "Scargle Studies in astronomical time series analysis II Statistical aspects of spectral analysis of unevenly spaced data", 1982, "263", "835"),
 ("VanderPlas2018", "VanderPlas Understanding the Lomb-Scargle Periodogram", 2018, "236", "16"),
 ("Baluev2008", "Baluev Assessing the statistical significance of periodogram peaks", 2008, "385", "1279"),
 ("Kovacs2002", "Kovacs Zucker Mazeh A box-fitting algorithm in the search for periodic transits", 2002, "391", "369"),
 ("Pedregosa2011", "Pedregosa Scikit-learn Machine Learning in Python", 2011, "12", "2825"),
 ("Astropy2022", "Astropy Collaboration The Astropy Project Sustaining and Growing a Community-oriented Open-source Project and the Latest Major Release v5.0", 2022, "935", "167"),
 ("Dey2019", "Dey Overview of the DESI Legacy Imaging Surveys", 2019, "157", "168"),
 ("CantatGaudin2023", "Cantat-Gaudin An empirical model of the Gaia DR3 selection function", 2023, "669", "A55"),
]

def cr(q):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": q, "rows": 5})
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "SearchChev-reference-check"}), timeout=40) as r:
                return json.load(r)["message"]["items"]
        except Exception:
            time.sleep(3 * (k + 1))
    return []

rows = []
for key, q, yr, vol, pg in REFS:
    items = cr(q); best = None
    for it in items:
        y = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        v, p = str(it.get("volume", "")), str(it.get("page", it.get("article-number", ""))).split("-")[0]
        art = str(it.get("article-number", ""))
        ok = (yr is None or y == yr) and (vol is None or v == vol) and (pg is None or p == pg or art == pg or p.lstrip("0") == pg.lstrip("0"))
        if ok: best = it; break
    it = best or (items[0] if items else {})
    rows.append(dict(key=key, match=bool(best), doi=it.get("DOI", ""), year=(it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                     journal=(it.get("container-title") or [""])[0], volume=it.get("volume", ""), page=it.get("page", it.get("article-number", "")),
                     first_author=((it.get("author") or [{}])[0].get("family", "")), title=(it.get("title") or [""])[0][:110],
                     expected=f"{yr} {vol} {pg}"))
    time.sleep(0.4)
D = pd.DataFrame(rows); D.to_csv("refs_checked.csv", index=False)
with open("refs_checked.md", "w") as fh:
    fh.write("# Reference check (checklist C4), Crossref, 28 Nov 2026\n\nmatch = year, volume and page/article agree with the registered record.\n\n")
    fh.write(D.to_markdown(index=False))
print(D[["key", "match", "doi", "year", "volume", "page", "first_author"]].to_string())
