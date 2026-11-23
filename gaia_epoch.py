"""Reader for Gaia epoch photometry bulk files (DR3 format; DR4 is expected to follow the same layout).

ESA serves epoch photometry as gzipped CSV files on an S3-style CDN, one file per HEALPix level-8 range
(EpochPhotometry_<first>-<last>.csv.gz). Each row is one source; the per-transit values are stored as
array columns written like "[1.0,2.0,null]". This module lists those files, streams one at a time and
returns clean per-band light curves, so a GitHub runner never holds more than one file on disk.

usage:
    python gaia_epoch.py list  gdr3            # print files with their HEALPix ranges and sizes
    python gaia_epoch.py read  <file.csv.gz>   # summary of one local file
"""
import gzip, io, re, sys, time
import numpy as np, pandas as pd, requests

BUCKET = "https://gaia.eu-1.cdn77-storage.com/"
CDN = "https://cdn.gea.esac.esa.int/"
PREFIX = {"gdr3": "Gaia/gdr3/Photometry/epoch_photometry/",
          "gdr4": "Gaia/gdr4/"}          # DR4 path is unknown until release; list_files searches under it
GAIA_T0_JD = 2455197.5                    # Gaia time origin: TCB JD 2455197.5 (2010-01-01T00:00)
BANDS = {"G": ("g_transit_time", "g_transit_mag", "g_transit_flux_over_error"),
         "BP": ("bp_obs_time", "bp_mag", "bp_flux_over_error"),
         "RP": ("rp_obs_time", "rp_mag", "rp_flux_over_error")}


def list_files(release="gdr3", pattern=r"EpochPhotometry_(\d+)-(\d+)\.csv\.gz$"):
    """All epoch-photometry files of a release: list of dict(key, hp_first, hp_last, size)."""
    out, marker, rx = [], "", re.compile(pattern)
    while True:
        r = requests.get(BUCKET, params=dict(prefix=PREFIX[release], marker=marker), timeout=120)
        r.raise_for_status()
        keys = re.findall(r"<Key>([^<]+)</Key>\s*<LastModified>[^<]*</LastModified>\s*<ETag>[^<]*</ETag>\s*<Size>(\d+)</Size>",
                          r.text) or [(k, 0) for k in re.findall(r"<Key>([^<]+)</Key>", r.text)]
        for k, s in keys:
            m = rx.search(k)
            if m:
                out.append(dict(key=k, hp_first=int(m.group(1)), hp_last=int(m.group(2)), size=int(s)))
        if "<IsTruncated>true</IsTruncated>" not in r.text:
            return out
        marker = re.search(r"<NextMarker>([^<]+)</NextMarker>", r.text)
        marker = marker.group(1) if marker else keys[-1][0]


def download(key, dest, tries=4):
    for k in range(tries):
        try:
            with requests.get(CDN + key, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return dest
        except Exception:
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"download failed: {key}")


def _arr(s):
    """'[1.0,null,2.0]' -> float array with NaN for null; empty or missing -> empty array."""
    if not isinstance(s, str) or len(s) < 3:
        return np.array([], float)
    return np.array([np.nan if v in ("null", "") else float(v) for v in s[1:-1].split(",")], float)


def read_file(path, bands=("G", "BP", "RP"), min_points=10, source_ids=None):
    """Yield (source_id, {band: DataFrame(t_bjd, mag, snr)}) for every source in one bulk file.
    Points with missing values or flagged as rejected are dropped; t_bjd is BJD(TCB) = Gaia time + T0."""
    cols = ["source_id"] + [c for b in bands for c in BANDS[b]] + ["rejected_by_photometry", "rejected_by_variability"]
    with gzip.open(path, "rt") as fh:
        head = [l for l in (fh.readline() for _ in range(2000)) if not l.startswith("#")]
    have = set(head[0].strip().split(",")) if head else set()
    use = [c for c in cols if c in have]
    for chunk in pd.read_csv(path, comment="#", usecols=use, chunksize=2000, dtype=str):
        for row in chunk.itertuples(index=False):
            r = row._asdict(); sid = int(r["source_id"])
            if source_ids is not None and sid not in source_ids:
                continue
            rej = np.zeros(0, bool)
            for flag in ("rejected_by_photometry", "rejected_by_variability"):
                if flag in r:
                    a = np.array([v == "true" for v in str(r[flag]).strip("[]").split(",")]) if isinstance(r[flag], str) else np.zeros(0, bool)
                    rej = a if rej.size == 0 else (rej | a if a.size == rej.size else rej)
            lcs = {}
            for b in bands:
                tc, mc, sc = BANDS[b]
                if tc not in r:
                    continue
                t, m, snr = _arr(r[tc]), _arr(r[mc]), _arr(r[sc])
                n = min(len(t), len(m), len(snr))
                t, m, snr = t[:n], m[:n], snr[:n]
                ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(snr) & (snr > 0)
                if b == "G" and rej.size == n:
                    ok &= ~rej
                if ok.sum() >= min_points:
                    lcs[b] = pd.DataFrame(dict(t_bjd=t[ok] + GAIA_T0_JD, mag=m[ok], magerr=1.0857 / snr[ok]))
            if lcs:
                yield sid, lcs


def healpix8(source_id):
    """HEALPix level-8 (nested) index of a Gaia source: source_id // 2**35 // 4**(12-8)."""
    return int(source_id) // (2 ** 35) // (4 ** 4)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        fs = list_files(sys.argv[2] if len(sys.argv) > 2 else "gdr3")
        print(len(fs), "files,", round(sum(f["size"] for f in fs) / 1e12, 3), "TB")
        for f in fs[:3] + fs[-2:]:
            print(f)
    elif cmd == "read":
        n = npts = 0; lens = []
        for sid, lcs in read_file(sys.argv[2]):
            n += 1; lens.append(len(lcs.get("G", [])))
        print("sources", n, "median G points", int(np.median(lens)), "max", max(lens))
