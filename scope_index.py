"""scope_index.py - compact prior-art index of the ZTF SCoPe classification catalogue, latest version
(Healy et al.; Zenodo record 20057104: 1,648,948,910 light curves in 636 ZTF fields, 24 zip bundles of parquet files).

SCoPe lists every light curve with a period and class probabilities. Kept here: every row where either model gives
the variable/non-variable class (vnv) a probability of at least 0.5, with its period and the two most probable
classes, so that the prior-art rule (R3.3 in PLAN_26NOV.md) can be applied exactly.

usage: python scope_index.py <bundle, e.g. 11_10> <out.parquet>
"""
import io, os, sys, zipfile, requests
import pandas as pd, pyarrow.parquet as pq

URL = "https://zenodo.org/api/records/20057104/files/{}_fields.zip/content"
SKIP = {"vnv", "pnp"}           # pnp = periodic/non-periodic, not a class


def download(bundle, dest):
    for k in range(5):
        try:
            with requests.get(URL.format(bundle), stream=True, timeout=900) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for ch in r.iter_content(8 << 20):
                        f.write(ch)
            zipfile.ZipFile(dest).testzip()
            return dest
        except Exception as ex:
            err = ex
    raise RuntimeError(f"download failed: {err}")


def index_zip(path, only_ids=None):
    """Stream every parquet member in batches, keeping only rows with vnv >= 0.5 (and, if given, only some Gaia IDs),
    so memory stays small even for 10 GB bundles."""
    import pyarrow.compute as pc
    z = zipfile.ZipFile(path); parts = []
    for n in z.namelist():
        if not n.endswith(".parquet"):
            continue
        with z.open(n) as fh:
            f = pq.ParquetFile(io.BytesIO(fh.read()))
        names_all = f.schema.names
        cls = sorted({c[:-4] for c in names_all if c.endswith("_xgb")} - SKIP)
        cols = ["Gaia_EDR3___id", "period", "field", "filter", "vnv_dnn", "vnv_xgb"] + \
               [f"{c}_{m}" for c in cls for m in ("xgb", "dnn") if f"{c}_{m}" in names_all]
        for b in f.iter_batches(batch_size=200_000, columns=cols):
            keep = pc.or_(pc.greater_equal(b["vnv_dnn"], 0.5), pc.greater_equal(b["vnv_xgb"], 0.5))
            if only_ids is not None:
                keep = pc.and_(keep, pc.is_in(b["Gaia_EDR3___id"], value_set=only_ids))
            b = b.filter(keep)
            if b.num_rows == 0:
                continue
            d = b.to_pandas()
            cl = [c for c in cls if f"{c}_dnn" in d and f"{c}_xgb" in d]
            sc = pd.DataFrame({c: d[[f"{c}_xgb", f"{c}_dnn"]].max(axis=1) for c in cl})
            order = sc.values.argsort(axis=1)[:, ::-1]; nm = sc.columns.values; ix = range(len(sc))
            parts.append(pd.DataFrame(dict(
                gaia_id=d.Gaia_EDR3___id.astype("int64").values, period=d.period.astype("float32").values,
                field=d.field.astype("int16").values, filter=d["filter"].astype("int8").values,
                vnv=d[["vnv_dnn", "vnv_xgb"]].max(axis=1).astype("float32").values,
                class1=nm[order[:, 0]], p1=sc.values[ix, order[:, 0]].astype("float32"),
                class2=nm[order[:, 1]], p2=sc.values[ix, order[:, 1]].astype("float32"))))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


if __name__ == "__main__":
    bundle, out = sys.argv[1], sys.argv[2]
    src = sys.argv[3] if len(sys.argv) > 3 else download(bundle, f"{bundle}.zip")
    d = index_zip(src); d.to_parquet(out, index=False)
    print(bundle, len(d), "rows with vnv >= 0.5")
    if len(sys.argv) <= 3:
        os.remove(src)
