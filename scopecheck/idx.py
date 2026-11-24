import sys,io,zipfile,json,glob
sys.path.insert(0,'/home/claude/searchchev')
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
d=pd.concat([pd.read_csv(f) for f in glob.glob('/tmp/rr/shards/*/*/dossier.csv')]).drop_duplicates('source_id')
d=d[d.auto_grade.isin(['A','B'])]
ids=pa.array(d.source_id.astype('int64').values)
SKIP={"vnv","pnp"}
z=zipfile.ZipFile(sys.argv[1]);parts=[]
for n in z.namelist():
    if not n.endswith('.parquet'):continue
    with z.open(n) as fh: f=pq.ParquetFile(io.BytesIO(fh.read()))
    na=f.schema.names; cls=sorted({c[:-4] for c in na if c.endswith('_xgb')}-SKIP)
    cols=["Gaia_EDR3___id","period","field","filter","vnv_dnn","vnv_xgb"]+[f"{c}_{m}" for c in cls for m in("xgb","dnn") if f"{c}_{m}" in na]
    for b in f.iter_batches(batch_size=200000,columns=cols):
        b=b.filter(pc.is_in(b["Gaia_EDR3___id"],value_set=ids))
        if b.num_rows==0:continue
        x=b.to_pandas()
        cl=[c for c in cls if f"{c}_dnn" in x and f"{c}_xgb" in x]
        sc=pd.DataFrame({c:x[[f"{c}_xgb",f"{c}_dnn"]].max(axis=1) for c in cl})
        o=sc.values.argsort(axis=1)[:,::-1];nm=sc.columns.values;ix=range(len(sc))
        parts.append(pd.DataFrame(dict(gaia_id=x.Gaia_EDR3___id.astype('int64').values,period=x.period.astype('float64').values,field=x.field.astype('int16').values,filter=x['filter'].astype('int8').values,vnv_xgb=x.vnv_xgb.values,vnv_dnn=x.vnv_dnn.values,class1=nm[o[:,0]],p1=sc.values[ix,o[:,0]],class2=nm[o[:,1]],p2=sc.values[ix,o[:,1]])))
r=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
r.to_parquet(sys.argv[2],index=False);print(sys.argv[1],len(r),r.gaia_id.nunique() if len(r) else 0)
