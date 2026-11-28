import pandas as pd, requests, io, json
from concurrent.futures import ThreadPoolExecutor
H=pd.read_pickle('/home/claude/sf/hits_freeze.pkl'); kn=H[H.status=='KNOWN'].copy()
gaia={'Gaia_DR3_vari','GaiaDR3_rotation','GaiaDR3_EB'}; weak={'GaiaDR2_LAV_Mowlavi2021','TESS_10k_unvetted','YSO_Delfini2025'}
s=kn.prior_art.fillna('').str.split(';').apply(lambda x:set(x)-{''}-gaia-weak)
a=kn[s.apply(lambda x:x=={'ZTF_ML_Arevalo2026'})]
def one(r):
    for k in range(3):
        try:
            x=requests.get("https://vizier.idia.ac.za/viz-bin/asu-tsv",params={"-source":"J/A+A/705/A247/catalog","-c":f"{r.ra:.6f} {r.dec:+.6f}","-c.rs":5,"-out":"_r,OID,predClass,predClassProb,predInitClass","-out.add":"_r","-sort":"_r","-out.max":5},timeout=90).text
            L=[l for l in x.splitlines() if l and not l.startswith('#')]
            if len(L)<=3: return dict(source_id=r.source_id, n=0)
            d=pd.read_csv(io.StringIO("\n".join([L[0]]+L[3:])),sep='\t')
            cls=set(d.predClass.astype(str).str.strip()); var=any(c not in ('nonvar-star','QSO','AGN','galaxy','Galaxy') for c in cls)
            return dict(source_id=r.source_id, n=len(d), classes=';'.join(sorted(cls)), any_variable_class=var)
        except Exception as e: err=repr(e)
    return dict(source_id=r.source_id, n=-1, error=err[:80])
import os, sys, time
OUT='/home/claude/sf/arevalo_classes_part.csv'
done=set(pd.read_csv(OUT).source_id) if os.path.exists(OUT) else set()
todo=[r for r in a.itertuples() if r.source_id not in done][:int(sys.argv[1]) if len(sys.argv)>1 else None]
t0=time.time()
with ThreadPoolExecutor(12) as ex: res=list(ex.map(one, todo))
D=pd.DataFrame(res); D.to_csv(OUT,mode='a',header=not os.path.exists(OUT),index=False)
A=pd.read_csv(OUT); print(f"chunk {len(D)} in {time.time()-t0:.0f}s; total {len(A)}/{len(a)}; errors {(A.n<0).sum()}")
