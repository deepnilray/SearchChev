"""Score the SearchChev advance prediction against Gaia DR4 (pre-registered 27 Nov 2026, before DR4).
Inputs (all CSV):
  prediction_candidates.csv, control_list.csv   (this folder, frozen; hashes in SHA256SUMS)
  dr4_xmatch.csv  columns: dr3_source_id, dr4_source_id, match ('neighbourhood' or 'position_1arcsec'),
                  phot_variable_flag (VARIABLE / POSSIBLE / NON_VARIABLE), has_epoch_photometry,
                  optional: best_class_name, best_class_score, period_<table> columns
  dr4_lc_periods.csv (optional) columns: dr3_source_id, fap_at_ztf_period  (Lomb-Scargle on the DR4 G light curve,
                  best of 0.5x, 1x, 2x the ZTF period, searched within 1%)
Output: score.json with every pre-registered number and its 95% Wilson interval."""
import sys,json,numpy as np,pandas as pd
def wil(k,n,z=1.96):
    if n==0: return [None,None]
    p=k/n;den=1+z*z/n;c=(p+z*z/2/n)/den;h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/den;return [c-h,c+h]
def frac(s,mask):
    n=len(s);k=int(mask.sum());return dict(k=k,n=n,f=(k/n if n else None),ci95=wil(k,n))
C=pd.read_csv('prediction_candidates.csv');K=pd.read_csv('control_list.csv')
X=pd.read_csv(sys.argv[1]);out={}
for name,S,key in(('candidates',C,'source_id'),('control',K,'control_source_id')):
    m=S.merge(X,left_on=key,right_on='dr3_source_id',how='left')
    got=m[m.phot_variable_flag.notna()]
    out[name]=dict(n_frozen=len(S),n_matched=len(got),match_type=got.match.value_counts().to_dict())
    for flag in('VARIABLE','POSSIBLE','NON_VARIABLE'):
        out[name][flag]=frac(got,got.phot_variable_flag.eq(flag))
    out[name]['VARIABLE_or_POSSIBLE']=frac(got,got.phot_variable_flag.isin(['VARIABLE','POSSIBLE']))
    if len(sys.argv)>2:
        L=pd.read_csv(sys.argv[2]);g=m.merge(L,on='dr3_source_id',how='inner')
        out[name]['period_confirmed_fap_1e-3']=frac(g,g.fap_at_ztf_period<1e-3)
# P1: two-proportion z-test, candidates vs control, VARIABLE
a=out['candidates']['VARIABLE'];b=out['control']['VARIABLE']
if a['n'] and b['n']:
    p=(a['k']+b['k'])/(a['n']+b['n']);se=np.sqrt(p*(1-p)*(1/a['n']+1/b['n']))
    out['P1_z']=(a['f']-b['f'])/se if se>0 else None
json.dump(out,open('score.json','w'),indent=1,default=float);print(json.dumps(out,indent=1,default=float))
