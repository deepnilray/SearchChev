import json,glob,sys,numpy as np,pandas as pd
from scipy.stats import beta
d=pd.concat([pd.read_csv(f) for f in glob.glob('/tmp/rr/shards/*/*/dossier.csv')]).drop_duplicates('source_id')
d=d[d.auto_grade.isin(['A','B'])].copy()
d['P']=d.period_refined.fillna(d.period)
def fl(x):
    L=json.loads(x);c={}
    for e in L:c[e['field']]=c.get(e['field'],0)+e.get('n',1)
    return c
d['fc']=d.detail.map(fl); d['prim']=d.fc.map(lambda c:max(c,key=c.get))
bund=sys.argv[1:]
fm=json.load(open('/tmp/fields.json'));fb={}
for k,v in fm.items():
    for x in v: fb.setdefault(x,set()).add(k[:5])
d['prim_in']=d.prim.map(lambda f:bool(fb.get(f,set())&set(bund)))
d['allf_in']=d.fc.map(lambda c:all(fb.get(f,set())&set(bund) for f in c))
S=pd.concat([pd.read_parquet(f'idx_{b}.parquet') for b in bund],ignore_index=True)
g={k:v for k,v in S.groupby('gaia_id')}
def cls(r):
    s=g.get(r.source_id)
    if s is None:return 'ABSENT',np.nan,np.nan,np.nan
    vmax=np.maximum(s.vnv_xgb,s.vnv_dnn)
    # per-model vnv>=0.8 (either model, any row)
    hi=(s.vnv_xgb>=0.8)|(s.vnv_dnn>=0.8)
    P=r.P;pm=np.zeros(len(s),bool)
    for m in(0.5,1,2): pm|=np.abs(s.period.values/(P*m)-1)<0.01
    pc_=(s.p1>=0.8)
    known=(hi&(pc_|pm)).any()
    if known:return 'KNOWN',vmax.max(),s.period.iloc[int(np.argmax(vmax.values))],s.class1.iloc[int(np.argmax(vmax.values))]
    if (vmax>=0.5).any() and not pm[(vmax>=0.5).values].any():
        i=int(np.argmax(vmax.values));return 'SCOPE_CANDIDATE',vmax.max(),s.period.iloc[i],s.class1.iloc[i]
    if (vmax>=0.5).any():  # vnv 0.5-0.8 but period matches (or >=0.8 w/o prob/period handled above)
        i=int(np.argmax(vmax.values));return 'KNOWN_PERIOD_MATCH_LOWVNV',vmax.max(),s.period.iloc[i],s.class1.iloc[i]
    i=int(np.argmax(vmax.values));return 'PRESENT_LOW',vmax.max(),s.period.iloc[i],s.class1.iloc[i]
r=d.apply(lambda x:pd.Series(cls(x),index=['cls','vnv','sp','sclass']),axis=1);d=pd.concat([d,r],axis=1)
d.to_csv('classified.csv',index=False)
def wil(k,n,z=1.96):
    p=k/n;den=1+z*z/n;c=(p+z*z/2/n)/den;h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/den;return c-h,c+h
for name,m in(('primary-field in bundles',d.prim_in),('all fields in bundles',d.allf_in)):
    print('==',name,m.sum())
    for gr in('A','B','A+B'):
        s=d[m&(d.auto_grade.isin(['A','B'] if gr=='A+B' else [gr]))];n=len(s)
        print(gr,n,s.cls.value_counts().to_dict())
        nk=(s.cls!='KNOWN').sum();lo,hi=wil(nk,n);print('  nonKNOWN %d/%d=%.3f  [%.3f,%.3f]'%(nk,n,nk/n,lo,hi))
        for c in s.cls.unique():
            k=(s.cls==c).sum();lo,hi=wil(k,n);print('   %s %.3f [%.3f,%.3f]'%(c,k/n,lo,hi))
