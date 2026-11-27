import requests,io,time,pandas as pd
from astropy.io.votable import parse_single_table
from concurrent.futures import ThreadPoolExecutor
C=pd.read_csv('/home/claude/searchchev/zenodo_v1/catalogue_v1.csv',low_memory=False,usecols=['source_id'])
K=pd.read_csv('/home/claude/searchchev/dr4_prediction/control_list.csv',usecols=['control_source_id'])
ids=list(C.source_id.astype('int64'))+list(K.control_source_id.astype('int64'))
U="https://gaia.aip.de/tap/sync"
def q(chunk):
    Q="SELECT source_id,parallax,parallax_over_error,phot_g_n_obs,phot_g_mean_mag,bp_rp,ruwe,phot_bp_rp_excess_factor,l,b FROM gaiadr3.gaia_source WHERE source_id IN ("+",".join(map(str,chunk))+")"
    for t in range(5):
        try:
            x=requests.post(U,data=dict(REQUEST="doQuery",LANG="ADQL",QUERY=Q),timeout=180)
            if 'QUERY_STATUS" value="OK"' in x.text:
                return parse_single_table(io.BytesIO(x.content)).to_table().to_pandas().rename(columns={'datalinkID':'source_id'})
        except Exception as e: pass
        time.sleep(10+10*t)
    print('FAIL',chunk[0],flush=True);return None
ch=[ids[i:i+300] for i in range(0,len(ids),300)]
with ThreadPoolExecutor(3) as ex: R=list(ex.map(q,ch))
D=pd.concat([r for r in R if r is not None]);D.to_csv('gaia_params.csv',index=False);print(len(D),len(ids))
