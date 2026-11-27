import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'savefig.dpi':300,'font.family':'serif','pdf.fonttype':42,'ps.fonttype':42})
c=pd.read_csv('/home/claude/searchchev/zenodo_v1/catalogue_v1.csv',low_memory=False)
g=pd.read_csv('gaia_params.csv');n=pd.read_csv('/home/claude/searchchev/zenodo_v1/new_variables_v1.csv')
c=c.merge(g[['source_id','parallax','parallax_over_error','phot_g_n_obs','l','b']],on='source_id',how='left')
known=(c.scope_status=='KNOWN')|c.known_nonscope
new=c.source_id.isin(n.source_id); other=~known&~new
GREY,BLUE,ORG='#b5b5b5','#257de0','#f47230'
def layers(ax,x,y,s=2):
    ax.scatter(x[known],y[known],s=s,c=GREY,lw=0,rasterized=True,label=f'known variable ({known.sum():,})')
    ax.scatter(x[other],y[other],s=s+0.5,c=BLUE,lw=0,rasterized=True,label=f'not previously classified ({other.sum():,})')
    ax.scatter(x[new],y[new],s=28,c=ORG,marker='*',edgecolors='k',linewidths=0.3,label=f'new, this work ({new.sum()})',zorder=5)
# Fig1 sky (Galactic, Mollweide)
fig=plt.figure(figsize=(7,3.4));ax=fig.add_subplot(111,projection='mollweide')
L=np.radians(((c.l+180)%360)-180)*-1;B=np.radians(c.b)
layers(ax,L,B,1.5);ax.grid(lw=0.3,color='#dddddd');ax.set_xticklabels([]);ax.tick_params(labelsize=6)
ax.legend(loc='lower center',bbox_to_anchor=(0.5,-0.2),ncol=3,frameon=False,fontsize=7,markerscale=2)
fig.savefig('fig_sky.pdf',bbox_inches='tight');plt.close()
# Fig2 period-amplitude
fig,ax=plt.subplots(figsize=(3.4,2.8))
P=c.P.copy();layers(ax,P,c.amp);ax.set_xscale('log');ax.set_yscale('log')
ax.set_xlabel('ZTF period (d)');ax.set_ylabel('ZTF $r$ amplitude, 5--95% range (mag)')
ax.legend(frameon=False,fontsize=6,loc='upper left',markerscale=2)
fig.savefig('fig_pamp.pdf',bbox_inches='tight');plt.close()
# Fig3 CMD
m=c.parallax_over_error>5
MG=c.G+5*np.log10(c.parallax/100.)
fig,ax=plt.subplots(figsize=(3.4,3.0))
ax.scatter(c.bp_rp[m&known],MG[m&known],s=2,c=GREY,lw=0,rasterized=True,label='known variable')
ax.scatter(c.bp_rp[m&other],MG[m&other],s=4,c=BLUE,lw=0,rasterized=True,label='not previously classified')
ax.scatter(c.bp_rp[m&new],MG[m&new],s=30,c=ORG,marker='*',edgecolors='k',linewidths=0.3,label='new, this work',zorder=5)
ax.invert_yaxis();ax.set_xlabel('$G_{BP}-G_{RP}$ (mag)');ax.set_ylabel('$M_G$ (mag)')
ax.legend(frameon=False,fontsize=6,markerscale=2);ax.set_title(f'parallax/error > 5 ({m.sum():,} stars)',fontsize=7)
fig.savefig('fig_cmd.pdf',bbox_inches='tight');plt.close()
# Fig4 transit counts candidates vs paired controls
k=pd.read_csv('/home/claude/searchchev/dr4_prediction/control_list.csv')
k=k.merge(g[['source_id','phot_g_n_obs']].rename(columns={'source_id':'control_source_id','phot_g_n_obs':'nc'}),on='control_source_id').merge(g[['source_id','phot_g_n_obs']].rename(columns={'source_id':'paired_candidate','phot_g_n_obs':'nk'}),on='paired_candidate')
fig,ax=plt.subplots(figsize=(3.4,2.6));bins=np.linspace(0,1000,41)
ax.hist(k.nk,bins=bins,histtype='step',color=BLUE,lw=1.4,label=f'candidates (n={len(k)})')
ax.hist(k.nc,bins=bins,histtype='step',color='#666666',lw=1.4,ls='--',label='matched quiet controls')
ax.set_xlabel('Gaia DR3 $G$-band observations (phot_g_n_obs)');ax.set_ylabel('stars');ax.legend(frameon=False,fontsize=6)
fig.savefig('fig_nobs.pdf',bbox_inches='tight');plt.close()
print('known',known.sum(),'other',other.sum(),'new',new.sum(),'cmd',m.sum(),'pairs',len(k))
# Fig 5: selection function (Gaia DR3 flagged fraction of ZTF-detected periodic variables)
T=pd.read_csv('selection_function_bins.csv')
fig,axs=plt.subplots(1,3,figsize=(7.1,2.4),sharey=True)
lab={'period':'ZTF period (d)','amp':'ZTF $r$ amplitude, 5--95% (mag)','G':'Gaia $G$ (mag)'}
for ax,var in zip(axs,['period','amp','G']):
    t=T[(T['var']==var)&((T.F+T.Q)>=30)].copy()
    if var=='amp': t.loc[t.hi>5,'hi']=0.8
    if var=='amp': t=t[t.lo>=0.05]
    x=np.sqrt(t.lo*t.hi) if var!='G' else (t.lo+t.hi)/2
    ax.fill_between(x,100*t.C_raw_lo,100*t.C_conf_hi,color=BLUE,alpha=0.18,lw=0,label='range of the two counts (95%)')
    ax.plot(x,100*t.C_raw,'-',color=BLUE,lw=1.2,label='all ZTF detections')
    ax.plot(x,100*t.C_conf,'--',color=BLUE,lw=1.2,label='independently supported')
    if 'C_sc' in t:
        ok=t.C_sc.notna()
        ax.errorbar(x[ok],100*t.C_sc[ok],yerr=[100*(t.C_sc-t.C_sc_lo)[ok],100*(t.C_sc_hi-t.C_sc)[ok]],fmt='o',ms=3.5,color=ORG,mec='k',mew=0.4,elinewidth=0.8,capsize=0,label='SCoPe-confirmed only',zorder=5)
    if var!='G': ax.set_xscale('log')
    if var=='amp':
        from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
        ax.xaxis.set_major_locator(FixedLocator([0.05,0.1,0.2,0.4])); ax.xaxis.set_major_formatter(FixedFormatter(['0.05','0.1','0.2','0.4']))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(lab[var]); ax.set_ylim(30,100); ax.grid(axis='y',lw=0.3,color='#e3e3e3')
axs[0].set_ylabel('Gaia DR3 flagged (%)')
axs[0].legend(frameon=False,fontsize=5.5,loc='lower left')
fig.tight_layout(w_pad=0.6); fig.savefig('fig_sf.pdf',bbox_inches='tight'); plt.close()
