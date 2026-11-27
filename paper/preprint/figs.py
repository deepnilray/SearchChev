import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'savefig.dpi':300,'font.family':'serif'})
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
