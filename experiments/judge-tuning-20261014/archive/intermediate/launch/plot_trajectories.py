"""Plot existing verified trials; no inference or experiment-state changes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

R=Path('/private/tmp/judgearena-tuning-intermediate-20261014')
A=R/'analysis'
df=pd.read_parquet(A/'verified-trials.parquet')
summary=json.loads((A/'comparison.json').read_text())
labels={'uniform':'Uniform MO-Hyperband','primo-none':'PriMO without priors','primo-prior':'PriMO with priors'}
colors={'uniform':'#2878b5','primo-none':'#d47721','primo-prior':'#328654'}
markers={400:'o',1200:'s',3548:'^'}
H=28540
rows=[]
for arm in labels:
    trials=df[df.arm==arm].set_index('neps_trial_id')
    logical=0
    for attempt_index,attempt in enumerate(summary[arm]['attempt_trace'],1):
        if attempt['status']!='finished':
            continue
        trial=Path(attempt['config_path']).parent.name.removeprefix('config_')
        row=trials.loc[trial].to_dict()
        logical+=1
        row.update(arm=arm,trial=trial,completed_trial=logical,attempt=attempt_index,
                   attempted_battles=attempt['cumulative_battles'])
        rows.append(row)
data=pd.DataFrame(rows)
data.to_csv(A/'trajectory-points.csv',index=False)

def front(d):
    keep=[]
    for i,r in d.iterrows():
        dom=(d.agreement>=r.agreement)&(d.cost_per_1k_battles<=r.cost_per_1k_battles)&((d.agreement>r.agreement)|(d.cost_per_1k_battles<r.cost_per_1k_battles))
        if not dom.any(): keep.append(i)
    return d.loc[keep].sort_values('cost_per_1k_battles')

def save(fig,name,caption):
    for ax in fig.axes:
        if ax.get_xscale()=='log':
            ax.xaxis.set_major_locator(FixedLocator([.3,.5,1,2,5,10]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
            ax.xaxis.set_minor_locator(NullLocator())
    fig.text(.02,.015,caption,fontsize=8,color='#444444',va='bottom')
    fig.tight_layout(rect=(0,.10,1,.88))
    for ext in ['png','pdf']: fig.savefig(A/f'{name}.{ext}',dpi=180)
    plt.close(fig)

# Use circle area to encode fidelity consistently in every figure.
sizes={400:24,1200:58,3548:125}
algorithm_handles=[Line2D([],[],color=colors[a],lw=2,label=l) for a,l in labels.items()]
fidelity_handles=[Line2D([],[],marker='o',color='#888888',ls='',markersize=np.sqrt(s),label=f'{n:,}-battle panel') for n,s in sizes.items()]

fig,ax=plt.subplots(figsize=(10,6.4))
axes=[ax]
for ax in axes:
    for arm in labels:
        d=data[data.arm==arm]; eligible=d[d.attempted_battles<=H]
        for n,size in sizes.items():
            q=eligible[eligible.n_battles==n]
            ax.scatter(q.cost_per_1k_battles,100*q.agreement,color=colors[arm],alpha=.20,s=size,marker='o',linewidths=0)
            q=d[(d.n_battles==n)&(d.attempted_battles>H)]
            ax.scatter(q.cost_per_1k_battles,100*q.agreement,edgecolors=colors[arm],facecolors='none',alpha=.38,s=size,marker='o')
    pool=data[data.attempted_battles<=H]
    f=front(pool)
    ax.plot(f.cost_per_1k_battles,100*f.agreement,color='#444444',lw=2,zorder=4)
    for arm in labels:
        q=f[f.arm==arm]
        ax.scatter(q.cost_per_1k_battles,100*q.agreement,color=colors[arm],s=q.n_battles.map(sizes),marker='o',edgecolor='white',linewidth=.6,zorder=5)
    ax.set_xlabel('Expected USD / 1,000 battles');ax.set_ylabel('Validation agreement (%)');ax.grid(alpha=.16)
ax.set(xscale='log',title='Observed Pareto front across all fidelities',xlabel='Expected USD / 1,000 battles (log scale)')
fig.legend(handles=[Line2D([],[],marker='o',color=colors[a],ls='',label=l) for a,l in labels.items()]+fidelity_handles+[Line2D([],[],color='#444444',lw=2,label='Combined observed Pareto front'),Line2D([],[],marker='o',color='#888888',markerfacecolor='none',ls='',label='Beyond common battle budget')],loc='upper center',ncol=3,fontsize=9)
save(fig,'pareto-all-points','Source: 50 verified trial reports, 14–15 Oct 2026. Circle area encodes panel size; color encodes algorithm. Faded: other evaluations; hollow: beyond common horizon.\nFront pools algorithms and fidelities within 28,540 attempted config-battles. Panel sizes differ: this compares observed estimates, not equal-fidelity performance.')

# Mixed-fidelity incumbent: maximum observed agreement across all completed trials.
fig,ax=plt.subplots(figsize=(10,6.4))
incumbents=[]
for arm in labels:
    d=data[data.arm==arm].sort_values('completed_trial')
    improving=d.agreement.gt(d.agreement.cummax().shift(fill_value=-np.inf))
    changes=d[improving]
    incumbents.append(changes)
    xs=changes.attempted_battles.tolist()+[d.attempted_battles.max()]
    ys=(100*changes.agreement).tolist()
    ax.step(xs,ys+[ys[-1]],where='post',color=colors[arm],lw=2)
    for n,size in sizes.items():
        q=changes[changes.n_battles==n]
        ax.scatter(q.attempted_battles,100*q.agreement,color=colors[arm],s=size,marker='o',edgecolor='white',linewidth=.6,zorder=4)
ax.axvline(H,color='#777777',ls='-.',lw=1)
ax.set(xlabel='Cumulative attempted config-battle evaluations',ylabel='Best observed validation agreement (%)',title='Agreement incumbent across all fidelities')
ax.grid(alpha=.18)
fig.legend(handles=algorithm_handles+fidelity_handles,loc='upper center',ncol=3,fontsize=9)
save(fig,'search-trajectories','Source: verified trial reports and attempt logs, 14–15 Oct 2026. One best-so-far curve per algorithm, pooling observed scores across all fidelities.\nCircle size identifies the panel size of each incumbent-setting evaluation. Dash-dot line: 28,540 attempted config-battles; timeout work is included.\nMixed-fidelity scores use different validation panel sizes; this is an observed-search trajectory, not a full-fidelity performance comparison. One seed per arm.')
pd.concat(incumbents).to_csv(A/'incumbent-points.csv',index=False)

# Merge algorithms on common axes at each budget snapshot.
fig,axes=plt.subplots(1,3,figsize=(14,5.2),sharex=True,sharey=True)
for ax,budget in zip(axes,[10000,20000,H]):
    for arm in labels:
        d=data[(data.arm==arm)&(data.attempted_battles<=budget)]
        for n,size in sizes.items():
            q=d[d.n_battles==n]
            ax.scatter(q.cost_per_1k_battles,100*q.agreement,color=colors[arm],alpha=.18,s=size,marker='o',linewidths=0)
    f=front(data[data.attempted_battles<=budget])
    ax.plot(f.cost_per_1k_battles,100*f.agreement,color='#444444',lw=2)
    for arm in labels:
        q=f[f.arm==arm]
        ax.scatter(q.cost_per_1k_battles,100*q.agreement,color=colors[arm],s=q.n_battles.map(sizes),marker='o',edgecolor='white',linewidth=.6,zorder=5)
    ax.set(title=f'By {budget:,} attempted battles',xlabel='Expected USD / 1,000 battles',xscale='log');ax.grid(alpha=.15)
axes[0].set_ylabel('Validation agreement (%)')
fig.legend(handles=[Line2D([],[],marker='o',color=colors[a],ls='',label=l) for a,l in labels.items()]+fidelity_handles+[Line2D([],[],color='#444444',lw=2,label='Combined observed Pareto front')],loc='upper center',ncol=3,fontsize=9)
save(fig,'pareto-evolution','Source: verified results and attempt logs, 14–15 Oct 2026. Each panel overlays all three algorithms at the same attempted battle budget.\nCircle size encodes fidelity; faded observations include lower fidelities. Neutral front pools algorithms and fidelities at each budget; agreement estimates use different panel sizes.')
print(data.groupby('arm').size().to_dict())

front(data[data.attempted_battles<=H]).to_csv(A/'combined-pareto-front.csv',index=False)
