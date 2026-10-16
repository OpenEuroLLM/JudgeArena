"""Compute predeclared held-out comparisons after all frozen candidates finish."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from judgearena.arenas_utils import extract_turn_text
from judgearena.pricing import TokenPrice
from judgearena.tuning.runner import _read_trial
R=Path('/work/dlclarge1/lushtake-hiwi/judge-tuning-heldout-3000-20261015')
m=json.loads((R/'configs/manifest.json').read_text())
ids=json.loads(Path(m['panel_file']).read_text())['test']['3000']
assert hashlib.sha256(Path(m['panel_file']).read_bytes()).hexdigest()==m['panel_sha256']
from judgearena.datasets import load_battles
from judgearena.tasks.registry import get_packaged_task
from judgearena.benchmarks.meta_eval.runner import _prepare_arena_battles
task=get_packaged_task('meta-eval-lmarena-140k-en')
arena=_prepare_arena_battles(load_battles(task),task=task.task,arena='LMArena-140k',languages=['en']).set_index('battle_id').loc[ids]
groups=arena.conversation_a.map(lambda turns:extract_turn_text(turns[0])).map(lambda t:hashlib.sha256(t.strip().encode()).hexdigest()).to_numpy()
rows=[];correct={}
for c in m['candidates']:
 cfg=Path(c['config_file']);assert hashlib.sha256(cfg.read_bytes()).hexdigest()==c['sha256']
 row=_read_trial(R/'results'/c['config_id'],TokenPrice(**m['prices'][c['model']],source='frozen_intermediate_reference'))
 folder=Path(row['result_dir']);b=pd.read_parquet(folder/'battles.parquet');b=b[b.sampled];assert len(b)==3000 and set(b.battle_id)==set(ids);b=b.set_index('battle_id').loc[ids]
 assert len(b)==3000 and set(b.index)==set(ids)
 pred=np.where(b.pref<.49,0,np.where(b.pref>.51,1,.5))
 values=((pred==b.reference_pref.to_numpy())&b.pref.notna().to_numpy()).astype(float)
 assert abs(values.mean()-row['agreement'])<1e-12
 assert abs(b.pref.notna().mean()-row['coverage'])<1e-12
 ann=pd.read_parquet(folder/'annotations.parquet');assert ann.battle_id.nunique()==len(ann)==3000 and set(ann.battle_id)==set(ids)
 usage=ann.usage_json.fillna('{}').map(json.loads)
 assert all((u or {}).get('output_tokens',0)<=8192 for u in usage)
 cost=sum(((u or {}).get('input_tokens',0) or 0)*m['prices'][c['model']]['input']+((u or {}).get('output_tokens',0) or 0)*m['prices'][c['model']]['output'] for u in usage)/1e6/3
 assert abs(cost-row['cost_per_1k_battles'])<1e-9
 correct[c['config_id']]=values
 row.update(**c,cap_requests=sum((u or {}).get('output_tokens',0)>=8192 for u in usage),unparsed=int(b.pref.isna().sum()))
 rows.append(row)
unique,inverse=np.unique(groups,return_inverse=True);counts=np.bincount(inverse)
rng=np.random.default_rng(0)
def bootstrap(values):
 sums=np.bincount(inverse,weights=values)
 draws=[]
 for _ in range(10000):
  ix=rng.integers(0,len(unique),size=len(unique));draws.append(sums[ix].sum()/counts[ix].sum())
 return np.asarray(draws)
for row in rows:
 draws=bootstrap(correct[row['config_id']]);row['bootstrap_se']=float(draws.std(ddof=1));row['bootstrap_95ci']=np.quantile(draws,[.025,.975]).tolist()
pairs=[]
for a,b in [('f1d8421d913f','ea6e4bd6e590'),('f1d8421d913f','325aa74cc942'),('ea6e4bd6e590','325aa74cc942')]:
 delta=correct[a]-correct[b];draws=bootstrap(delta)
 pairs.append({'a':a,'b':b,'delta':float(delta.mean()),'paired_95ci':np.quantile(draws,[.025,.975]).tolist(),'primary':a=='f1d8421d913f' and b=='ea6e4bd6e590'})
out={'n_battles':3000,'distinct_prompt_clusters':len(unique),'bootstrap_replicates':10000,'bootstrap_seed':0,'bootstrap_unit':'prompt cluster','configs':rows,'pairs':pairs,'interpretation':'Frozen validation-selected configs; no optimizer-seed uncertainty; no post-test selection or formal significance claims.'}
(R/'analysis/test-comparison.json').write_text(json.dumps(out,indent=2,default=str))
pd.DataFrame(rows).to_parquet(R/'analysis/test-configs.parquet',index=False)
fig,ax=plt.subplots(figsize=(8,5))
colors={'uniform':'#2878b5','primo-none':'#d47721','primo-prior':'#328654'}
for row in rows:
 y=row['agreement']*100;lo,hi=np.array(row['bootstrap_95ci'])*100
 ax.errorbar(row['cost_per_1k_battles'],y,yerr=[[y-lo],[hi-y]],fmt='o',color=colors[row['arm']],capsize=3)

ax.set(xlabel='Expected USD / 1,000 battles',ylabel='Held-out human agreement (%)',title='Five frozen configurations on 3,000 shared held-out battles')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',linestyle='',color=colors[a],label=l) for a,l in [('uniform','Uniform MO-Hyperband'),('primo-none','PriMO without priors'),('primo-prior','PriMO with priors')]],fontsize=9)
ax.grid(alpha=.2);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(R/f'analysis/test-cost-agreement.{ext}',dpi=180)
print('ANALYSIS_COMPLETE',len(rows),len(unique),flush=True)
