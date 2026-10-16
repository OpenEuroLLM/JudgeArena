"""Freeze an outcome-blind held-out panel and validation-selected candidates."""
import gzip,hashlib,json
from pathlib import Path
from collections import Counter
import yaml
from judgearena.arenas_utils import extract_turn_text
from judgearena.benchmarks.meta_eval.runner import _prepare_arena_battles
from judgearena.benchmarks.meta_eval.sampling import filter_split,select_top_models,comparison_components
from judgearena.datasets import load_battles
from judgearena.tasks.registry import get_packaged_task
from judgearena.config import RunConfig
R=Path('/work/dlclarge1/lushtake-hiwi/judge-tuning-heldout-3000-20261015')
OLD=Path('/work/dlclarge1/lushtake-hiwi/judge-tuning-intermediate-20261014')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(text):return hashlib.sha256(text.strip().encode()).hexdigest()
task=get_packaged_task('meta-eval-lmarena-140k-en')
arena=_prepare_arena_battles(load_battles(task),task=task.task,arena='LMArena-140k',languages=['en'])
models,pool=select_top_models(arena,top_models=20)
pool=pool.set_index('battle_id',drop=False)
prompts=pool.conversation_a.map(lambda turns:extract_turn_text(turns[0]))
validation=filter_split(pool.reset_index(drop=True),prompts,split='validation',validation_fraction=.5)
test=filter_split(pool.reset_index(drop=True),prompts,split='test',validation_fraction=.5)
v1=json.load(gzip.open(R/'data/v1-input.json.gz'))
old_test={b['battle_id'] for row in v1['heldout'] for b in row['battles']}
assert old_test<=set(pool.index)
excluded_ids=set(validation.battle_id)|old_test
excluded_prompts=set(prompts.loc[list(excluded_ids)].map(identity))
fresh=test[~test.conversation_a.map(lambda turns:identity(extract_turn_text(turns[0]))).isin(excluded_prompts)].copy()
existing=json.loads((OLD/'panels/panels.json').read_text())['test']['100']
audit={'pool_battles':len(pool),'test_partition_battles':len(test),'fresh_battles':len(fresh),'old_test_battles':len(old_test),'excluded_prompt_identities':len(excluded_prompts),'existing_reserved_battles':len(existing),'existing_still_eligible':set(existing)<=set(fresh.battle_id),'selected_count':0,'complete':False}
(R/'analysis/panel-audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit),flush=True)
invalid=[b for b in existing if b not in set(fresh.battle_id)]
audit['removed_reserved_battles']=[{'battle_id':b,'in_test_partition':b in set(test.battle_id),'excluded_raw_prompt':prompts.loc[b] in set(prompts.loc[list(excluded_ids)]),'excluded_stripped_prompt':identity(prompts.loc[b]) in excluded_prompts} for b in invalid]
assert all(x['in_test_partition'] and x['excluded_stripped_prompt'] for x in audit['removed_reserved_battles']), 'Unexpected reserved-panel discrepancy; stop'
existing=[b for b in existing if b in set(fresh.battle_id)]
audit['retained_reserved_battles']=len(existing)
(R/'analysis/panel-audit.json').write_text(json.dumps(audit,indent=2))
print('RESERVED_PANEL_REVIEW',json.dumps(audit['removed_reserved_battles']),flush=True)
assert len(fresh)>=3000,'Insufficient fresh held-out battles; do not change population'
remaining=sorted(set(fresh.battle_id)-set(existing),key=lambda b:hashlib.sha256(f'heldout3000-seed0\0{b}'.encode()).hexdigest())
ids=existing+remaining[:3000-len(existing)]
selected=pool.loc[ids]
assert len(ids)==len(set(ids))==3000
assert not set(ids)&excluded_ids
assert not set(selected.conversation_a.map(lambda turns:identity(extract_turn_text(turns[0]))))&excluded_prompts
assert comparison_components(selected.reset_index(drop=True),models)==[frozenset(models)]
payload={'test':{'3000':ids}}
panel=R/'panels/heldout-3000.json';panel.write_text(json.dumps(payload,indent=2))
counts=Counter(selected.model_a)+Counter(selected.model_b)
audit.update(selected_count=len(ids),distinct_prompts=selected.conversation_a.map(lambda turns:identity(extract_turn_text(turns[0]))).nunique(),per_model_battles=dict(counts),panel_sha256=sha(panel),complete=True,selection=f'Retain{len(existing)} eligible reserved rows; replace ineligible rows and expand to3000 by fixed seed0 SHA256 priority; no label-dependent selection')
(R/'analysis/panel-audit.json').write_text(json.dumps(audit,indent=2))
choices=[('uniform','325aa74cc942','primary'),('primo-none','ea6e4bd6e590','primary'),('primo-prior','f1d8421d913f','primary'),('primo-none','88685e7dc944','secondary'),('primo-prior','4d39e6174e26','secondary')]
summary=json.loads((OLD/'analysis/comparison.json').read_text())
import pandas as pd
rows=pd.read_parquet(OLD/'analysis/verified-trials.parquet')
candidates=[]
for arm,cid,role in choices:
 row=rows[(rows.arm==arm)&(rows.config_id==cid)&(rows.n_battles==3548)&rows.inside_common_horizon].iloc[0]
 original=OLD/'searches'/arm/'trials'/Path(row.result_dir).parent.name/'config.yaml'
 cfg=yaml.safe_load(original.read_text())
 cfg['meta_eval'].update(split='test',battles_per_model=3000,panel_file=str(panel))
 cfg['run']['result_folder']=str(R/'results'/cid)
 RunConfig(**cfg)
 out=R/'configs'/f'{cid}.yaml';out.write_text(yaml.safe_dump(cfg,sort_keys=False))
 candidates.append({'arm':arm,'config_id':cid,'role':role,'model':cfg['judge']['model'],'preset':cfg['judge']['prompt_preset'],'config_file':str(out),'sha256':sha(out),'source_config':str(original),'source_sha256':sha(original),'validation_agreement':float(row.agreement),'validation_cost_per1k':float(row.cost_per_1k_battles)})
manifest={'stage':'heldout3000','approved_allocated_L40S_hours':32,'max_own_GPUs':8,'panel_file':str(panel),'panel_sha256':sha(panel),'candidates':candidates,'prices':json.loads((OLD/'configs/manifest.json').read_text())['prices'],'source_root':str(OLD/'source'),'source_provenance_sha256':sha(OLD/'provenance.json'),'selection':'Frozen before held-out inference; common28540/full3548 arm fronts','analysis':'primary prior-vs-none; all3paired contrasts;10000bootstrap seed0;prompt-cluster if repeated;no test-driven selection','paper':'https://arxiv.org/html/2501.17178v3','API_spend':0}
(R/'configs/manifest.json').write_text(json.dumps(manifest,indent=2))
print('PREPARATION_COMPLETE',json.dumps(audit),flush=True)
