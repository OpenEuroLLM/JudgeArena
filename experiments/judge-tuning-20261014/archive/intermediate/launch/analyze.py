"""Verify frozen panels and summarize completed searches at a common work horizon."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

R=Path('/private/tmp/judgearena-tuning-intermediate-20261014')
PAN=json.loads((R/'panels/panels.json').read_text())['validation']
SIZES={20:400,60:1200,180:3548}
HORIZON=28540
summary={}
allrows=[]
for arm in ['uniform','primo-none','primo-prior']:
    root=R/'searches'/arm
    trials=pd.read_parquet(root/'trials.parquet')
    obs=json.loads((R/f'analysis/{arm}-optimizer-observations.json').read_text())
    work=0;eligible=set();prefix=[]
    for attempt in obs['execution_attempts']:
        work+=SIZES[attempt['fidelity']]
        if work<=HORIZON and attempt['status']=='finished':
            eligible.add(Path(attempt['config_path']).parent.name.removeprefix('config_'))
        prefix.append(dict(attempt,cumulative_battles=work))
    verified=[]
    for _,row in trials.iterrows():
        remote=Path(row.result_dir);folder=root/'trials'/remote.parent.name/remote.name
        ann=pd.read_parquet(folder/'annotations.parquet')
        cfg=yaml.safe_load((folder.parent/'config.yaml').read_text());j=cfg['judge'];n=SIZES[int(row.battles_per_model)]
        ids=set(ann.battle_id);assert ids==set(PAN[str(int(row.battles_per_model))]) and len(ids)==n
        assert len(ann)==n*(2 if j['swap_mode']=='both' else 1)
        assert j['temperature']==0 and j['max_out_tokens']==8192 and j['max_model_len']==65536 and j['seed']==42
        assert j['engine_kwargs']['chat_template_kwargs']['enable_thinking'] and j['engine_kwargs']['thinking_token_budget'] is None
        assert cfg['generation']['truncate_judge_input_chars']==8192
        usage=ann.usage_json.fillna('{}').map(json.loads)
        assert all((x or {}).get('output_tokens',0)<=8192 for x in usage)
        record=row.to_dict();record['inside_common_horizon']=row.neps_trial_id in eligible
        record['input_tokens']=sum((x or {}).get('input_tokens',0) or 0 for x in usage)
        record['output_tokens']=sum((x or {}).get('output_tokens',0) or 0 for x in usage)
        record['cap_requests']=sum((x or {}).get('output_tokens',0)>=8192 for x in usage)
        verified.append(record)
    frame=pd.DataFrame(verified);full=frame[(frame.battles_per_model==180)&frame.inside_common_horizon].drop_duplicates('config_id')
    pareto=[]
    for idx,row in full.iterrows():
        dominates=(full.agreement>=row.agreement)&(full.cost_per_1k_battles<=row.cost_per_1k_battles)&((full.agreement>row.agreement)|(full.cost_per_1k_battles<row.cost_per_1k_battles))
        if not dominates.any():pareto.append(idx)
    front=full.loc[pareto]
    summary[arm]={'trial_evaluations':len(trials),'execution_attempts':len(obs['execution_attempts']),'attempted_config_battles':work,'common_horizon':HORIZON,'full_fidelity_inside_horizon':len(full),'models_inside_horizon':sorted(full.judge_model.unique()),'bo_proposals':len(obs['bo_proposals']),'gp_prior_calls':sum(x['prior_nonnull'] for x in obs['gp_acquisitions']),'pareto':front[['config_id','judge_model','agreement','cost_per_1k_battles','overrides']].to_dict('records'),'attempt_trace':prefix}
    frame['arm']=arm;allrows.append(frame)
    front.to_parquet(R/f'analysis/{arm}-common-pareto.parquet',index=False)
pd.concat(allrows).to_parquet(R/'analysis/verified-trials.parquet',index=False)
(R/'analysis/comparison.json').write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps({k:{a:b for a,b in v.items() if a not in ['attempt_trace','pareto']} for k,v in summary.items()},indent=2))
# Independently recompute metrics and expected costs, including cached token usage.
prices=json.loads((R/'configs/manifest.json').read_text())['prices']
metric_checks=[];best_correct={};rng=np.random.default_rng(0)
for arm in summary:
    trials=pd.read_parquet(R/f'searches/{arm}/trials.parquet')
    selected=[]
    for _,row in trials.iterrows():
        rp=Path(row.result_dir);folder=R/'searches'/arm/'trials'/rp.parent.name/rp.name
        b=pd.read_parquet(folder/'battles.parquet');b=b[b.sampled].set_index('battle_id')
        pred=np.where(b.pref<.49,0,np.where(b.pref>.51,1,.5))
        correct=pd.Series((pred==b.reference_pref.to_numpy())&b.pref.notna().to_numpy(),index=b.index)
        assert abs(correct.mean()-row.agreement)<1e-12
        assert abs(b.pref.notna().mean()-row.coverage)<1e-12
        ann=pd.read_parquet(folder/'annotations.parquet');us=ann.usage_json.fillna('{}').map(json.loads)
        price=prices[row.judge_model]
        expected=sum(((u or {}).get('input_tokens',0) or 0)*price['input']+((u or {}).get('output_tokens',0) or 0)*price['output'] for u in us)/1e6*1000/len(b)
        assert abs(expected-row.cost_per_1k_battles)<1e-9,(expected,row.cost_per_1k_battles)
        metric_checks.append({'arm':arm,'trial':row.neps_trial_id,'metrics_recomputed':True})
        if row.battles_per_model==180 and row.neps_trial_id in {Path(x['config_path']).parent.name.removeprefix('config_') for x in summary[arm]['attempt_trace'] if x['cumulative_battles']<=HORIZON and x['status']=='finished'}:
            selected.append((row.agreement,correct,row.to_dict()))
    _,best_correct[arm],best=max(selected,key=lambda x:x[0]);summary[arm]['best_agreement_config']=best
pairs=[]
for a,b in [('primo-prior','primo-none'),('primo-prior','uniform'),('primo-none','uniform')]:
    x,y=best_correct[a].align(best_correct[b]);assert len(x)==3548 and not x.isna().any()
    delta=x.astype(float).to_numpy()-y.astype(float).to_numpy()
    boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
    pairs.append({'a':a,'b':b,'delta_agreement':delta.mean(),'paired_battle_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'n':len(delta),'scope':'adaptive validation-selected configurations; not held-out or optimizer-seed uncertainty'})
(R/'analysis/paired-validation.json').write_text(json.dumps(pairs,indent=2))
(R/'analysis/metric-verification.json').write_text(json.dumps(metric_checks,indent=2))
(R/'analysis/comparison.json').write_text(json.dumps(summary,indent=2,default=str))
print('Recomputed all agreement/coverage/native-token expected costs:',len(metric_checks));print(json.dumps(pairs,indent=2))
