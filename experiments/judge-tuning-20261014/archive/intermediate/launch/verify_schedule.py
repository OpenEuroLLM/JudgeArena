"""Require a real BO phase inside the matched panel-work horizon."""
import json
from pathlib import Path
root=Path('/work/dlclarge1/lushtake-hiwi/judge-tuning-intermediate-20261014')
runs=json.loads((root/'analysis/schedules.json').read_text())
assert len(runs)==3
horizon=next(r['actual_config_battles'] for r in runs if r['arm']=='uniform')
rows=[]
for run in runs:
 total=0;prefix=0
 for trial in run['trials']:
  if total+trial['n_battles']>horizon:break
  total+=trial['n_battles'];prefix+=1
 inside=sum(p['trial_index']<prefix for p in run['bo_proposals'])
 if run['arm']!='uniform':assert inside>=5,(run['arm'],inside)
 rows.append({'arm':run['arm'],'horizon':horizon,'native_work':run['actual_config_battles'],'completed_prefix_work':total,'bo_inside_horizon':inside,'gp_with_priors':sum(x['prior_nonnull'] for x in run['gp_acquisitions'])})
(root/'analysis/schedule-gate.json').write_text(json.dumps(rows,indent=2))
print('SCHEDULE_GATE_PASSED',json.dumps(rows),flush=True)
