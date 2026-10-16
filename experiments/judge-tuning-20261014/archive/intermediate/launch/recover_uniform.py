from pathlib import Path
import json
from neps.state import NePSState
root=Path("/work/dlclarge1/lushtake-hiwi/judge-tuning-intermediate-20261014")
state=NePSState.create_or_load(root/"searches/uniform/neps",load_only=True)
trials=state.lock_and_read_trials()
active=[t for t in trials.values() if t.metadata.state=="evaluating"]
assert len(active)<=1
for t in active:
    print("Reset confirmed-dead job29997142 trial",t.id,flush=True)
    state.lock_and_reset_trial_by_id(t.id)
(root/"analysis/uniform-recovery-29997142.json").write_text(json.dumps({"dead_job":"29997142","reset_trials":[t.id for t in active]}))
