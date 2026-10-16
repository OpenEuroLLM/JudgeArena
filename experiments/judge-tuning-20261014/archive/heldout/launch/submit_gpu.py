"""Submit only the approved held-out workers after reviewing the frozen audit."""
import json,secrets
from pathlib import Path
from slurmpilot import JobCreationInfo,SlurmPilot
R=Path('/private/tmp/judgearena-tuning-heldout-3000-20261015')
audit=json.loads((R/'analysis/panel-audit.json').read_text())
assert audit['complete'] and audit['selected_count']==3000
assert len(json.loads((R/'configs/manifest.json').read_text())['candidates'])==5
assert not (R/'launch/gpu-submissions.json').exists(),'Already submitted; do not duplicate'
records=[]
for group,minutes in [('gemma',300),('qwen',120)]:
 info=JobCreationInfo(jobname='r'+secrets.token_hex(3),entrypoint=f'worker_{group}.sh',cluster='kislurm',src_dir=str(R/'launch'),remote_dir='/work/dlclarge1/lushtake-hiwi/slurmpilot',partition='mldlc2_gpu-l40s',nodes=1,n_gpus=4,n_cpus=16,mem=None,max_runtime_minutes=minutes,sbatch_arguments='--exclude=dlc2gpu10,dlc2gpu12,lmbgpu09,lmbgpu10 --mem-per-gpu=24G')
 job=SlurmPilot().schedule_job(info)
 records.append({'group':group,'job_id':str(job),'jobname':info.jobname,'gpus':4,'max_minutes':minutes,'stage_cap_gpu_hours':32})
 (R/'launch/gpu-submissions.json').write_text(json.dumps(records,indent=2))
 print('SUBMITTED',records[-1],flush=True)
