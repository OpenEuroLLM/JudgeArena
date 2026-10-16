"""Evaluate only frozen held-out candidates; reuse one backend per model."""
import argparse,json,time,hashlib
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from judgearena.benchmarks.execution import build_judge
from judgearena.benchmarks.meta_eval.runner import run_meta_eval
from judgearena.config import load_config,dump_config
from judgearena.tasks.registry import get_packaged_task
from judgearena.tuning.runner import _read_trial
from judgearena.pricing import TokenPrice
from judgearena.usage import track_usage
R=Path('/work/dlclarge1/lushtake-hiwi/judge-tuning-heldout-3000-20261015')
def main():
    p=argparse.ArgumentParser();p.add_argument('group',choices=['gemma','qwen']);a=p.parse_args()
    m=json.loads((R/'configs/manifest.json').read_text())
    assert hashlib.sha256(Path(m['panel_file']).read_bytes()).hexdigest()==m['panel_sha256']
    rows=[c for c in m['candidates'] if ('gemma' in c['model'])==(a.group=='gemma')]
    first=load_config(rows[0]['config_file']);prepared=build_judge(first)
    task=deepcopy(get_packaged_task(first.task))
    scoring=task.spec.protocol.scoring
    metric=next(v for v in scoring.metrics if v.metric=='meta_eval_agreement')
    metric.parameters['n_bootstraps']=0
    # Joint predeclared paired uncertainty is computed once, after all five finish.
    task=replace(task,spec=task.spec.model_copy(update={'protocol':task.spec.protocol.model_copy(update={'scoring':scoring.model_copy(update={'metrics':(metric,)})})}))
    progress_file=R/'analysis'/f'{a.group}-progress.json'
    previous={r['config_id']:r for r in json.loads(progress_file.read_text())} if progress_file.exists() else {}
    progress=[]
    for c in rows:
     path=Path(c['config_file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256']
     cfg=load_config(path);folder=Path(cfg.run.result_folder);folder.mkdir(parents=True,exist_ok=True)
     dump_config(cfg,folder/'config.yaml')
     if c['config_id'] in previous:
      assert len(list(folder.glob('*/results.json')))==1
      progress.append(previous[c['config_id']]);continue
     start=time.monotonic()
     with track_usage() as tracker:
      if not list(folder.glob('*/results.json')):
       run_meta_eval(cfg,task,judge_chat_model=prepared)
     result=_read_trial(folder,TokenPrice(**m['prices'][c['model']],source='frozen_intermediate_reference'))
     result.update(config_id=c['config_id'],arm=c['arm'],role=c['role'],elapsed_s=time.monotonic()-start,fresh_usage=tracker.snapshot().to_dict())
     assert result['n_battles']==3000
     progress.append(result)
     progress_file.write_text(json.dumps(progress,indent=2,default=str))
     print('CANDIDATE_COMPLETE',json.dumps(result,default=str),flush=True)
    print('GROUP_COMPLETE',a.group,flush=True)


if __name__ == '__main__':
    main()
