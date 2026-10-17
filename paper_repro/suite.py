"""Launch repeatable seed/variant jobs; official missing baselines are not fabricated."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true')
 p.add_argument('--gui',action='store_true');a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
 jobs=[dict(seed=seed,variant=variant,output=str(a.root/f'{variant}_seed{seed}'))
       for variant in ('baseline','no_los','constant_descent') for seed in range(1,6)]
 (a.root/'planned_jobs.json').write_text(json.dumps(dict(jobs=jobs,
  missing=['DTD/real pad-photo randomized variant','Vision SAC','FF-IBVS with/without estimator noise',
           'DDPGLN with/without estimator noise','PerchRL original simulator and baselines'],
  numerical_reproduction_complete=False),indent=2))
 for job in jobs:
  cmd=[sys.executable,'-m','paper_repro.train','--output',job['output'],'--seed',str(job['seed']),
       '--variant',job['variant'],'--reward-interpretation','normalized','--vertical-reward','actual_guard']
  if a.gui:cmd.append('--gui')
  if (Path(job['output'])/'latest.pt').exists():cmd.append('--resume')
  print(' '.join(cmd),flush=True)
  if a.run:
   result=subprocess.run(cmd)
   if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
