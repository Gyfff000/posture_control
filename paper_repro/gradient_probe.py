"""Bounded, synthetic gradient/memory probe; not a landing experiment."""
import argparse,json,resource,time
import numpy as np
import torch
from .protocols import ladosz_configs
from .agent import PaperAgent

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=1024);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    a=p.parse_args()
    # CUDA reserves large virtual address ranges; a CPU address-space cap is inappropriate there.
    if a.device=='cpu' or (a.device=='auto' and not torch.cuda.is_available()):
        resource.setrlimit(resource.RLIMIT_AS,(7*1024**3,7*1024**3))
    env,cfg=ladosz_configs();cfg.batch_size=a.batch
    started=time.monotonic()
    result={'purpose':'synthetic gradient and memory feasibility only; no landing performance','batch':a.batch}
    try:
        agent=PaperAgent(cfg,device=a.device);rng=np.random.default_rng(19)
        batch={'pixels':rng.integers(0,256,(a.batch,3,84,84),dtype=np.uint8),
               'next_pixels':rng.integers(0,256,(a.batch,3,84,84),dtype=np.uint8),
               'proprio':rng.random((a.batch,21),dtype=np.float32),
               'next_proprio':rng.random((a.batch,21),dtype=np.float32),
               'action':rng.uniform(-1,1,(a.batch,3)).astype(np.float32),
               'reward':np.ones((a.batch,1),np.float32)*.1,'discount':np.ones((a.batch,1),np.float32)*.99}
        metrics=agent.update(batch,10000)
        result.update(device=str(agent.device),cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if agent.device.type=='cuda' else None,status='passed',metrics=metrics,seconds=time.monotonic()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    except Exception as exc:
        result.update(status='failed',error=repr(exc),seconds=time.monotonic()-started)
    from pathlib import Path
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if result['status']!='passed':raise SystemExit(2)
if __name__=='__main__':main()