"""Disk-backed replay preserves Table I capacity without allocating >12 GB RAM."""
import json
from pathlib import Path
from collections import deque
import numpy as np
from landing_repro.agent import Replay

class DiskReplay(Replay):
    def __init__(self,path,capacity,n_step=3,discount=.99,seed=1,resume=False):
        super().__init__(capacity,n_step,discount,seed)
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        if resume:
            state=json.loads((self.path/'state.json').read_text())
            if state['capacity']!=capacity:raise ValueError('Replay capacity changed')
            self.data={k:np.load(self.path/f'{k}.npy',mmap_mode='r+') for k in state['keys']} or None
            self.index,self.size=state['index'],state['size'];self.rng.bit_generator.state=state['rng']
    def _commit(self):
        if self.data is None:
            obs,action,_,next_obs,*_=self.queue[0]
            row={'pixels':obs['pixels'],'proprio':obs['proprio'],
                 'next_pixels':next_obs['pixels'],'next_proprio':next_obs['proprio'],
                 'action':np.asarray(action,np.float32),'reward':np.zeros(1,np.float32),
                 'discount':np.zeros(1,np.float32)}
            self.data={k:np.lib.format.open_memmap(self.path/f'{k}.npy',mode='w+',dtype=v.dtype,
                        shape=(self.capacity,*v.shape)) for k,v in row.items()}
        super()._commit()
    def flush(self):
        if self.data is None:
            state=dict(capacity=self.capacity,index=0,size=0,keys=[],rng=self.rng.bit_generator.state,pending_queue_discarded_on_resume=0)
            (self.path/'state.json').write_text(json.dumps(state));return
        for arr in self.data.values():arr.flush()
        state=dict(capacity=self.capacity,index=self.index,size=self.size,keys=list(self.data),
                   rng=self.rng.bit_generator.state,pending_queue_discarded_on_resume=len(self.queue))
        tmp=self.path/'state.json.tmp';tmp.write_text(json.dumps(state));tmp.replace(self.path/'state.json')
