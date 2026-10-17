"""Buffered file replay: published capacity without giant memory mappings."""
import json,os
from pathlib import Path
import numpy as np
from landing_repro.agent import Replay

class DiskArray:
    def __init__(self,path,capacity=None,example=None):
        self.path=Path(path)
        if example is not None:
            self.shape=(capacity,*example.shape);self.dtype=example.dtype
            with self.path.open('wb') as f:
                np.lib.format.write_array_header_1_0(f,dict(descr=self.dtype.str,fortran_order=False,shape=self.shape))
                self.offset=f.tell();f.truncate(self.offset+int(np.prod(self.shape))*self.dtype.itemsize)
        else:
            with self.path.open('rb') as f:
                version=np.lib.format.read_magic(f)
                self.shape,fortran,self.dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0)
                                            else np.lib.format.read_array_header_2_0(f))
                if fortran:raise ValueError('Replay must use row-major records')
                self.offset=f.tell()
        self.fd=os.open(self.path,os.O_RDWR);self.row_shape=self.shape[1:]
        self.row_bytes=int(np.prod(self.row_shape))*self.dtype.itemsize
    def __setitem__(self,index,value):
        raw=np.asarray(value,dtype=self.dtype).reshape(self.row_shape).tobytes()
        n=os.pwrite(self.fd,raw,self.offset+int(index)*self.row_bytes)
        if n!=len(raw):raise IOError('Short replay write')
    def __getitem__(self,index):
        if np.isscalar(index):
            raw=os.pread(self.fd,self.row_bytes,self.offset+int(index)*self.row_bytes)
            if len(raw)!=self.row_bytes:raise IOError('Short replay read')
            return np.frombuffer(raw,dtype=self.dtype).reshape(self.row_shape).copy()
        return np.stack([self[int(i)] for i in index])
    def __array__(self,dtype=None):
        # Used only by small equivalence tests; production samples individual rows.
        raw=os.pread(self.fd,int(np.prod(self.shape))*self.dtype.itemsize,self.offset)
        return np.frombuffer(raw,dtype=self.dtype).reshape(self.shape).astype(dtype or self.dtype,copy=False)
    def flush(self):os.fsync(self.fd)
    def __del__(self):
        try:os.close(self.fd)
        except (AttributeError,OSError):pass

class DiskReplay(Replay):
    def __init__(self,path,capacity,n_step=3,discount=.99,seed=1,resume=False):
        super().__init__(capacity,n_step,discount,seed)
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        if resume:
            state=json.loads((self.path/'state.json').read_text())
            if state['capacity']!=capacity:raise ValueError('Replay capacity changed')
            self.data={k:DiskArray(self.path/f'{k}.npy') for k in state['keys']} or None
            self.index,self.size=state['index'],state['size'];self.rng.bit_generator.state=state['rng']
    def _commit(self):
        if self.data is None:
            obs,action,_,next_obs,*_=self.queue[0]
            row={'pixels':obs['pixels'],'proprio':obs['proprio'],
                 'next_pixels':next_obs['pixels'],'next_proprio':next_obs['proprio'],
                 'action':np.asarray(action,np.float32),'reward':np.zeros(1,np.float32),
                 'discount':np.zeros(1,np.float32)}
            self.data={k:DiskArray(self.path/f'{k}.npy',self.capacity,v) for k,v in row.items()}
        super()._commit()
    def flush(self):
        if self.data is not None:
            for arr in self.data.values():arr.flush()
        state=dict(capacity=self.capacity,index=self.index,size=self.size,keys=list(self.data or {}),
                   rng=self.rng.bit_generator.state,pending_queue_discarded_on_resume=len(self.queue))
        tmp=self.path/'state.json.tmp';tmp.write_text(json.dumps(state));tmp.replace(self.path/'state.json')
