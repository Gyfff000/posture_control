"""Record a small, portable handoff manifest; no model/replay data are uploaded."""
import argparse,hashlib,importlib.metadata,json,platform,sys
from pathlib import Path
import torch
from .train import write_json,METHOD

def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):sha.update(block)
    return sha.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parent.parent
    versions={}
    for name in ('torch','numpy','scipy','pybullet','gymnasium','gym-pybullet-drones',
                 'opencv-python','Pillow','hydra-core','imageio','setuptools'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    result=dict(python=platform.python_version(),platform=platform.platform(),packages=versions,
        cuda_available=torch.cuda.is_available(),torch_cuda_build=torch.version.cuda,
        cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        source_sha256={str(path.relative_to(root)):digest(path)
            for folder in ('paper_repro','landing_repro') for path in sorted((root/folder).glob('*.py'))},
        gpu_validation='not tested on this CPU-only machine',
        reproduction_complete=False)
    if a.run:
        checkpoint=a.run/'latest.pt';data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        metadata=data.get('training_metadata') or {}
        if metadata.get('method')!=METHOD:raise ValueError('Expected paper-derived checkpoint')
        replay=json.loads((a.run/'replay/state.json').read_text())
        summary=json.loads((a.run/'summary.json').read_text())
        if summary.get('state')!='stopped_and_saved':raise ValueError('Stop and save before creating a handoff manifest')
        if data['steps']!=summary['step'] or data['updates']!=summary['updates']:
            raise ValueError('Checkpoint does not match saved summary')
        if replay['size']!=summary['replay_size']:raise ValueError('Replay size differs from summary')
        result['saved_run']=dict(path=str(a.run),checkpoint_sha256=digest(checkpoint),
            steps=data['steps'],updates=data['updates'],episodes=metadata['episodes'],
            training_successes=metadata['successes'],replay_size=replay['size'],
            replay_capacity=replay['capacity'],pending_tail_discarded=replay['pending_queue_discarded_on_resume'],
            stopped_summary_matches_checkpoint=True,
            config_sha256=digest(a.run/'config.json'),
            transfer_requirement='Copy entire run including replay/*.npy and replay/state.json for resume; latest.pt alone is evaluation only.')
    a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,result)
    print(json.dumps({key:value for key,value in result.items() if key!='source_sha256'}),flush=True)
if __name__=='__main__':main()
