from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,importlib.util,platform,shutil,torch
out=Path('/mnt/d/paper_reproductions/posture_control/results/paper_audit_20261017')
raw=Path('/mnt/d/paper_reproductions/posture_control/.paper_sources_20261017');raw.mkdir(exist_ok=True)
urls={
 'perchrl_v3':'https://arxiv.org/html/2606.03441v3',
 'author_repos':'https://api.github.com/users/lzh2001-debug/repos?per_page=100',
 'omnidrones_commit':'https://api.github.com/repos/btx0424/OmniDrones/commits/main',
 'omnidrones_install':'https://omnidrones.readthedocs.io/en/latest/installation.html',
 'drqv2_reference':'https://raw.githubusercontent.com/facebookresearch/drqv2/c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef/drqv2.py'}
def fetch(item):
 name,url=item
 try:
  with urlopen(Request(url,headers={'User-Agent':'paper-reproduction-audit/1.0'}),timeout=20) as r:data=r.read(5_000_000);status=r.status
  (raw/(name+'.txt')).write_bytes(data)
  row=dict(name=name,url=url,status=status,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
  if name=='author_repos':row['matching_repo_names']=[x['full_name'] for x in json.loads(data) if any(s in x['name'].lower() for s in ('perch','landing'))]
  if name=='omnidrones_commit':row['sha']=json.loads(data)['sha']
  return row
 except Exception as exc:return dict(name=name,url=url,error=repr(exc))
with ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(fetch,urls.items()))
report=dict(checked_date='2026-10-17',python=platform.python_version(),torch=torch.__version__,
 cuda_available=torch.cuda.is_available(),cuda_device_count=torch.cuda.device_count(),
 isaacsim_installed=importlib.util.find_spec('isaacsim') is not None,
 omnidrones_installed=importlib.util.find_spec('omni_drones') is not None,
 disk_free_gib=shutil.disk_usage('/mnt/d').free/1024**3,sources=results,
 exact_perchrl_training_ready=False,
 blockers=['No NVIDIA GPU; installed torch is CPU only','Isaac Sim and OmniDrones not installed',
 'Task-specific code and incomplete hyperparameters still need verification'],
 claims='No full or numerical reproduction has been completed')
(out/'preflight_and_sources.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
