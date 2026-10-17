import json,os,subprocess,tempfile
from pathlib import Path
import numpy as np,torch
from paper_repro.agent import PaperAgent
from paper_repro.protocols import ladosz_configs
from paper_repro.train import write_json

def main():
    root=Path.cwd();cfg,acfg=ladosz_configs()
    automatic=PaperAgent(acfg,device="auto")
    assert str(automatic.device)=="cpu"
    try:PaperAgent(acfg,device="cuda")
    except RuntimeError as exc:cuda_error=str(exc)
    else:raise AssertionError("Explicit CUDA must fail rather than silently use CPU")
    del automatic
    with tempfile.TemporaryDirectory(prefix="paper_handoff_check_") as temp:
        run=Path(temp)/"run"
        base=[os.sys.executable,"-m","paper_repro.train","--device","cpu","--output",str(run),
              "--seed","1","--eval-every-episodes","0","--final-eval-episodes","0"]
        first=subprocess.run(base+["--max-steps","3"],capture_output=True,text=True)
        if first.returncode:raise RuntimeError(first.stdout+first.stderr)
        model,data=PaperAgent.load(run/"latest.pt")
        assert data["steps"]==3 and data["updates"]==0 and data["training_metadata"]["training_device"]=="cpu"
        second=subprocess.run(base+["--max-steps","5","--resume"],capture_output=True,text=True)
        if second.returncode:raise RuntimeError(second.stdout+second.stderr)
        _,data=PaperAgent.load(run/"latest.pt",device="auto")
        assert data["steps"]==5
        state=json.loads((run/"replay/state.json").read_text())
        assert state["size"]==1
    out=dict(status="passed",auto_device="cpu",explicit_cuda_unavailable_error=cuda_error,
             temporary_headless_train_steps=3,resumed_total_steps=5,
             saved_device_metadata=True,main_training_run_unchanged=True,
             gpu_execution_tested=False)
    write_json(root/"results/paper_audit_20261017/device_and_resume_check.json",out)
    print(json.dumps(out))

if __name__=='__main__':main()
