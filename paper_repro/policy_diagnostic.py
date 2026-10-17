"""Diagnose action saturation on saved replay, without updating weights or replay RNG on disk."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .agent import PaperAgent
from .replay import DiskReplay
from .train import METHOD,write_json
from .evaluate import sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=128)
    a=p.parse_args()
    if a.batch<2:p.error('batch must be >=2 for BatchNorm')
    checkpoint=a.run/'latest.pt';before=sha256(checkpoint)
    replay_state_hash=sha256(a.run/'replay/state.json')
    agent,data=PaperAgent.load(checkpoint)
    if (data.get('training_metadata') or {}).get('method')!=METHOD:raise ValueError('Expected paper-derived checkpoint')
    cfg=agent.cfg
    replay=DiskReplay(a.run/'replay',cfg.replay_capacity,cfg.n_step,cfg.discount,cfg.seed,resume=True)
    batch=replay.sample(a.batch)
    with torch.no_grad():
        features=torch.cat([agent.encoder(torch.as_tensor(batch['pixels'])),
                            torch.as_tensor(batch['proprio'])],dim=-1)
        modes={}
        # Training-mode BN statistics below change only this temporary model in memory.
        for training in (False,True):
            agent.actor.train(training)
            logits=agent.actor.policy(agent.actor.trunk(features))
            output=2*torch.sigmoid(logits)-1
            sigmoid=torch.sigmoid(logits)
            modes['training_batch_statistics' if training else 'inference_running_statistics']=dict(
                mean_action=output.mean(0).tolist(),
                saturated_component_fraction=float((output.abs()>.95).float().mean()),
                max_absolute_logit=float(logits.abs().max()),
                mean_absolute_logit=float(logits.abs().mean()),
                mean_action_derivative_to_logit=float((2*sigmoid*(1-sigmoid)).mean()))
        parameters_finite=all(bool(torch.isfinite(param).all()) for net in
            (agent.encoder,agent.actor,agent.critic,agent.target) for param in net.parameters())
    assert sha256(checkpoint)==before and sha256(a.run/'replay/state.json')==replay_state_hash
    result=dict(checkpoint_training_steps=data['steps'],sampled_replay_batch=a.batch,parameters_finite=parameters_finite,
        actor_modes=modes,checkpoint_unchanged=True,replay_state_unchanged=True,optimization_updates=0,
        interpretation='Saturation diagnosis only; this comparison does not establish which hyperparameter caused it.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.output,result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
