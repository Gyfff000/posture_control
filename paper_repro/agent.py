"""Paper Fig.3 BatchNorm architecture using the existing DrQ-v2 update equations.
No expert actions, behavior cloning, smoothness losses or guidance are used.
Unknown update frequency / exploration schedule inherit the documented DrQ-v2 adapter.
"""
import numpy as np
import torch
from torch import nn
from landing_repro.agent import DrQV2,Encoder,head,initialize
from landing_repro.config import AgentConfig

class CheckpointEncoder(Encoder):
    def forward(self,x):
        if len(x)<=32:return super().forward(x)
        from torch.utils.checkpoint import checkpoint
        outputs=[]
        for chunk in x.split(32):
            if torch.is_grad_enabled():
                outputs.append(checkpoint(lambda pixels:Encoder.forward(self,pixels),chunk,use_reentrant=False))
            else:outputs.append(Encoder.forward(self,chunk))
        return torch.cat(outputs,0)

class PaperActor(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(39200+21,50),nn.BatchNorm1d(50),nn.Tanh())
        self.policy=head(50,512,3);self.apply(initialize)
    def forward(self,x):
        # Eq.6 maps [0,1] to velocity; expose its equivalent normalized [-1,1] command.
        return 2.*torch.sigmoid(self.policy(self.trunk(x)))-1.

class PaperCritic(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(39200+21,50),nn.BatchNorm1d(50),nn.Tanh())
        self.q1=head(53,512,1);self.q2=head(53,512,1);self.apply(initialize)
    def forward(self,x,action):
        x=torch.cat([self.trunk(x),action],-1)
        return self.q1(x),self.q2(x)

class PaperAgent(DrQV2):
    def __init__(self,cfg=None):
        self.cfg=cfg or AgentConfig()
        if self.cfg.proprio_dim!=21:raise ValueError('Paper agent requires 21D own state')
        torch.set_num_threads(self.cfg.torch_threads);torch.manual_seed(self.cfg.seed)
        self.encoder=CheckpointEncoder();self.actor=PaperActor(self.cfg);self.critic=PaperCritic(self.cfg)
        self.target=PaperCritic(self.cfg);self.target.load_state_dict(self.critic.state_dict())
        self.target.requires_grad_(False);self.target.eval()
        self.optimizers=[torch.optim.Adam(net.parameters(),lr=self.cfg.learning_rate)
                         for net in (self.encoder,self.actor,self.critic)]
        self.updates=0
    @torch.no_grad()
    def act(self,obs,step=0,evaluation=False):
        training=self.actor.training
        self.actor.eval()
        try:return super().act(obs,step,evaluation)
        finally:self.actor.train(training)
    def update(self,batch,step,**kwargs):
        if kwargs:raise ValueError('Paper updates accept no imitation/smoothness options')
        result=super().update(batch,step)
        # Target BN statistics are state, not optimized parameters.
        with torch.no_grad():
            for actual,target in zip(self.critic.buffers(),self.target.buffers()):
                if target.is_floating_point():target.lerp_(actual,self.cfg.tau)
                else:target.copy_(actual)
        return result
