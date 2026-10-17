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
    def __init__(self,cfg=None,device='cpu'):
        requested=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        self.device=torch.device(requested)
        if self.device.type not in ('cpu','cuda'):raise ValueError('Supported devices: cpu, cuda, auto')
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; install a CUDA PyTorch build on an NVIDIA GPU machine')
        self.cfg=cfg or AgentConfig()
        if self.cfg.proprio_dim!=21:raise ValueError('Paper agent requires 21D own state')
        torch.set_num_threads(self.cfg.torch_threads);torch.manual_seed(self.cfg.seed)
        self.encoder=CheckpointEncoder();self.actor=PaperActor(self.cfg);self.critic=PaperCritic(self.cfg)
        self.target=PaperCritic(self.cfg);self.target.load_state_dict(self.critic.state_dict())
        self.target.requires_grad_(False);self.target.eval()
        for net in (self.encoder,self.actor,self.critic,self.target):net.to(self.device)
        self.optimizers=[torch.optim.Adam(net.parameters(),lr=self.cfg.learning_rate)
                         for net in (self.encoder,self.actor,self.critic)]
        self.updates=0
    @torch.no_grad()
    def act(self,obs,step=0,evaluation=False):
        training=self.actor.training
        self.actor.eval()
        try:
            pixels=torch.as_tensor(obs['pixels'],device=self.device).unsqueeze(0)
            own=torch.as_tensor(obs['proprio'],device=self.device).unsqueeze(0)
            mean=self.actor(torch.cat([self.encoder(pixels),own],dim=-1))
            action=mean if evaluation else self.sample(mean,self.stddev(step))
            return action[0].cpu().numpy()
        finally:self.actor.train(training)
    def update(self,batch,step,**kwargs):
        if kwargs:raise ValueError('Paper updates accept no imitation/smoothness options')
        batch={key:torch.as_tensor(value,device=self.device) for key,value in batch.items()}
        result=super().update(batch,step)
        # Target BN statistics are state, not optimized parameters.
        with torch.no_grad():
            for actual,target in zip(self.critic.buffers(),self.target.buffers()):
                if target.is_floating_point():target.lerp_(actual,self.cfg.tau)
                else:target.copy_(actual)
        return result
    @classmethod
    def load(cls,path,device='cpu'):
        data=torch.load(path,map_location='cpu',weights_only=False)
        agent=cls(AgentConfig(**data['agent_config']),device=device)
        for key in ('encoder','actor','critic','target'):
            getattr(agent,key).load_state_dict(data[key])
        # PyTorch loads floating optimizer state onto each parameter's device.
        for optimizer,state in zip(agent.optimizers,data['optimizers']):optimizer.load_state_dict(state)
        agent.updates=data['updates']
        return agent,data
