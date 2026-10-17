"""CPU-checkable PerchRL components, not an OmniDrones task or trained reproduction.
Unpublished architecture sizes in this module are explicit implementation choices.
"""
import numpy as np
import torch
from torch import nn
from .protocols import PERCHRL_PAPER

def sample_initialization(rng,n=1):
    ranges=PERCHRL_PAPER['initialization']
    return {k:rng.uniform(low,high,n) for k,(low,high) in ranges.items()}

def reliability(loss_seconds,decay):
    if decay<=0:raise ValueError('Positive reliability decay required')
    return np.exp(-decay*np.maximum(loss_seconds,0.))

def active_reward(estimated_position,true_position,loss_seconds,camera_axis,target_direction,
                  alpha,beta,eta,tau):
    if tau<=0:raise ValueError('Positive loss time scale required')
    gate=1.+alpha*np.clip(loss_seconds/tau,0.,1.)
    error=np.linalg.norm(np.asarray(estimated_position)-np.asarray(true_position),axis=-1)
    axis=np.asarray(camera_axis);direction=np.asarray(target_direction)
    if not np.allclose(np.linalg.norm(axis,axis=-1),1.) or not np.allclose(np.linalg.norm(direction,axis=-1),1.):
        raise ValueError('Camera axis and target direction must be normalized')
    estimation=-beta*gate*error
    perception=-eta*(1.-np.sum(axis*direction,axis=-1))
    return estimation+perception

class CausalTCN(nn.Module):
    def __init__(self,channels=64,kernel_size=3):
        super().__init__();self.kernel_size=kernel_size
        self.layers=nn.ModuleList([nn.Conv1d(10,channels,kernel_size),
            nn.Conv1d(channels,channels,kernel_size,dilation=2),
            nn.Conv1d(channels,channels,kernel_size,dilation=4)])
    def forward(self,history):
        x=history.transpose(1,2)
        for layer in self.layers:
            x=torch.relu(layer(nn.functional.pad(x,((self.kernel_size-1)*layer.dilation[0],0))))
        return x[:,:,-1]

class PerchPolicy(nn.Module):
    def __init__(self,history_length,channels=64,hidden=128,temporal=True):
        super().__init__();self.history_length=history_length;self.temporal=temporal
        self.encoder=CausalTCN(channels) if temporal else None
        feature=channels if temporal else 10
        self.actor=nn.Sequential(nn.Linear(feature+12+1+4,hidden),nn.ELU(),nn.Linear(hidden,4),nn.Tanh())
        self.critic=nn.Sequential(nn.Linear(feature+12+1+4+15*10,hidden),nn.ELU(),nn.Linear(hidden,1))
    def inputs(self,history,own,reliability,previous_action):
        if history.shape[1:]!=(self.history_length,10):raise ValueError('Expected H x 10 platform history')
        feature=self.encoder(history) if self.temporal else history[:,-1]
        return torch.cat([feature,own,reliability,previous_action],-1)
    def action(self,history,own,reliability,previous_action):
        # Actor has no argument or access path for future observations.
        return self.actor(self.inputs(history,own,reliability,previous_action))
    def value(self,history,own,reliability,previous_action,future):
        if future.shape[1:]!=(15,10):raise ValueError('Critic requires 15 future observations')
        return self.critic(torch.cat([self.inputs(history,own,reliability,previous_action),future.flatten(1)],-1))
