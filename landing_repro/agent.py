"""DrQ-v2 adaptation with stacked own-vehicle state.

Algorithm reference: facebookresearch/drqv2 commit c0c650b7 (MIT).
Preserves random shifts, twin Q, truncated exploration, n-step targets and
critic target averaging. See NOTICE.md for attribution and deviations.
"""
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import AgentConfig


def initialize(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu") if isinstance(module, nn.Conv2d) else 1)
        nn.init.zeros_(module.bias)


class Encoder(nn.Module):
    output_dim = 39200

    def __init__(self):
        super().__init__()
        layers = []
        for i in range(4):
            layers.extend([nn.Conv2d(3 if i == 0 else 32, 32, 3, stride=2 if i == 0 else 1), nn.ReLU()])
        self.conv = nn.Sequential(*layers)
        self.apply(initialize)

    def forward(self, pixels):
        return self.conv(pixels.float() / 255. - .5).flatten(1)


def trunk(size, feature):
    return nn.Sequential(nn.Linear(size, feature), nn.LayerNorm(feature), nn.Tanh())


def head(size, hidden, output):
    return nn.Sequential(nn.Linear(size, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                         nn.ReLU(), nn.Linear(hidden, output))


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.trunk = trunk(Encoder.output_dim + 21, cfg.feature_dim)
        self.policy = head(cfg.feature_dim, cfg.hidden_dim, 3)
        self.apply(initialize)

    def forward(self, features):
        return self.policy(self.trunk(features)).tanh()


class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.trunk = trunk(Encoder.output_dim + 21, cfg.feature_dim)
        self.q1 = head(cfg.feature_dim + 3, cfg.hidden_dim, 1)
        self.q2 = head(cfg.feature_dim + 3, cfg.hidden_dim, 1)
        self.apply(initialize)

    def forward(self, features, action):
        x = torch.cat([self.trunk(features), action], dim=-1)
        return self.q1(x), self.q2(x)


class DrQV2:
    def __init__(self, cfg=None):
        self.cfg = cfg or AgentConfig()
        torch.set_num_threads(self.cfg.torch_threads)
        torch.manual_seed(self.cfg.seed)
        self.encoder, self.actor, self.critic = Encoder(), Actor(self.cfg), Critic(self.cfg)
        self.target = Critic(self.cfg)
        self.target.load_state_dict(self.critic.state_dict())
        self.target.requires_grad_(False)
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=self.cfg.learning_rate)
                           for net in (self.encoder, self.actor, self.critic)]
        self.updates = 0

    def stddev(self, step):
        c = self.cfg
        return c.noise_start + (c.noise_end - c.noise_start) * min(step / c.noise_decay_steps, 1.)

    def sample(self, mean, std, clip=None):
        noise = torch.randn_like(mean) * std
        if clip is not None:
            noise = noise.clamp(-clip, clip)
        raw = mean + noise
        bounded = raw.clamp(-1. + 1e-6, 1. - 1e-6)
        return raw + (bounded - raw).detach()

    def augment(self, x):
        x = x.float()
        n, c, h, w = x.shape
        x = F.pad(x, (4, 4, 4, 4), mode=self.cfg.augmentation_padding)
        # Integer shifts match the support of the official random-shift operation.
        shifts = torch.randint(0, 9, (n, 2))
        return torch.stack([x[i, :, sy:sy + h, sx:sx + w] for i, (sy, sx) in enumerate(shifts.tolist())])

    @torch.no_grad()
    def act(self, obs, step=0, evaluation=False):
        feat = self.encoder(torch.from_numpy(obs["pixels"]).unsqueeze(0))
        feat = torch.cat([feat, torch.from_numpy(obs["proprio"]).unsqueeze(0)], dim=-1)
        mean = self.actor(feat)
        action = mean if evaluation else self.sample(mean, self.stddev(step))
        return action[0].numpy()

    def update(self, batch, step, expert=None, bc_weight=0., rl_weight=1., gradient_clip=0.):
        b = {key: torch.as_tensor(value) for key, value in batch.items()}
        feat = torch.cat([self.encoder(self.augment(b["pixels"])), b["proprio"]], dim=-1)
        with torch.no_grad():
            next_feat = torch.cat([self.encoder(self.augment(b["next_pixels"])), b["next_proprio"]], dim=-1)
            next_action = self.sample(self.actor(next_feat), self.stddev(step), self.cfg.noise_clip)
            q1, q2 = self.target(next_feat, next_action)
            target = b["reward"] + b["discount"] * torch.minimum(q1, q2)
        q1, q2 = self.critic(feat, b["action"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Non-finite critic loss")
        self.optimizers[0].zero_grad(set_to_none=True)
        self.optimizers[2].zero_grad(set_to_none=True)
        critic_loss.backward()
        encoder_grad = float(torch.linalg.vector_norm(self.encoder.conv[0].weight.grad))
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), gradient_clip)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), gradient_clip)
        self.optimizers[0].step()
        self.optimizers[2].step()
        self.critic.requires_grad_(False)
        sampled = self.sample(self.actor(feat.detach()), self.stddev(step), self.cfg.noise_clip)
        aq1, aq2 = self.critic(feat.detach(), sampled)
        actor_loss = -rl_weight * torch.minimum(aq1, aq2).mean()
        imitation_loss = torch.tensor(0.)
        if expert is not None and bc_weight:
            with torch.no_grad():
                expert_feat = torch.cat([self.encoder(torch.as_tensor(expert['pixels'])),
                                         torch.as_tensor(expert['proprio'])], dim=-1)
            imitation_loss = F.mse_loss(self.actor(expert_feat), torch.as_tensor(expert['action']))
            actor_loss = actor_loss + bc_weight * imitation_loss
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("Non-finite actor loss")
        self.optimizers[1].zero_grad(set_to_none=True)
        actor_loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), gradient_clip)
        self.optimizers[1].step()
        self.critic.requires_grad_(True)
        with torch.no_grad():
            for actual, target_param in zip(self.critic.parameters(), self.target.parameters()):
                target_param.lerp_(actual, self.cfg.tau)
        self.updates += 1
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach()),
                "batch_reward": float(b["reward"].mean()), "encoder_grad_norm": encoder_grad,
                "target_q_mean": float(target.mean()), "updates": self.updates,
                "imitation_loss": float(imitation_loss.detach())}

    def save(self, path, env_config, steps, training_metadata=None):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save({"agent_config": self.cfg.to_dict(), "env_config": env_config.to_dict(),
                    "steps": steps, "updates": self.updates,
                    "training_metadata": training_metadata,
                    **{k: getattr(self, k).state_dict() for k in ("encoder", "actor", "critic", "target")},
                    "optimizers": [opt.state_dict() for opt in self.optimizers]}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(AgentConfig(**data["agent_config"]))
        for key in ("encoder", "actor", "critic", "target"):
            getattr(agent, key).load_state_dict(data[key])
        for opt, state in zip(agent.optimizers, data["optimizers"]):
            opt.load_state_dict(state)
        agent.updates = data["updates"]
        return agent, data


class Replay:
    """Small CPU ring buffer; terminal and timeout bootstrap are distinct."""
    def __init__(self, capacity, n_step=3, discount=.99, seed=1):
        self.capacity, self.n_step, self.gamma = capacity, n_step, discount
        self.rng = np.random.default_rng(seed)
        self.queue, self.data = deque(), None
        self.index, self.size = 0, 0

    def add(self, obs, action, reward, next_obs, terminated, truncated):
        self.queue.append((obs, np.array(action, copy=True), reward, next_obs, terminated, truncated))
        if terminated or truncated:
            while self.queue:
                self._commit()
                self.queue.popleft()
        elif len(self.queue) >= self.n_step:
            self._commit()
            self.queue.popleft()

    def _commit(self):
        total, discount = 0., 1.
        last = None
        for transition in list(self.queue)[:self.n_step]:
            total += discount * transition[2]
            discount *= self.gamma * (not transition[4])
            last = transition
            if transition[4] or transition[5]:
                break
        first = self.queue[0]
        row = {"pixels": first[0]["pixels"], "proprio": first[0]["proprio"],
               "action": first[1].astype(np.float32), "reward": np.array([total], np.float32),
               "discount": np.array([discount], np.float32),
               "next_pixels": last[3]["pixels"], "next_proprio": last[3]["proprio"]}
        if self.data is None:
            self.data = {k: np.empty((self.capacity, *v.shape), dtype=v.dtype) for k, v in row.items()}
        for k, value in row.items():
            self.data[k][self.index] = value
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.integers(self.size, size=batch_size)
        return {k: v[indices].copy() for k, v in self.data.items()}
