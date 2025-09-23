import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import random

from models import DDPGActor, DDPGCritic

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=int(1e6), device=None):
        self.capacity = capacity
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)

    def add(self, s, a, r, ns, d):
        idx = self.ptr
        self.states[idx] = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        self.actions[idx] = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        self.rewards[idx] = torch.as_tensor([r], dtype=torch.float32, device=self.device)
        self.next_states[idx] = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        self.dones[idx] = torch.as_tensor([float(d)], dtype=torch.float32, device=self.device)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        idxs = torch.as_tensor(idxs, dtype=torch.long, device=self.device)
        return (self.states[idxs],
                self.actions[idxs],
                self.rewards[idxs],
                self.next_states[idxs],
                self.dones[idxs])

class DDPGAgent:
    def __init__(
        self,
        env,
        episodes=150,
        gamma=0.99,
        tau=5e-3,
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_size=int(5e5),
        batch_size=256,
        warmup_steps=1000,
        noise_start=0.2,
        noise_end=0.05,
        noise_decay_episodes=200,
        seed=42,
        max_grad_norm=1.0
    ):
        self.env = env
        self.episodes = episodes
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.noise_decay_episodes = noise_decay_episodes
        self.max_grad_norm = max_grad_norm
        self.lambda_entropy = 0.01
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.actor = DDPGActor(self.state_dim, self.action_dim).to(self.device)
        self.critic = DDPGCritic(self.state_dim, self.action_dim).to(self.device)
        self.actor_targ = DDPGActor(self.state_dim, self.action_dim).to(self.device)
        self.critic_targ = DDPGCritic(self.state_dim, self.action_dim).to(self.device)

        self._hard_update(self.actor_targ, self.actor)
        self._hard_update(self.critic_targ, self.critic)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = ReplayBuffer(self.state_dim, self.action_dim, buffer_size, device=self.device)

        self.total_steps = 0
        self.training_rewards = []

    def _hard_update(self, target, source):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target, source, tau):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    def _noise_scale_for_episode(self, ep):
        if self.noise_decay_episodes <= 0:
            return self.noise_end
        frac = min(ep / self.noise_decay_episodes, 1.0)
        return self.noise_end + (self.noise_start - self.noise_end) * (1 + np.cos(np.pi * frac)) / 2

    @torch.no_grad()
    def select_action(self, state, explore=True, noise_scale=0.1):
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        w = self.actor.act(st, noise_scale=noise_scale, explore=explore)[0]
        return w.detach().cpu().numpy()

    def update(self):
        if self.replay.size < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        with torch.no_grad():
            next_actions = self.actor_targ.act(next_states, explore=False)
            target_q = self.critic_targ(next_states, next_actions)
            y = rewards + (1.0 - dones) * self.gamma * target_q

        q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        pred_actions = self.actor(states)
        entropy = -(pred_actions * torch.log(pred_actions + 1e-12)).sum(dim=1).mean()
        actor_loss = -self.critic(states, pred_actions).mean() - self.lambda_entropy * entropy

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        self._soft_update(self.actor_targ, self.actor, self.tau)
        self._soft_update(self.critic_targ, self.critic, self.tau)

    def train(self, episodes=None, max_steps=None, updates_per_step=1, log_every=10):
        if episodes is None:
            episodes = self.episodes
        if max_steps is None:
            max_steps = self.env.max_steps

        self.training_rewards = []
        for ep in trange(episodes, desc="DDPG Training"):
            state, _ = self.env.reset(seed=self.seed + ep)
            state = state.astype(np.float32)
            done = False
            ep_reward = 0.0
            noise_scale = self._noise_scale_for_episode(ep)

            for t in range(max_steps):
                explore = True
                if self.total_steps < self.warmup_steps:
                    rand = np.random.rand(self.action_dim)
                    action = rand / (rand.sum() + 1e-8)
                else:
                    action = self.select_action(state, explore=explore, noise_scale=noise_scale)

                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = next_state.astype(np.float32)
                ep_reward += reward

                self.replay.add(state, action, reward, next_state, done or truncated)
                state = next_state
                self.total_steps += 1

                for _ in range(updates_per_step):
                    self.update()

                if done:
                    break

            self.training_rewards.append(ep_reward)
            if ep % log_every == 0:
                print(f"[DDPG] Episode {ep:4d} | Reward: {ep_reward:.4f} | buffer={self.replay.size} | noise={noise_scale:.3f}")

        return self.training_rewards

    @torch.no_grad()
    def test(self, env=None, max_steps=None):
        env = env or self.env
        if max_steps is None:
            max_steps = env.max_steps

        state, _ = env.reset(seed=self.seed + 999)
        state = state.astype(np.float32)
        done = False
        total_reward = 0.0

        while not done:
            action = self.select_action(state, explore=False, noise_scale=0.0)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state.astype(np.float32)
            total_reward += reward

        return total_reward, env.history, env.allocations_history

    def save_checkpoint(self, path_prefix="ddpg_checkpoint.pth"):
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_targ": self.actor_targ.state_dict(),
            "critic_targ": self.critic_targ.state_dict(),
            "total_steps": self.total_steps,
            "training_rewards": self.training_rewards
        }
        torch.save(payload, path_prefix)
        print(f"DDPG saved -> {path_prefix}")

    def load_checkpoint(self, path_prefix="ddpg_checkpoint.pth"):
        payload = torch.load(path_prefix, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.actor_targ.load_state_dict(payload["actor_targ"])
        self.critic_targ.load_state_dict(payload["critic_targ"])
        self.total_steps = payload.get("total_steps", 0)
        self.training_rewards = payload.get("training_rewards", [])
        print(f"DDPG loaded <- {path_prefix}")
