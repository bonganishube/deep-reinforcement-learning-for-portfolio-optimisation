import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange
import random
import copy
import time

from models import TD3Actor, TD3Critic

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, s, a, r, s2, d):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr] = float(d)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.state[idxs]),
            torch.tensor(self.action[idxs]),
            torch.tensor(self.reward[idxs]),
            torch.tensor(self.next_state[idxs]),
            torch.tensor(self.done[idxs]),
        )

class TD3Agent:
    def __init__(
        self,
        env,
        episodes=100,
        max_steps=None,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        batch_size=256,
        buffer_size=int(2e5),
        warmup_steps=1000,
        updates_per_step=1,
        seed=42,
    ):
        self.env = env
        self.episodes = episodes
        self.max_steps = max_steps or env.max_steps
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.seed = seed

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.actor = TD3Actor(self.state_dim, self.action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TD3Critic(self.state_dim, self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = ReplayBuffer(self.state_dim, self.action_dim, max_size=buffer_size)

        self.total_steps = 0
        self.training_rewards = []

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def select_action(self, state, noise_std=0.1, deterministic=False, temperature=20.0):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state_t)
            if not deterministic:
                logits = logits + torch.randn_like(logits) * noise_std
            scaled_logits = logits / temperature
            weights = F.softmax(scaled_logits, dim=-1)
            weights = torch.clamp(weights, 0.0, 0.5)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights.cpu().numpy().squeeze().astype(np.float32)

    def _soft_update(self, net, target_net):
        for p, p_targ in zip(net.parameters(), target_net.parameters()):
            p_targ.data.copy_(self.tau * p.data + (1 - self.tau) * p_targ.data)

    def update(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        if self.replay.size < batch_size:
            return

        s, a, r, s2, d = self.replay.sample(batch_size)

        with torch.no_grad():
            target_logits = self.actor_target(s2)
            noise = torch.randn_like(target_logits) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            noisy_logits = target_logits + noise
            noisy_exp = torch.exp(noisy_logits / 20.0 - noisy_logits.max(dim=1, keepdim=True)[0])
            a2 = noisy_exp / (noisy_exp.sum(dim=1, keepdim=True) + 1e-12)

            q1_target, q2_target = self.critic_target(s2, a2)
            q_target = torch.min(q1_target, q2_target)
            target_q = r + (1 - d) * self.gamma * q_target

        current_q1, current_q2 = self.critic(s, a)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        if self.total_steps % self.policy_delay == 0:
            logits = self.actor(s)
            scaled_logits = logits / 20.0
            a_pred = F.softmax(scaled_logits, dim=-1)

            q_val = self.critic.q1(torch.cat([s, a_pred], dim=-1))
            entropy = - (a_pred * torch.log(a_pred + 1e-12)).sum(dim=1).mean()
            actor_loss = -q_val.mean() - 0.01 * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

    def train(self, episodes=None, max_steps=None, log_every=10):
        episodes = episodes or self.episodes
        max_steps = max_steps or self.max_steps
        
        self.training_rewards = []
        start_time = time.time()

        for ep in trange(episodes, desc="TD3 Training"):
            state, _ = self.env.reset(seed=self.seed + ep)
            state = state.astype(np.float32)
            done = False
            ep_reward = 0.0

            for step in range(max_steps):
                if self.total_steps < self.warmup_steps:
                    rand = np.random.rand(self.action_dim)
                    action = rand / (rand.sum() + 1e-8)
                else:
                    action = self.select_action(state, noise_std=0.1, deterministic=False)

                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = next_state.astype(np.float32)
                ep_reward += reward

                self.replay.add(state, action, reward, next_state, done or truncated)
                state = next_state
                self.total_steps += 1

                for _ in range(self.updates_per_step):
                    self.update()

                if done:
                    break

            self.training_rewards.append(ep_reward)
            
            if ep % log_every == 0:
                print(f"[TD3] Episode {ep:4d} | Reward: {ep_reward:.4f} | Buffer: {self.replay.size}/{self.replay.max_size}")

        elapsed = time.time() - start_time
        print(f"TD3 training finished in {elapsed:.1f}s, total steps {self.total_steps}")
        
        return self.training_rewards

    def test(self, env=None, deterministic=True):
        if env is None:
            env = self.env
            
        state, _ = env.reset(seed=self.seed)
        state = state.astype(np.float32)
        done = False
        total_reward = 0.0
        
        while not done:
            action = self.select_action(state, noise_std=0.0, deterministic=deterministic)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state.astype(np.float32)
            total_reward += reward
            
        return total_reward, env.history, env.allocations_history

    def save_checkpoint(self, path="td3_checkpoint.pth"):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "total_steps": self.total_steps,
            "training_rewards": self.training_rewards
        }
        torch.save(checkpoint, path)
        print(f"TD3 checkpoint saved to {path}")

    def load_checkpoint(self, path="td3_checkpoint.pth"):
        checkpoint = torch.load(path, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.training_rewards = checkpoint.get("training_rewards", [])
        print(f"TD3 checkpoint loaded from {path}")
