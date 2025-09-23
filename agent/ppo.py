import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange
import random
import copy
import time

from models import PPOActor, PPOCritic

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)
        log_probs = np.array(self.log_probs, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        values_plus = np.append(values, last_value)

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values_plus[t + 1] * mask - values_plus[t]
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "log_probs": torch.tensor(log_probs, dtype=torch.float32),
            "returns": torch.tensor(returns, dtype=torch.float32),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
        }

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

class PPOAgent:
    def __init__(
        self,
        env,
        episodes=100,
        max_steps=None,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        epochs=10,
        minibatch_size=64,
        temperature=20.0,
        seed=42,
    ):
        self.env = env
        self.episodes = episodes
        self.max_steps = max_steps or env.max_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.temperature = temperature
        self.seed = seed

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.actor = PPOActor(self.state_dim, self.action_dim)
        self.actor_old = copy.deepcopy(self.actor)
        self.critic = PPOCritic(self.state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = RolloutBuffer()
        self.total_steps = 0
        self.training_rewards = []
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_action_and_logprob(self, state_np, deterministic=False):
        s = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(s)
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=-1)
            probs = torch.clamp(probs, 1e-8, 1.0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

            if deterministic:
                action = probs.squeeze(0).cpu().numpy()
            else:
                m = torch.distributions.Categorical(probs.squeeze(0))
                idx = m.sample()
                action = probs.squeeze(0).cpu().numpy()

            log_prob = torch.log(probs + 1e-12).sum(dim=-1).item()
            value = self.critic(s).item()
        return action.astype(np.float32), float(log_prob), float(value)

    def train(self, episodes=None, max_steps=None, log_every=10):
        episodes = episodes or self.episodes
        max_steps = max_steps or self.max_steps
        
        self.training_rewards = []
        start_time = time.time()

        for ep in trange(episodes, desc="PPO Training"):
            self.buffer.clear()
            state, _ = self.env.reset(seed=self.seed + ep)
            state = state.astype(np.float32)
            done = False
            ep_reward = 0.0

            for step in range(max_steps):
                action, logp, value = self._get_action_and_logprob(state, deterministic=False)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = next_state.astype(np.float32)
                
                self.buffer.add(state, action, logp, reward, float(done or truncated), value)
                state = next_state
                ep_reward += reward
                self.total_steps += 1

                if done:
                    break

            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                last_value = self.critic(s_t).item()

            batch = self.buffer.compute_returns_and_advantages(last_value, gamma=self.gamma, gae_lambda=self.gae_lambda)
            adv = batch["advantages"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            batch["advantages"] = adv

            old_log_probs = batch["log_probs"]
            n = batch["states"].shape[0]
            idxs = np.arange(n)

            for epoch in range(self.epochs):
                np.random.shuffle(idxs)
                for start in range(0, n, self.minibatch_size):
                    mb_idx = idxs[start : start + self.minibatch_size]
                    s_mb = batch["states"][mb_idx]
                    a_mb = batch["actions"][mb_idx]
                    old_logp_mb = old_log_probs[mb_idx]
                    ret_mb = batch["returns"][mb_idx]
                    adv_mb = batch["advantages"][mb_idx]

                    logits = self.actor(s_mb)
                    scaled_logits = logits / self.temperature
                    probs = F.softmax(scaled_logits, dim=-1)
                    clipped = torch.clamp(probs, 0.0, 0.5)
                    probs = clipped / (clipped.sum(dim=-1, keepdim=True) + 1e-8)

                    new_logp = torch.log(probs + 1e-12).sum(dim=-1)

                    ratio = torch.exp(new_logp - old_logp_mb)

                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_mb
                    actor_loss = -torch.min(surr1, surr2).mean()

                    entropy = - (probs * torch.log(probs + 1e-12)).sum(dim=1).mean()

                    value_pred = self.critic(s_mb)
                    value_loss = F.mse_loss(value_pred, ret_mb)

                    total_loss = actor_loss - self.entropy_coef * entropy + self.value_coef * value_loss

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

            self.training_rewards.append(ep_reward)
            
            if ep % log_every == 0:
                print(f"[PPO] Episode {ep:4d} | Reward: {ep_reward:.4f} | Steps: {n}")

            self.buffer.clear()

        elapsed = time.time() - start_time
        print(f"PPO training finished in {elapsed:.1f}s, total steps {self.total_steps}")
        
        return self.training_rewards

    def select_action(self, state, deterministic=False):
        a, logp, v = self._get_action_and_logprob(state, deterministic=deterministic)
        return a

    def test(self, env=None, deterministic=True):
        if env is None:
            env = self.env
            
        state, _ = env.reset(seed=self.seed)
        state = state.astype(np.float32)
        done = False
        total_reward = 0.0
        
        while not done:
            action = self.select_action(state, deterministic=deterministic)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state.astype(np.float32)
            total_reward += reward
            
        return total_reward, env.history, env.allocations_history

    def save_checkpoint(self, path="ppo_checkpoint.pth"):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "total_steps": self.total_steps,
            "training_rewards": self.training_rewards
        }
        torch.save(checkpoint, path)
        print(f"PPO checkpoint saved to {path}")

    def load_checkpoint(self, path="ppo_checkpoint.pth"):
        checkpoint = torch.load(path, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.training_rewards = checkpoint.get("training_rewards", [])
        print(f"PPO checkpoint loaded from {path}")
