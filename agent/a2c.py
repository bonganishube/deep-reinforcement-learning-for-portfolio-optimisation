import os
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from models import Actor, Critic

_eps = 1e-12


def allocation_entropy(allocs: torch.Tensor) -> torch.Tensor:
    return -(allocs * torch.log(allocs + _eps)).sum(dim=1).mean()


def hhi(allocs: torch.Tensor) -> torch.Tensor:
    return (allocs ** 2).sum(dim=1).mean()


class A2CAgent:
    def __init__(
        self,
        env,
        max_steps: int,
        episodes: int = 100,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        entropy_coef: float = 0.01,
        diversification_coef: float = 0.01,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        self.env = env
        self.max_steps = max_steps
        self.episodes = episodes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.diversification_coef = diversification_coef
        self.seed = seed

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = Critic(self.state_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.training_rewards: List[float] = []

        self._set_seed(self.seed)

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self, episodes: Optional[int] = None, log_every: int = 10) -> List[float]:
        if episodes is None:
            episodes = self.episodes

        self.training_rewards = []
        self._set_seed(self.seed)

        for episode in trange(episodes, desc="A2C Episodes"):
            state, _ = self.env.reset(seed=self.seed + episode)
            state = state.astype(np.float32)
            done = False

            log_probs_list, values_list = [], []
            rewards_list, entropies_list = [], []
            raw_actions_list = []

            total_reward = 0.0

            for step in range(self.max_steps):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

                raw_action_logits = self.actor(state_tensor)
                raw_actions_list.append(raw_action_logits)

                probs = torch.softmax(raw_action_logits, dim=-1) + 1e-8
                dist = torch.distributions.Dirichlet(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                value = self.critic(state_tensor).squeeze(0)

                next_state, reward, done, truncated, _ = self.env.step(action.cpu().numpy()[0])
                next_state = next_state.astype(np.float32)

                log_probs_list.append(log_prob)
                values_list.append(value)
                rewards_list.append(float(reward))
                entropies_list.append(entropy)

                total_reward += float(reward)
                state = next_state

                if done:
                    break

            returns = []
            G = 0.0
            for r in reversed(rewards_list):
                G = r + self.gamma * G
                returns.insert(0, G)

            if len(returns) == 0:
                continue

            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
            values_tensor = torch.cat([v.unsqueeze(0) if v.dim() == 0 else v for v in values_list]).squeeze()
            log_probs_tensor = torch.cat(log_probs_list).to(self.device)
            entropies_tensor = torch.cat(entropies_list).to(self.device)

            advantages = returns_tensor - values_tensor.detach()

            raw_batch = torch.cat(raw_actions_list, dim=0).to(self.device)
            allocs = torch.softmax(raw_batch, dim=-1)
            alloc_entropy = allocation_entropy(allocs)
            alloc_hhi = hhi(allocs)

            actor_loss = (
                -(log_probs_tensor * advantages).mean()
                - self.entropy_coef * entropies_tensor.mean()
                - self.diversification_coef * alloc_entropy
            )
            critic_loss = nn.MSELoss()(values_tensor, returns_tensor)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            self.training_rewards.append(total_reward)

            if episode % log_every == 0:
                print(
                    f"[A2C] Ep {episode:4d} | Reward: {total_reward:.4f} | "
                    f"ActorLoss: {actor_loss.item():.4f} | CriticLoss: {critic_loss.item():.4f} | "
                    f"AllocEntropy: {alloc_entropy.item():.4f} | HHI: {alloc_hhi.item():.4f}"
                )

        return self.training_rewards

    def test(self, env: Optional[object] = None, deterministic: bool = True) -> Tuple[float, List[float], List[List[float]]]:
        env = env or self.env
        state, _ = env.reset(seed=self.seed + 999)
        state = state.astype(np.float32)
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                raw_logits = self.actor(state_tensor)
                if deterministic:
                    alloc = torch.softmax(raw_logits, dim=-1).cpu().numpy()[0]
                else:
                    probs = torch.softmax(raw_logits, dim=-1) + 1e-8
                    dist = torch.distributions.Dirichlet(probs)
                    alloc = dist.sample().cpu().numpy()[0]

            next_state, reward, done, truncated, _ = env.step(alloc)
            total_reward += float(reward)
            state = next_state.astype(np.float32)

        return total_reward, env.history, env.allocations_history

    def save_checkpoint(self, path: str = "a2c_checkpoint.pth"):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "training_rewards": self.training_rewards,
            "episodes": self.episodes,
            "max_steps": self.max_steps,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "entropy_coef": self.entropy_coef,
            "diversification_coef": self.diversification_coef,
            "seed": self.seed,
        }
        torch.save(checkpoint, path)
        print(f"[A2C] Checkpoint saved → {path}")

    def load_checkpoint(
        self,
        path: str = "a2c_checkpoint.pth",
        map_location: Optional[torch.device] = None,
        load_optimizers: bool = True
    ):
        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

        if load_optimizers:
            if "actor_optimizer_state_dict" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            if "critic_optimizer_state_dict" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        self.training_rewards = checkpoint.get("training_rewards", [])
        self.episodes = checkpoint.get("episodes", self.episodes)
        self.max_steps = checkpoint.get("max_steps", self.max_steps)
        self.gamma = checkpoint.get("gamma", self.gamma)
        self.learning_rate = checkpoint.get("learning_rate", self.learning_rate)
        self.entropy_coef = checkpoint.get("entropy_coef", self.entropy_coef)
        self.diversification_coef = checkpoint.get("diversification_coef", self.diversification_coef)
        self.seed = checkpoint.get("seed", self.seed)

        print(f"[A2C] Checkpoint loaded ← {path} (episodes={self.episodes}, rewards={len(self.training_rewards)})")
