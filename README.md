# Deep Reinforcement Learning for Portfolio Optimisation

This repository presents an end-to-end implementation of multiple Deep Reinforcement Learning (DRL) algorithms applied to financial portfolio optimisation. The models leverage historical price data, technical indicators, and sentiment signals to make dynamic allocation decisions, and are evaluated against benchmark strategies such as buy-and-hold on the S&P 500 and NASDAQ.

---

## Implemented Agents

The following DRL algorithms are implemented and evaluated:

- **A2C** (Advantage Actor-Critic)
- **DDPG** (Deep Deterministic Policy Gradient)
- **TD3** (Twin Delayed DDPG)
- **PPO** (Proximal Policy Optimization)

Each agent interacts with a custom OpenAI Gym-compatible environment that simulates the dynamics of a multi-asset portfolio with rebalancing.

---

## Key Features

- ✅ Modular and extensible agent implementations
- ✅ Custom Gym environment for portfolio management
- ✅ Technical indicator integration via the `ta` library
- ✅ Sentiment signal ingestion
- ✅ Sliding window time-series cross-validation
- ✅ Visualisation of agent performance and learning curves
- ✅ Baseline comparison with traditional benchmarks
- ✅ Checkpointing and reproducibility support

---

## Installation

To set up the environment and install dependencies:

```bash
# Clone the repository
git clone https://github.com/bonganishube/deep-reinforcement-learning-for-portfolio-optimisation.git
cd deep-reinforcement-learning-for-portfolio-optimisation

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
