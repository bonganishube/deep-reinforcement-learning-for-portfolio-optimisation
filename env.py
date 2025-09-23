import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class PortfolioEnv(gym.Env):
    def __init__(self, price_data, indicator_data=None, sentiment_data=None, initial_balance=10000):
        super().__init__()
        self.price_data = price_data
        self.indicator_data = indicator_data
        self.sentiment_data = sentiment_data

        self.tickers = price_data.columns.tolist()
        self.n_assets = price_data.shape[1]
        self.max_steps = price_data.shape[0] - 1
        self.initial_balance = initial_balance
        
        self.dates = self.price_data.index.to_list()

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self.n_indicators = 0
        self.n_sentiments = 0

        if self.indicator_data is not None:
            sample_df = next(iter(indicator_data.values()))
            self.n_indicators = sample_df.shape[1]

        if self.sentiment_data is not None:
            self.n_sentiments = self.sentiment_data.shape[1]

        obs_dim = self.n_assets * (1 + self.n_indicators + (1 if self.n_sentiments > 0 else 0))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def _get_observation(self, step=None):
        if step is None:
            step = self.current_step

        prices = self.price_data.iloc[step].values.astype(np.float32)

        indicators = []
        if self.indicator_data is not None:
            for ticker in self.tickers:
                ind_values = self.indicator_data[ticker].iloc[step].values.astype(np.float32)
                indicators.append(ind_values)
        indicators = np.concatenate(indicators) if indicators else np.array([], dtype=np.float32)

        sentiments = np.array([], dtype=np.float32)
        if self.sentiment_data is not None:
            sentiments = self.sentiment_data.iloc[step].values.astype(np.float32)

        obs = np.concatenate([prices, indicators, sentiments])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.history = [self.balance]
        self.history_dates = [self.dates[self.current_step]]
        self.allocations_history = []
        return self._get_observation(), {}

    def step(self, action):
        weights = np.clip(action, 0, 1)
        weights /= weights.sum() + 1e-8
        self.allocations_history.append(weights.copy())

        current_prices = self.price_data.iloc[self.current_step].values
        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done:
            obs = self._get_observation(step=self.max_steps - 1)
            return obs, 0.0, True, False, {}

        next_prices = self.price_data.iloc[self.current_step].values

        current_prices = np.where(current_prices == 0, 1e-8, current_prices)
        returns = next_prices / current_prices - 1

        portfolio_return = np.dot(returns, weights)
        new_balance = self.balance * (1 + portfolio_return)

        reward = np.log(new_balance / self.balance) if self.balance > 0 else -10
        self.balance = new_balance

        obs = self._get_observation()
        self.history.append(self.balance)
        self.history_dates.append(self.dates[self.current_step])

        return obs, reward, done, False, {}

    def evaluate_performance(self, risk_free_rate=0.0, trading_days=252):
        portfolio_values = np.array(self.history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        if len(returns) == 0:
            return {
                "Cumulative Return": 0.0,
                "Annualized Return": 0.0,
                "Annualized Volatility": 0.0,
                "Sharpe Ratio": 0.0,
                "Max Drawdown": 0.0,
            }

        cumulative_return = portfolio_values[-1] / portfolio_values[0] - 1
        n_periods = len(returns)
        annualized_return = (1 + cumulative_return) ** (trading_days / n_periods) - 1
        annualized_volatility = np.std(returns) * np.sqrt(trading_days)
        excess_returns = returns - (risk_free_rate / trading_days)
        sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
        sharpe_ratio *= np.sqrt(trading_days)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        return {
            "Cumulative Return": cumulative_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
        }
