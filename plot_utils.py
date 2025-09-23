import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_close_price(close_data, tickers=None, title="Stock Prices (Close)", figsize=(12, 6)):
    if isinstance(close_data, dict) and "Close" in close_data:
        close_data = close_data["Close"]

    plt.figure(figsize=figsize)

    if tickers is None:
        tickers = close_data.columns
    elif isinstance(tickers, str):
        tickers = [tickers]

    for ticker in tickers:
        plt.plot(close_data.index, close_data[ticker], label=ticker)

    plt.title(title, fontsize=14)
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_indicators(indicators_data, ticker, indicators_to_plot=None, figsize=(14, 10)):
    if ticker not in indicators_data:
        print(f"Ticker '{ticker}' not found in indicators data. Available tickers: {list(indicators_data.keys())}")
        return

    df = indicators_data[ticker]

    if indicators_to_plot is None:
        indicators_to_plot = df.columns

    n = len(indicators_to_plot)
    plt.figure(figsize=figsize)

    for i, ind in enumerate(indicators_to_plot, 1):
        plt.subplot(n, 1, i)
        plt.plot(df.index, df[ind], label=ind)
        plt.title(ind, fontsize=10)
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle(f"Technical Indicators for {ticker}", fontsize=16, y=1.02)
    plt.show()

def plot_sentiments(sentiment_data, tickers=None, title="Average Sentiment Over Time", figsize=(12, 6)):
    plt.figure(figsize=figsize)

    if tickers is None:
        tickers = sentiment_data.columns
    elif isinstance(tickers, str):
        tickers = [tickers]

    for ticker in tickers:
        smoothed = sentiment_data[ticker].rolling(window=7, min_periods=1).mean()
        plt.plot(sentiment_data.index, smoothed, label=ticker)

    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.title(title, fontsize=14)
    plt.ylabel("Sentiment Score")
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training_results(rewards_history):
    plt.figure(figsize=(12, 5))

    for i, rewards in enumerate(rewards_history):
        plt.plot(rewards, alpha=0.3, label=f"Run {i+1}")

    avg_rewards = np.mean(rewards_history, axis=0)
    plt.plot(avg_rewards, linewidth=2, color="black", label="Average")

    plt.title("Training Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_portfolio_evaluation(env, title_suffix=""):
    n_steps = min(len(env.history), len(env.allocations_history))
    values = np.array(env.history[:n_steps])
    allocations = np.array(env.allocations_history[:n_steps])

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    axes[0].plot(env.history_dates[:n_steps], values, label="Portfolio Value")
    axes[0].set_title(f"Portfolio Value Over Time {title_suffix}")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].grid(True)

    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    axes[1].plot(env.history_dates[:n_steps], drawdown, label="Drawdown")
    axes[1].fill_between(env.history_dates[:n_steps], drawdown, alpha=0.3)
    axes[1].set_title(f"Drawdown Over Time {title_suffix}")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True)

    if allocations.size > 0:
        bottom = np.zeros(n_steps)
        for i, ticker in enumerate(env.tickers):
            asset_alloc = allocations[:, i]
            if np.max(asset_alloc) > 0.01:
                axes[2].fill_between(
                    env.history_dates[:n_steps],
                    bottom,
                    bottom + asset_alloc,
                    label=ticker,
                    alpha=0.7,
                )
                bottom += asset_alloc
        
        if n_steps > 0 and bottom[-1] < 0.99:
            axes[2].fill_between(
                range(n_steps),
                bottom,
                np.ones(n_steps),
                label="Other/Error",
                alpha=0.3,
                color="gray"
            )
            
        axes[2].set_title(f"Asset Allocation Over Time {title_suffix}")
        axes[2].set_ylabel("Allocation Weight")
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].grid(True)
        
        if n_steps > 0:
            print("\nFinal Allocation Percentages:")
            for i, ticker in enumerate(env.tickers):
                final_alloc = allocations[-1, i] * 100
                if final_alloc > 0.01:
                    print(f"  {ticker}: {final_alloc:.2f}%")
    else:
        axes[2].text(
            0.5,
            0.5,
            "No allocation history available",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title(f"Asset Allocation Over Time {title_suffix}")

    plt.tight_layout()
    plt.show()

def compare_performance(agents_data, benchmark_data, dates, title="Portfolio Performance Comparison"):
    plt.figure(figsize=(14, 7))
    
    for agent_name, values in agents_data.items():
        plt.plot(dates, values, label=agent_name)
    
    for benchmark_name, values in benchmark_data.items():
        plt.plot(dates, values, label=benchmark_name, linewidth=2)
    
    plt.title(title)
    # plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
