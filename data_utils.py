import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volume import AccDistIndexIndicator
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from tqdm import tqdm

def get_ohlcv_data(
    stock_tickers=[
        "AAPL", "NVDA", "JNJ", "JPM", "AMZN", "T", "BA", "PG", "XOM", "NEE", "SPG",
    ],
    start_date="2011-01-01",
    end_date="2019-12-31",
):
    if end_date is None:
        end_date = dt.date.today()

    data = yf.download(
        stock_tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
    )

    ohlcv_dict = {field: {} for field in ["Open", "High", "Low", "Close", "Volume"]}

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in stock_tickers:
            for field in ohlcv_dict.keys():
                ohlcv_dict[field][ticker] = data[ticker][field]
    else:
        ticker = stock_tickers[0]
        for field in ohlcv_dict.keys():
            ohlcv_dict[field][ticker] = data[field]

    ohlcv_df = {}
    for field in ohlcv_dict:
        ohlcv_df[field] = pd.DataFrame(ohlcv_dict[field])
        ohlcv_df[field] = ohlcv_df[field].ffill().dropna()

    return ohlcv_df

def get_indicators(ohlcv_data):
    indicators = {}
    for ticker in ohlcv_data["Close"].columns:
        close = ohlcv_data["Close"][ticker]
        high = ohlcv_data["High"][ticker]
        low = ohlcv_data["Low"][ticker]
        volume = ohlcv_data["Volume"][ticker]

        df = pd.DataFrame(index=close.index)
        df["SMA_20"] = SMAIndicator(close=close, window=20).sma_indicator()
        df["EMA_20"] = EMAIndicator(close=close, window=20).ema_indicator()

        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_diff"] = macd.macd_diff()

        df["A/D"] = AccDistIndexIndicator(high=high, low=low, close=close, volume=volume).acc_dist_index()

        df["RSI_14"] = RSIIndicator(close=close, window=14).rsi()
        try:
            df["ADX_14"] = ADXIndicator(high=high, low=low, close=close, window=14).adx()
        except ValueError:
            df["ADX_14"] = np.nan

        df["PSI_14"] = ((close - close.rolling(14).min()) / (close.rolling(14).max() - close.rolling(14).min())) * 100

        df.dropna(inplace=True)

        scaler = MinMaxScaler()
        indicator_cols = [col for col in df.columns if col != "date"]
        df[indicator_cols] = scaler.fit_transform(df[indicator_cols])

        indicators[ticker] = df

    return indicators

def get_sentiment_data(
    sentiment_csv="data/analyst_ratings_processed.csv",
    stock_tickers=None,
    model_name="ProsusAI/finbert",
    batch_size=128,
    device=0,
):
    df = pd.read_csv(sentiment_csv)
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
    if stock_tickers is not None:
        df = df[df['stock'].isin(stock_tickers)]

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

    def batchify(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    tqdm.pandas()
    titles = df['title'].tolist()
    results = []
    for batch in tqdm(batchify(titles, batch_size), total=(len(titles) // batch_size) + 1):
        outputs = finbert(batch)
        results.extend(outputs)

    df['sentiment'] = [r['label'] for r in results]
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment'].str.lower().map(sentiment_map)

    avg_sentiment = df.groupby(['stock', 'date'])['sentiment_score'].mean().reset_index()
    avg_sentiment.columns = ['stock', 'date', 'average_sentiment']

    sentiment_df = avg_sentiment.pivot(index="date", columns="stock", values="average_sentiment")

    if stock_tickers is not None:
        sentiment_df = sentiment_df.reindex(columns=stock_tickers)

    sentiment_df.columns.name = None
    sentiment_df.index.name = 'Date'

    return sentiment_df

def get_sentiment_data_from_csv(
    ohlcv_data,
    sentiment_csv="data/sentiment_data.csv",
    save_csv=None,
    save_excel=None,
    fill_method="hybrid_realistic",
    start_date="2011-01-01",
    end_date="2019-12-31"
):
    import pandas as pd

    stock_tickers = list(ohlcv_data["Close"].columns)
    sentiment_df = pd.read_csv(sentiment_csv, index_col='Date', parse_dates=True)
    sentiment_df = sentiment_df.reindex(columns=stock_tickers)
    sentiment_df.index.name = 'Date'

    if fill_method == "zero":
        sentiment_df = sentiment_df.fillna(0)
    elif fill_method == "ffill":
        sentiment_df = sentiment_df.ffill()
    elif fill_method == "bfill":
        sentiment_df = sentiment_df.bfill()
    elif fill_method == "interpolate":
        sentiment_df = sentiment_df.interpolate(method='linear')
    elif fill_method == "ffill_bfill":
        sentiment_df = sentiment_df.ffill().bfill()
    elif fill_method == "ffill_bfill_interpolate":
        sentiment_df = sentiment_df.interpolate(method='linear')
        sentiment_df = sentiment_df.ffill().bfill()
    elif fill_method == "rolling_mean":
        sentiment_df = sentiment_df.rolling(window=7, min_periods=1, center=True).mean()
        sentiment_df = sentiment_df.ffill().bfill()
    elif fill_method == "hybrid_realistic":
        original_missing = sentiment_df.isna()
        sentiment_df = sentiment_df.rolling(window=5, min_periods=1, center=True).mean()
        sentiment_df = sentiment_df.interpolate(method='linear', limit_direction='both')
        sentiment_df = sentiment_df.ffill().bfill()
        min_valid_ratio = 0.1
        for col in sentiment_df.columns:
            valid_ratio = 1 - original_missing[col].mean()
            if valid_ratio < min_valid_ratio:
                sentiment_df[col] = -2
    else:
        raise ValueError("Invalid fill_method. Choose from: "
                         "'zero', 'ffill', 'bfill', 'interpolate', "
                         "'ffill_bfill', 'ffill_bfill_interpolate', "
                         "'rolling_mean', 'hybrid_realistic'.")

    sentiment_df = sentiment_df.loc[start_date:end_date]

    if save_csv is not None:
        sentiment_df.to_csv(save_csv, index=True)
    if save_excel is not None:
        sentiment_df.to_excel(save_excel, index=True)

    return sentiment_df

def train_test_split_data(price_data, indicator_data=None, sentiment_data=None, train_ratio=0.8):
    common_idx = price_data.index

    if indicator_data is not None:
        for ticker in indicator_data:
            common_idx = common_idx.intersection(indicator_data[ticker].index)

    if sentiment_data is not None:
        common_idx = common_idx.intersection(sentiment_data.index)

    split_idx = int(len(common_idx) * train_ratio)
    train_idx = common_idx[:split_idx]
    test_idx = common_idx[split_idx:]

    price_train = price_data.loc[train_idx]
    price_test = price_data.loc[test_idx]

    indicator_train, indicator_test = None, None
    if indicator_data is not None:
        indicator_train = {ticker: indicator_data[ticker].loc[train_idx] for ticker in indicator_data}
        indicator_test = {ticker: indicator_data[ticker].loc[test_idx] for ticker in indicator_data}

    sentiment_train, sentiment_test = None, None
    if sentiment_data is not None:
        sentiment_train = sentiment_data.loc[train_idx]
        sentiment_test = sentiment_data.loc[test_idx]

    return price_train, price_test, indicator_train, indicator_test, sentiment_train, sentiment_test

def benchmark(close_price_df, initial_balance=10_000):
    _, test = train_test_split(close_price_df, test_size=0.2, shuffle=False)

    daily_returns = test.pct_change().dropna()
    normalized_returns = (1 + daily_returns).cumprod()
    n_assets = normalized_returns.shape[1] if len(normalized_returns.shape) > 1 else 1

    if n_assets > 1:
        equal_weights = np.full(n_assets, 1 / n_assets)
        capital_allocation = initial_balance * equal_weights
        asset_values = normalized_returns * capital_allocation
        portfolio_value = asset_values.sum(axis=1)
        portfolio_value.name = "Portfolio Value"
    else:
        portfolio_value = normalized_returns.squeeze() * initial_balance
        portfolio_value.name = "Portfolio Value"

    return portfolio_value
