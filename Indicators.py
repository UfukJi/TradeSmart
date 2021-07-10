
from tti.indicators import RelativeMomentumIndex
import pandas as pd
from Binance import Binance
import jhtalib
from pandas_ta import qqe


def QQE(df):
    dfclose = df["close"]
    a = qqe(dfclose)
    b = pd.DataFrame(a["QQE_14_5_4.236"])
    b.rename(columns={'QQE_14_5_4.236': 'close'}, inplace=True)
    b["time"] = df["time"]
    b["date"] = df["date"]
    return b

def imi(data):
    data = pd.DataFrame(jhtalib.IMI(df, open="open", close="close"), columns=["close"])
    data["time"] = df["time"]
    data["date"] = df["date"]
    return data


def on_balance_volume(data, trend_periods=21, close_col='close', vol_col='volume'):
    for index, row in data.iterrows():
        if index > 0:
            last_obv = data.at[index - 1, 'obv']
            if row[close_col] > data.at[index - 1, close_col]:
                current_obv = last_obv + row[vol_col]
            elif row[close_col] < data.at[index - 1, close_col]:
                current_obv = last_obv - row[vol_col]
            else:
                current_obv = last_obv
        else:
            last_obv = 0
            current_obv = row[vol_col]

        data.at[index, 'obv'] = current_obv

    data = data.drop(["open", "high", "close", "low", "volume"], axis=1)
    data = data.rename(columns={"obv": "close"})
    return data


def acc_dist(data, trend_periods=21, open_col='open', high_col='high', low_col='low', close_col='close',
             vol_col='volume'):
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            ac = ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * \
                 row[vol_col]
        else:
            ac = 0
        data.at[index, 'acc_dist'] = ac

    data = data.drop(["open", "high", "close", "low", "volume"], axis=1)
    data = data.rename(columns={"acc_dist": "close"})
    return data


def positive_volume_index(data, periods=255, close_col='close', vol_col='volume'):
    data['pvi'] = 0.

    for index, row in data.iterrows():
        if index > 0:
            prev_pvi = data.at[index - 1, 'pvi']
            prev_close = data.at[index - 1, close_col]
            if row[vol_col] > data.at[index - 1, vol_col]:
                pvi = prev_pvi + (row[close_col] - prev_close / prev_close * prev_pvi)
            else:
                pvi = prev_pvi
        else:
            pvi = 1000
        data.at[index, 'pvi'] = pvi
    data = data.drop(["open", "high", "close", "low", "volume"], axis=1)
    data = data.rename(columns={"pvi": "close"})
    return data


###Relative strength index
def rsi(data, periods=14, close_col='close'):
    data['rsi_u'] = 0.
    data['rsi_d'] = 0.
    data['rsi'] = 0.

    for index, row in data.iterrows():
        if index >= periods:

            prev_close = data.at[index - periods, close_col]
            if prev_close < row[close_col]:
                data.at[index, 'rsi_u'] = (row[close_col] - prev_close)
            elif prev_close > row[close_col]:
                data.at[index, 'rsi_d'] = (prev_close - row[close_col])

    data['rsi'] = data['rsi_u'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean() / (
                data['rsi_u'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean() + data['rsi_d'].ewm(
            ignore_na=False, min_periods=0, com=periods, adjust=True).mean())

    data = data.drop(['rsi_u', 'rsi_d', "close", "high", "volume", "low", "open"], axis=1)
    data = data.rename(columns={"rsi": "close"})

    return data


def williams_ad(data, high_col='high', low_col='low', close_col='close'):
    data['williams_ad'] = 0.

    for index, row in data.iterrows():
        if index > 0:
            prev_value = data.at[index - 1, 'williams_ad']
            prev_close = data.at[index - 1, close_col]
            if row[close_col] > prev_close:
                ad = row[close_col] - min(prev_close, row[low_col])
            elif row[close_col] < prev_close:
                ad = row[close_col] - max(prev_close, row[high_col])
            else:
                ad = 0.

            data.at[index, 'williams_ad'] = (ad + prev_value)
    data = data.drop(["open", "high", "close", "low", "volume"], axis=1)
    data = data.rename(columns={"williams_ad": "close"})
    return data


def s_slow(data,x=int):
    i = 0
    j = 0

    high_fourteen = []
    low_fourteen = []
    fast_stochastic = []
    slow_stochastic = []
    date = []
    for index, row in data.iterrows():
        high_fourteen = data['high'].rolling(x).max()
        low_fourteen = data['low'].rolling(x).min()
        fast_stochastic = (data['close'] - low_fourteen) * 100 / (high_fourteen - low_fourteen)
        slow_stochastic = fast_stochastic.rolling(3).mean()
        data["stochastic slow"] = slow_stochastic

    data = data.drop(["open", "high", "close", "low", "volume"], axis=1)
    data = data.rename(columns={"stochastic slow": "close"})
    return data


def volume_price_trend(data, trend_periods=21, close_col="close", vol_col="volume"):
    for index, row in data.iterrows():
        if index > 0:
            last_val = data.at[index - 1, 'Volume Price Trend']
            last_close = data.at[index - 1, close_col]
            today_close = row[close_col]
            today_vol = row[vol_col]
            current_val = last_val + (today_vol * (today_close - last_close) / last_close)
        else:
            current_val = row[vol_col]

        data.at[index, 'Volume Price Trend'] = current_val

    data = data.drop(["open", "high", "close", "low", "volume"], axis=1)
    data = data.rename(columns={"Volume Price Trend": "close"})
    return data


def relative_momentum(data):
    df = pd.DataFrame(data["close"])
    df["date"] = data["date"]
    df = df.set_index(["date"])
    relative = RelativeMomentumIndex(df).getTiData()
    relative = relative.reset_index()

    relative["time"] = data["time"].astype(int)

    return relative


exchange = Binance("credentials.txt")
df = exchange.GetSymbolKlines("BTCUSDT", "1h", 1000)
obvdf = on_balance_volume(df)
print(obvdf)

