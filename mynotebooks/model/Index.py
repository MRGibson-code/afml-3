import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt

"""
技术指标，考虑ma5，不考虑金叉等0-1变量

"""


class TechnicalIndicator(object):
    def __init__(self):
        pass

    def getIndex(self, df):
        df_ = df.copy()
        df_['macd'] = self.macd(df_['close'])
        df_['roc'] = self.roc(df_['close'])
        df_ = pd.concat([df_, self.boll(df_['close'])], axis=1)
        df_['cci'] = self.cci(df_)
        df_['rsi'] = self.rsi(df_['close'])
        df_['wr'] = self.wr(df_)
        df_ = pd.concat([df_, self.kdj(df_)], axis=1)

        return df_.iloc[30:, ]

    def ma(self, sr, period):
        sr_ = sr.rolling(period).mean()
        sr_.fillna(method='bfill', inplace=True)
        return sr_

    def ema(self, sr, period):
        # ema = sr.copy()
        # for i in range(period - 1, len(sr)):
        #     ema[i] = 2 / (period + 1) * (sr[i] - ema[i - 1]) + ema[i - 1]
        ema = sr.ewm(span=period).mean()
        return ema

    def macd(self, sr, short_period=12, long_period=26, dea_period=9):
        ema_short = self.ema(sr, short_period)
        ema_long = self.ema(sr, long_period)
        diff_ = ema_short - ema_long
        dea = self.ema(diff_, dea_period)
        macd = 2 * (diff_ - dea)
        return macd

    def roc(self, sr, period=12):
        sr_ = sr.shift(period)
        roc = sr / sr_ - 1
        roc.fillna(value=0, inplace=True)
        return roc * 100

    # 返回布林线
    def boll(self, sr, period=20, width=2):
        middle_ = self.ma(sr, period=20)
        df_ = pd.DataFrame({'middleboll': middle_})
        df_['upboll'] = middle_.copy()
        df_['downboll'] = middle_.copy()
        std_ = sr.rolling(period).std()
        std_.fillna(0, inplace=True)
        df_['upboll'] = df_['upboll'] + width * std_
        df_['downboll'] = df_['downboll'] - width * std_
        return df_

    # 需要dataframe
    def cci(self, df, period=14):
        tp = (df['high'] + df['close'] + df['low']) / 3
        ma_ = self.ma(df['close'], period)
        md_ = self.ma((ma_ - df['close']).abs(), period)
        cci = (tp - ma_) / (md_ * 0.015)
        return cci

    def rsi(self, sr, period=12):
        # 通过choice里rsi的公式查出
        sr_ = sr.diff(1)
        sr_.fillna(value=0, inplace=True)
        up = sr_.copy()
        down = sr_.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.ewm(period - 1).mean()
        roll_down = (down.abs()).ewm(period - 1).mean()
        rs = roll_up / roll_down
        rsi = 100 - 100 / (rs + 1)
        return rsi

    # 需要dataframe
    def wr(self, df, period=10):
        high_period = df['high'].rolling(period).apply(lambda x: max(x), raw=False)
        low_period = df['low'].rolling(period).apply(lambda x: min(x), raw=False)
        wr_ = (high_period - df['close']) / (high_period - low_period) * 100
        return wr_

    # 需要df, return (k,d,j)
    def kdj(self, df, period=9):
        rsv = (df['close'] - df['low'].rolling(period).min()) / (
                df['high'].rolling(period).max() - df['low'].rolling(period).min()) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d
        df_ = pd.DataFrame({"k": k, "d": d, "j": j})
        return df_


if __name__ == '__main__':
    data = ts.get_hist_data('600000')
    data.sort_values(by='date', inplace=True)

    myindex = TechnicalIndicator()
    data = myindex.getIndex(data)
    print(data)
