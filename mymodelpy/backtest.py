import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./result/test_data.csv")
df["trade_date"] = pd.to_datetime(df["trade_date"])

df.rename({"y_label": "pred_y"}, inplace=True, axis=1)
print(df.head())
print(df.columns)

fund = 10000
equity = fund
stock_num = 0

df.loc[0, "equity_simple"] = equity
for i in range(len(df) - 1):
    df.loc[i, "equity_simple"] = stock_num * df.loc[i, "close"] + fund
    if df.loc[i, "pred_y"] == 1 and df.loc[i, "y_primary"] == 1 and stock_num == 0:
        stock_num = fund / df.loc[i + 1, "close"]
        fund = 0
    if df.loc[i, "pred_y"] == 1 and df.loc[i, "y_primary"] == -1 and stock_num != 0:
        fund = stock_num * df.loc[i + 1, "close"]
        stock_num = 0

sns.set()
plt.plot(df["trade_date"][:-1], df["equity_simple"][:-1])
plt.xticks(rotation=45)

plt.show()
