import pandas as pd
import datetime
import tushare as ts

import pickle

from MyMethods.ADF_adj import ADF4DF
from MyMethods.ThreeThreshold import ThreeThred
from MyMethods.Metalabel import MetaLabel
from MyMethods.train_ts_rf import TS_Rf

# read data
# df = pd.read_hdf("../../datasets/data_factor.h5")
# df = df[df["field_"] == "医药生物"]
# del df["field"]
# del df["field_"]

df = pd.read_csv("./feature_train_rename.csv")
df["trade_date"] = pd.to_datetime(df["trade_date"])
df = df[df["trade_date"] >= datetime.datetime(2015, 1, 1)]
df.sort_values(by="trade_date", inplace=True)

code_0 = df.iloc[0]["ts_code"]
result_df = df[df["ts_code"] == code_0]
result_df.reset_index(drop=True, inplace=True)

# three threshold
tt = ThreeThred()
result_df.rename({"return_rate": "return_daily"}, inplace=True, axis=1)
result = tt.get_result(result_df)

result = pd.merge(result, result_df, how="inner", on="trade_date")

# meta label
cols = result.columns
X_cols = [ci for ci in cols if ci not in ['return_rate', 'ts_code', 'FirstTime', 'close',
                                          'label', 'trade_date',
                                          'return', 'label', 'sh_chg']]
y_col = "label"
mt = MetaLabel(result, X_cols, y_col)

# 应该是这么来确定c
# precision1_list, precision_ne_1_list = mt.get_prec()
c = 0.2
result = mt.fit(c)

with open("./result/rf.pkl", "rb+") as f:
    rf = pickle.load(f)
X_cols = result.columns[:-3]
y_col = "y_label"

print(result)
ts_rf = TS_Rf()
ts_rf.transform(result, rf, X_cols, y_col)
print(result)

result_df = result_df[["trade_date", "close"]]

result = pd.merge(result, result_df, on="trade_date", how="inner")
result.to_csv("./result/test_data.csv", index=False)
