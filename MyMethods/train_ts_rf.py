import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, roc_auc_score
import warnings


class TS_Rf():
    def __init__(self):
        pass

    def get_independence(self, sr1, sr2, date_sr):
        C = pd.DataFrame()
        C["date"] = date_sr
        C["count"] = 0
        C["date"] = pd.to_datetime(C["date"])
        for start_date, end_date in zip(sr1, sr2):
            C.loc[(C["date"] > start_date) & (C["date"] <= end_date), "count"] += 1
        result = pd.DataFrame()
        result["start_date"] = sr1
        for i, (start_date, end_date) in enumerate(zip(sr1, sr2)):
            c = C.loc[(C["date"] > start_date) & (C["date"] <= end_date), "count"]
            if len(c) == 0:
                result.loc[i, "p"] = 1 / 2
                continue
            result.loc[i, "p"] = ((1 / (c + 1)).sum()) / len(c)
        return result

    def get_score_list(self, df, K=5):
        # 5-fold 交叉验证
        df_len = len(df) // K
        start_time_list = [df.loc[df_len * i, "trade_date"]
                           for i in range(K)] + [df.iloc[-1]["trade_date"]]

        # 防止信息泄露
        date_df = pd.DataFrame()
        date_df["trade_date"] = df["trade_date"]
        date_df["trade_date_pre_30"] = date_df["trade_date"].shift(30)
        date_df["trade_date_30"] = date_df["trade_date"].shift(-30)
        date_df.fillna(method="bfill", inplace=True)
        date_df.fillna(method="ffill", inplace=True)
        date_df.set_index("trade_date", inplace=True)

        max_depth_list = [None, 2, 5]
        max_features_list = ["auto", 2, 5]
        score_list = []
        for max_depth in max_depth_list:
            for max_features in max_features_list:
                score = 0
                for i in range(K):
                    df_valid = df[(df["trade_date"] > start_time_list[i])
                                  & (df["trade_date"] < start_time_list[i + 1])]

                    X_valid = df_valid.values[:, :-3]
                    y_valid = df_valid.values[:, -3].astype("int")

                    valid_date_start = df_valid.iloc[0]["trade_date"]
                    valid_date_end = df_valid.iloc[-1]["trade_date"]

                    df_train = df[(df["trade_date"] < min(date_df.loc[valid_date_start, "trade_date_pre_30"])) |
                                  (df["trade_date"] > max(date_df.loc[valid_date_end, "trade_date_30"]))]

                    X_train = df_train.values[:, :-3]
                    y_train = df_train.values[:, -3].astype("int")

                    # 500棵树
                    myforest = RandomForestClassifier(
                        max_depth=max_depth, n_estimators=500, max_features=max_features)
                    myforest.fit(X_train, y_train)
                    score_ = myforest.score(X_valid, y_valid)
                    score += score_
                score_list.append(score / K)

        return score_list

    def fit(self, df, Xcols, ycol):
        # 模型训练
        max_depth = 5
        max_features = "auto"
        n_estimators = 1000

        X_train = df[Xcols].values
        y_train = df[ycol].values.astype("int")
        myforest = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators, max_features=max_features)
        myforest.fit(X_train, y_train)

        return myforest

    def transform(self, df, rf, Xcols, ycol):
        X_ = df[Xcols].values
        df[ycol] = rf.predict(X_)
