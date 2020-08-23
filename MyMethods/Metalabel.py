import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import linear_model


class MetaLabel():
    def __init__(self, df, X_cols, y_col):
        self.df = df

        self.X_cols = X_cols
        self.y_col = y_col

        self.X = df[X_cols]
        self.y = df[y_col]

        # 标准化
        myscaler = StandardScaler()
        self.X_scale = myscaler.fit_transform(self.X)

        model = linear_model.LogisticRegression(
            multi_class='multinomial', solver='newton-cg')

        model.fit(self.X_scale, self.y)
        self.y_pred_p = model.predict_proba(self.X_scale)
        self.y_pred_c = model.predict(self.X_scale)

    def get_prec(self):
        precision1_list = []
        precision_ne_1_list = []
        for c in np.arange(0, 1, 0.01):
            self.y_pred_c[self.y_pred_p[:, 0] > self.y_pred_p[:, 2]] = -1
            self.y_pred_c[self.y_pred_p[:, 0] <= self.y_pred_p[:, 2]] = 1
            self.y_pred_c[self.y_pred_p[:, 1] > c] = 0

            TP_1 = self.y_pred_c[(self.y_pred_c == 1) & (y == 1)].shape[0]
            P_1 = self.y_pred_c[self.y_pred_c == 1].shape[0]

            TP_ne_1 = self.y_pred_c[(self.y_pred_c == -1) & (y == -1)].shape[0]
            P_ne_1 = self.y_pred_c[self.y_pred_c == -1].shape[0]

            precision1 = TP_1 / P_1 if P_1 != 0 else 0
            precision_ne_1 = TP_ne_1 / P_ne_1 if P_ne_1 != 0 else 0

            precision1_list.append(precision1)
            precision_ne_1_list.append(precision_ne_1)

        return precision1_list, precision_ne_1_list

    def fit(self, c):
        # 混淆矩阵
        self.y_pred_c[self.y_pred_p[:, 0] > self.y_pred_p[:, 2]] = -1
        self.y_pred_c[self.y_pred_p[:, 0] <= self.y_pred_p[:, 2]] = 1
        self.y_pred_c[self.y_pred_p[:, 1] > c] = 0

        # 最终结果，注意加上trade_date
        result_df = pd.DataFrame(columns=self.X_cols, data=self.X_scale)
        result_df["y_primary"] = self.y_pred_c.reshape(-1, 1)
        result_df["y_label"] = np.maximum(self.y * self.y_pred_c, 0)

        result_df["trade_date"] = self.df["trade_date"]
        result_df["FirstTime"] = self.df["FirstTime"]

        return result_df
