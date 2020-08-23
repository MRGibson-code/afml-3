import numpy as np
from statsmodels.tsa.stattools import adfuller
import pandas as pd


class ADFAdjust():
    def get_diff(self, d, ratio=0.01):
        result, w_ = [], 1
        sum_, len_, ratio_ = 0, 0, ratio + 1
        while ratio_ > ratio:
            result.append(w_)
            len_ += 1
            w_ = -result[-1] / len_ * (d - len_ + 1)
            sum_ += abs(w_)
            ratio_ = abs(w_) / sum_
        return result

    def get_diff_list(self, ratio=0.01):
        results = {}
        for d in np.arange(0.01, 4, 0.01):
            results[d] = self.get_diff(d, ratio=ratio)
        return results

    # 二分查找
    def binary_search(self, f, x0, x1, max_xe=1e-10, max_fe=1e-10):
        if abs(f(x0)) < max_fe:
            return x0
        if abs(f(x1)) < max_fe:
            return x1

        if f(x0) * f(x1) > 0:
            return x1
        assert f(x0) * f(x1) < 0
        xe, mid = np.inf, np.inf
        while (abs(xe) > max_xe) and (abs(mid) > max_fe):
            left = f(x0)
            mid = f((x0 + x1) / 2)
            xe = (x1 - x0) / 2
            if mid * left < 0:
                x1 = (x0 + x1) / 2
            else:
                x0 = (x0 + x1) / 2
        else:
            return (x0 + x1) / 2

    def loss_function_adf(self, sr):
        def loss_function_adf_(d):
            d_sr = self.get_diff(d)
            sr_ = sr.rolling(len(d_sr)).apply(lambda x: (x[::-1] * d_sr).sum())
            sr_.dropna(inplace=True)
            p = adfuller(sr_, maxlag=1, regression='c', autolag=None)[1]
            return p - 0.05

        return loss_function_adf_

    def get_diff_n(self, sr):
        df2 = adfuller(sr, maxlag=1, regression='c', autolag=None)
        p = df2[1]
        if p < 0.05:
            # 代表不需要差分
            return 0
        else:
            d = self.binary_search(self.loss_function_adf(sr), 0.1, 5)
            return d

    def fit(self, sr, d):
        d_sr = self.get_diff(d)
        return sr.rolling(len(d_sr)).apply(
            lambda x: (x[::-1] * d_sr).sum(), raw=False)


class ADF4DF():
    '''
    df : code return_rate label
    '''
    def get_dict(self, df):
        adf_adj = ADFAdjust()
        result_dict = {}
        for code in df["code"].unique():
            df_ = df[df["code"] == code]
            cols = list(df_.columns[1:-2])
            d_dict = {}
            for col in cols:
                d_dict[col] = adf_adj.get_diff_n(df_[col])
            result_dict[code] = d_dict

        result_dict_all = {}
        for code, result in result_dict.items():
            for k, v in result.items():
                if k in result_dict_all.keys():
                    result_dict_all[k] = max(result_dict_all[k], v)
                else:
                    result_dict_all[k] = v
        return result_dict_all

    def get_adf_adj(self, df):
        result_dict_all = self.get_dict(df)
        adf_adj = ADFAdjust()
        result_df = pd.DataFrame()
        for code in df["code"].unique():
            df_ = df[df["code"] == code]
            for col, d in result_dict_all.items():
                if abs(d) < 0.01:
                    continue
                sr = df_[col]
                sr_ = adf_adj.fit(sr, d)
                df_[col] = sr_

            result_df = result_df.append(df_)

        result_df.dropna(inplace=True)
        result_df.reset_index(drop=True, inplace=True)

        return result_df, result_dict_all


if __name__ == "__main__":
    import pickle

    df = pd.read_csv("../mydatasets/feature_train.csv")
    ad = ADF4DF()
    result_df, result_dict_all = ad.get_adf_adj(df)

    result_df.to_csv("../mydatasets/ADF_adjust_train.csv", index=False)
    print(result_df)
    with open("../mydatasets/d_dict.pkl", "wb+") as f:
        pickle.dump(result_dict_all, f)
