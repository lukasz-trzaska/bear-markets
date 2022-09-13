import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, swindow=30, lwindow=60, alpha=0.025):
        self.swindow = swindow
        self.lwindow = lwindow
        self.alpha = alpha

    def daily_returns(self, X):
        daily_returns = (X.iloc[:, 1] - X.iloc[:, 1].shift(1)) * 100
        return daily_returns

    def momentum(self, X, window):
        X = X.iloc[:, 1]
        shifted_X = X.shift(window)
        momentum = X / shifted_X * 100
        return momentum

    def vs_normal_distr(self, X):
        def vs_quantile_binary(array):
            cut_points = [
                np.quantile(array.dropna(), self.alpha),
                np.quantile(array.dropna(), 1 - self.alpha),
            ]

            vs_quantile_binary = []
            for i in array:
                if np.isnan(i):
                    vs_quantile_binary.append(i)
                else:
                    if i >= cut_points[0] and i <= cut_points[1]:
                        vs_quantile_binary.append(0)
                    else:
                        vs_quantile_binary.append(1)
            return vs_quantile_binary

        daily_returns = self.daily_returns(X)
        osm, osr = stats.probplot(daily_returns, dist="norm")[0]
        quantile = []

        for i in range(len(X)):
            ind = np.where(osr == daily_returns[i])[0]
            if ind.size >= 1:
                ind = ind[0]
                quantile.append(np.float64(osm[ind]))
            else:
                quantile.append(np.nan)

        vs_quantile = daily_returns - quantile
        vs_quantile_binary = vs_quantile_binary(vs_quantile)
        vs_quantile_binary_freq = (
            pd.Series(vs_quantile_binary).rolling(self.lwindow, min_periods=1).sum()
            / self.lwindow
        )

        return vs_quantile, vs_quantile_binary, vs_quantile_binary_freq

    def divergence(self, X):
        sMA = X.iloc[:, 1].rolling(window=self.swindow, min_periods=1).mean()
        lMA = X.iloc[:, 1].rolling(window=self.lwindow, min_periods=1).mean()
        divergence = sMA - lMA
        return divergence

    def lags(self, X):
        lag_returns = self.daily_returns(X).shift(1)
        return lag_returns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        output = X.copy()

        output.loc[:, "daily_returns"] = self.daily_returns(X)
        output.loc[:, "momentum30"] = self.momentum(X, window=self.swindow)
        output.loc[:, "momentum60"] = self.momentum(X, window=self.lwindow)
        output.loc[:, "momentum120"] = self.momentum(X, window=120)
        (
            output.loc[:, "vs_quantile"],
            output.loc[:, "vs_quantile_binary"],
            output.loc[:, "vs_quantile_binary_freq"],
        ) = self.vs_normal_distr(X)
        output.loc[:, "divergence"] = self.divergence(X)
        output.loc[:, "lag_returns"] = self.lags(X)

        output = output.dropna(axis=0).reset_index(drop=True)
        # output = output.drop(
        #     output.columns[1], axis=1
        # )

        return output
