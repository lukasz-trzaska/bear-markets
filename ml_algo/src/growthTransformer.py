import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GrowthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log=True):

        self.log = log

    def calculate_growth(self, X):

        x = X.copy()
        tickers = x.loc[:, x.columns != "Date"].columns
        for ticker in tickers:
            x.loc[:, ticker] = x[ticker] / x[ticker].shift(1)
        x = x.dropna(axis=0, how="all", subset=tickers).reset_index(drop="True")
        return x

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        input = self.calculate_growth(X)
        df = pd.DataFrame(
            {"Date": input["Date"], "Index": input.mean(axis=1, numeric_only=True)}
        )

        # Create equal-weighted index from growth rates
        df.loc[0, "Index"] = 100
        for i in range(1, len(df)):
            df.loc[i, "Index"] = round(
                df.loc[i, "Index"] * df.loc[i - 1, "Index"], ndigits=2
            )

        x = lambda a: np.log(a)
        if self.log == True:
            df.loc[:, "Index"] = df.loc[:, "Index"].apply(x)

        return df
