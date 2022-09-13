from sklearn.base import BaseEstimator, TransformerMixin


class SellSignalsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, dates):
        self.dates = dates

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[:, "sell_signal"] = 0

        for date in self.dates:
            X.loc[(X.Date >= date[0]) & (X.Date <= date[1]), "sell_signal"] = 1
        return X
