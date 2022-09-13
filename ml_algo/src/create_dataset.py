# PATH = os.path.join(os.getcwd(), "ml_algo")
# os.chdir(PATH)


import os
import pickle
from pathlib import Path

import re
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.pipeline import Pipeline

from src.featureEngineering import FeatureEngineering
from src.growthTransformer import GrowthTransformer
from src.sellSignalEncoder import SellSignalsEncoder

DATA_PATH = os.path.join(Path(os.getcwd()).parent, "data\\data.pkl")
TRAIN_PATH = os.path.join(Path(os.getcwd()).parent, "data\\train_set.pkl")
VALID_PATH = os.path.join(Path(os.getcwd()).parent, "data\\valid_set.pkl")

load_dotenv()
HOST = os.getenv("HOST")
SECTOR = os.getenv("SECTOR")

bear_regime_dates = []
with open(os.path.join(os.getcwd(), "src", "bear_regime_dates.txt")) as f:
    lines = f.readlines()
    for line in lines:
        range = re.findall("[0-9]*-[0-9]*-[0-9]*", line)
        bear_regime_dates.append(range)
bear_regime_dates


def main(sector, host, coll="snp500", db="prices"):

    client = MongoClient(host)
    collection = client[coll]
    db = collection[db]

    dates = db.find_one({"series.2010-01-04": {"$lt": 9999}}, {"_id": 0, "series": 1})
    dates = dates["series"].keys()
    df = pd.DataFrame({"Date": dates})
    tickers = db.find({"sector": sector}).distinct("ticker")

    for ticker in tickers:
        dat = db.find_one(
            {"ticker": ticker}, {"_id": 0, "industry": 0, "sector": 0, "ticker": 0}
        )["series"].keys()
        pri = db.find_one(
            {"ticker": ticker}, {"_id": 0, "industry": 0, "sector": 0, "ticker": 0}
        )["series"].values()
        input = pd.DataFrame({"Date": dat, f"{ticker}": pri})
        df = df.merge(input, how="left", on="Date")

    pipe = Pipeline(
        [
            ("growth_transformer", GrowthTransformer()),
            ("feature_engineering", FeatureEngineering()),
            ("sell_signals_encoder", SellSignalsEncoder(bear_regime_dates)),
        ]
    )

    output = pipe.fit_transform(df)
    return output


if __name__ == "__main__":
    output = main(sector=SECTOR, host=HOST)
    with open(DATA_PATH, "wb") as f:
        pickle.dump(output, f)

    with open(TRAIN_PATH, "wb") as f1:
        pickle.dump(
            output.loc[output.Date < "2022-01-01", :].reset_index(drop=True), f1
        )

    with open(VALID_PATH, "wb") as f2:
        pickle.dump(
            output.loc[output.Date >= "2022-01-01", :].reset_index(drop=True), f2
        )
