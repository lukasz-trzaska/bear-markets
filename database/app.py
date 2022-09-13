from datetime import datetime
from threading import Thread

import bs4 as bs
import click
import pymongo
import requests
import yfinance as yf
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv

# from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
HOST = os.getenv("HOST")
print(HOST)
# HOST = "mongodb://localhost:27017"


class manageDatabase:
    """
    Class comprising of functions necessary to fetch SPX500 close prices and push them to mongodb collection.
    """

    def __init__(self, db, coll, host=HOST):

        self.client = MongoClient(host)
        self.database = self.client[f"{db}"]
        self.collection = self.database[f"{coll}"]

    def fetch_tickers(self):
        """
        Get tickers from wikipedia, push them to mongodb collection.
        """
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        soup = bs.BeautifulSoup(html.text, features="lxml")
        table = soup.find("table", {"class": "wikitable sortable"})
        rows = table.findAll("tr")[1:]  # exclude header row

        for row in rows:
            ticker = row.findAll("td")[0].text
            stock = {"ticker": ticker.replace("\n", "")}
            self.collection.insert_one(stock)

        # ind = 0
        # for row in rows:
        #     if ind < 4:
        #         ticker = row.findAll("td")[0].text
        #         stock = {"ticker": ticker.replace("\n", "")}
        #         self.collection.insert_one(stock)
        #         ind += 1

    def fetch_sectors(self):
        """
        Get GICS sector classification from yahoo finance, push updates to mongodb collection.
        """
        count = self.collection.count_documents({})
        with click.progressbar(length=count) as bar:
            ind = 0
            documents = self.collection.find({}, {"_id": 1, "ticker": 1})
            for doc in documents:
                try:
                    stock = yf.Ticker(doc["ticker"])
                    info = stock.info
                    sector, industry = [info.get(key) for key in ["sector", "industry"]]
                except KeyError:
                    sector, industry = "KeyError", "KeyError"
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"sector": sector, "industry": industry}},
                )
                ind += 1
                bar.update(n_steps=1, current_item=ind / count)

    def get_prices(
        self,
        start_date="2010-01-01",
        end_date=datetime.today().strftime("%Y-%m-%d"),
        ohlc="Close",
    ):
        """
        Download prices/volume for each company, push result in {date: price} format into mongodb collection as an array.
        """
        documents = self.collection.find({}, {"_id": 1, "ticker": 1})
        for doc in documents:
            stock = yf.Ticker(doc["ticker"])
            price = stock.history(start=start_date, end=end_date)[ohlc]
            dates_formatted = [i.strftime(format="%Y-%m-%d") for i in price.keys()]
            dictionary = dict(zip(dates_formatted, price.values))
            self.collection.update_one(
                {"_id": doc["_id"]}, {"$set": {"series": dictionary}}
            )

    def update_prices(
        self, end_date=datetime.today().strftime("%Y-%m-%d"), ohlc="Close"
    ):
        """
        Download prices/volume for each company, push result in {date: price} format into mongodb collection as an array.
        """
        last_date = list(
            self.collection.find({}, {"_id": 0, "series": 1})[0]["series"].keys()
        )[-1]
        last_date_formatted = datetime.strptime(last_date, "%Y-%m-%d")
        documents = self.collection.find({}, {"_id": 1, "ticker": 1})
        for doc in documents:
            stock = yf.Ticker(doc["ticker"])
            price = stock.history(start=last_date_formatted)[ohlc]
            dates_formatted = [i.strftime(format="%Y-%m-%d") for i in price.keys()]
            dictionary = dict(zip(dates_formatted, price.values))
            for key in dictionary:
                self.collection.find_one_and_update(
                    {"_id": doc["_id"]}, {"$set": {f"series.{key}": dictionary[key]}}
                )

    def run(self):
        """
        For new database fetch tickers, fetch sectors, collect prices and push into mongodb.
        If database already exists, update prices for each document.
        """
        if __name__ == "__main__":
            if self.collection.estimated_document_count() == 0:
                Thread(target=self.fetch_tickers()).start()
                Thread(target=self.fetch_sectors()).start()
                Thread(target=self.get_prices()).start()
            else:
                self.update_prices()


db = manageDatabase("snp500", "prices")
db.run()
