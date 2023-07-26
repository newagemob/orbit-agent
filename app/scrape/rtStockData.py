import yfinance as yf
import json
import pandas as pd
import datetime as dt
from pathlib import Path
from app.dataHandler.featureEngineering import FeatureEngineering

app_path = Path(__file__).parent.parent


class RTStockData:
    def __init__(self, tickers):
        self.tickers = tickers
        self.historical_data = {}

    def store_historical_data(self, period):
        for ticker in self.tickers:
            # Scrape ticker price via yfinance API
            ticker_info = yf.Ticker(ticker)
            historical_ticker_data = ticker_info.history(period=period)

            # Convert the Ticker_Price_Data to a list of dictionaries
            ticker_price_data = []
            for timestamp, price_data in historical_ticker_data.iterrows():
                stock_date = timestamp.strftime("%Y-%m-%d")
                data_dict = {
                    "Timestamp": stock_date,
                    "Open": price_data["Open"],
                    "High": price_data["High"],
                    "Low": price_data["Low"],
                    "Close": price_data["Close"],
                    "Volume": price_data["Volume"],
                    "Dividends": price_data["Dividends"],
                    "Stock Splits": price_data["Stock Splits"],
                }
                ticker_price_data.append(data_dict)

            # Create a dictionary to store all the data
            ticker_combined_data = {
                "Ticker": ticker,
                "Ticker_Price_Data": ticker_price_data,
                "Date Collected": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

            # Save Machine Readable DataFrame for future ML training
            ticker_df = pd.DataFrame([ticker_combined_data])
            ticker_df.to_json(
                f"{app_path}/scrape/output/machine_readable/historical_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json"
            )

    def get_last_ten_days(self, historical_data):
        # Create a dictionary to store the last ten days of stock data for each ticker
        last_ten_days_data = {}

        for ticker in self.tickers:
            # Extract the last ten days data and convert it to a DataFrame
            ticker_price_data = historical_data[ticker]["Ticker_Price_Data"]["0"]
            last_ten_days_data[ticker] = ticker_price_data[-10:]

            # # Save the last ten days data to a JSON file
            # with open(
            #     f"{app_path}/scrape/output/machine_readable/ten_days_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json",
            #     "w",
            # ) as file:
            #     json.dump(last_ten_days_data[ticker], file)

        return last_ten_days_data

    def add_features_to_historical_data(self, features_data):
        for ticker, features in features_data.items():
            # Load the existing historical data for the ticker
            filename = f"{app_path}/scrape/output/machine_readable/historical_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json"
            with open(filename, "r") as file:
                historical_data = json.load(file)
                # add the features to the historical data
                historical_data["Features"] = features

            # Save the updated historical data to a JSON file
            with open(filename, "w") as file:
                json.dump(historical_data, file)

    def add_bullish_engulfing_to_historical_data(self, pattern_data):
        for ticker, pattern in pattern_data.items():
            # Load the existing historical data for the ticker
            filename = f"{app_path}/scrape/output/machine_readable/historical_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json"
            with open(filename, "r") as file:
                historical_data = json.load(file)
                # add the bullish engulfing pattern to the historical data
                historical_data["Bullish_Engulfing_Pattern"] = pattern

            # Save the updated historical data to a JSON file
            with open(filename, "w") as file:
                json.dump(historical_data, file)

    def add_pattern_similarity_to_historical_data(self, pattern_similarity_data):
        for ticker, correlation in pattern_similarity_data.items():
            # Load the existing historical data for the ticker
            filename = f"{app_path}/scrape/output/machine_readable/historical_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json"
            with open(filename, "r") as file:
                historical_data = json.load(file)
                # add the pattern similarity to the historical data
                historical_data["Pattern_Similarity"] = correlation

            # Save the updated historical data to a JSON file
            with open(filename, "w") as file:
                json.dump(historical_data, file)

    def add_pattern_direction_to_historical_data(self, pattern_direction_data):
        for ticker, direction in pattern_direction_data.items():
            # Load the existing historical data for the ticker
            filename = f"{app_path}/scrape/output/machine_readable/historical_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json"
            with open(filename, "r") as file:
                historical_data = json.load(file)
                # add the pattern direction to the historical data
                historical_data["Pattern_Direction"] = direction

            # Save the updated historical data to a JSON file
            with open(filename, "w") as file:
                json.dump(historical_data, file)


if __name__ == "__main__":
    tickers = ["MSFT", "META", "NVDA", "FUBO", "MBRX"]

    rt_stock_data = RTStockData(tickers)
    rt_stock_data.store_historical_data("1y")
    # Load historical data for the tickers
    historical_data = {}
    for ticker in tickers:
        filename = f"{app_path}/scrape/output/machine_readable/historical_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json"
        with open(filename, "r") as file:
            historical_data[ticker] = json.load(file)

    last_ten_days_data = rt_stock_data.get_last_ten_days(historical_data)

    # Continue with feature engineering as before
    feature_engineering = FeatureEngineering(tickers)
    features_data = feature_engineering.calculate_features(
        historical_data, last_ten_days_data
    )

    # Pattern analysis calculations on last 10 days of data
    pattern_engineering = FeatureEngineering(tickers)

    # Add the calculated pattern analysis results to the existing JSON files
    rt_stock_data.add_features_to_historical_data(features_data)
