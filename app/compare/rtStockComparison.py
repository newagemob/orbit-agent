from app.scrape.rtStockData import RTStockData
import argparse
import yfinance as yf
import json
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

app_path = Path(__file__).parent.parent


class FinancialDataProcessor:
    def __init__(self, tickers, lookback=10):
        self.tickers = tickers
        self.lookback = lookback

    def get_historical_ticker_data(self, ticker, period):
        all_ticker_data = {}

        for ticker in self.tickers:
            # Scrape ticker price via yfinance API
            ticker_info = yf.Ticker(ticker)
            historical_ticker_data = ticker_info.history(period=period)

            # Convert the Ticker_Price_Data to a DataFrame
            ticker_price_data = historical_ticker_data.reset_index()

            # Save Machine Readable DataFrame for future ML training
            filename = f"{app_path}/scrape/output/machine_readable/historical_mr_scraped_data_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json"
            ticker_price_data.to_json(filename)

            # Store the DataFrame in the dictionary
            all_ticker_data[ticker] = ticker_price_data

    def get_last_ten_days(self):
        # Create a dictionary to store the last ten days of stock data for each ticker
        last_ten_days_data = {}

        for ticker in self.tickers:
            # Load historical data for the ticker
            filename = f"{app_path}/scrape/output/machine_readable/historical_mr_scraped_data_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json"
            ticker_price_data = pd.read_json(filename)

            # Extract the last ten days data and convert it to a DataFrame
            last_ten_days_data[ticker] = ticker_price_data[-self.lookback:]

        return last_ten_days_data

    def preprocess_data(self, data):
        print(data)
        # Combine data from different tickers into a single DataFrame
        combined_data = pd.DataFrame()
        for ticker in self.tickers:
            combined_data = combined_data.append(data[f"{ticker}"])

        # Convert the "Timestamp" column to datetime
        combined_data["Timestamp"] = pd.to_datetime(combined_data["Timestamp"])

        # Handle missing values, if any (e.g., forward fill or interpolation)
        combined_data.fillna(method="ffill", inplace=True)

        # Normalize or standardize numerical features (e.g., "Open", "High", "Low", "Close", "Volume")
        scaler = MinMaxScaler()  # or StandardScaler, depending on the model
        numerical_features = ["Open", "High", "Low", "Close", "Volume"]
        combined_data[numerical_features] = scaler.fit_transform(
            combined_data[numerical_features]
        )

        # Feature Engineering: Create additional relevant features
        # For example, you can calculate technical indicators like moving averages, RSI, MACD, etc.
        # You can also consider creating lagged features, difference features, etc.
        combined_data["Moving_Average_5"] = (
            combined_data["Close"].rolling(window=5).mean()
        )
        combined_data["Moving_Average_10"] = (
            combined_data["Close"].rolling(window=10).mean()
        )
        # Add more feature engineering as needed

        return combined_data


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", action="store_true", help="Use OpenAI and SerpAPI")
    args = parser.parse_args()

    tickers = ["MSFT", "META", "NVDA", "FUBO", "MBRX"]

    rt_stock_data = RTStockData(tickers)
    rt_stock_data.get_historical_ticker_data(tickers, "1y")
    last_ten_days_data = rt_stock_data.get_last_ten_days()

    # Create the FinancialDataProcessor instance
    data_processor = FinancialDataProcessor(tickers)

    # Preprocess the data
    preprocessed_data = data_processor.preprocess_data(last_ten_days_data)

    # Save preprocessed data to CSV
    preprocessed_data.to_csv("preprocessed_data.csv", index=False)
