"""
Scrape real-time stock data from Yahoo Finance (yfinance) API
Looking to use PyCoinGecko API for crypto data in the future
  
  - serpAPI: https://serpapi.com/
  - yfinance API: https://pypi.org/project/yfinance/
  
Objectives:

  - Gather "general intelligence" on a specific stock/[stocks] via serpAPI to CSV file
  - Add historical price data for a specific stock/[stocks] via yfinance API to CSV file
  - Pass system_prompt + CSV file to OpenAI API for analysis
  - Return analysis to user
"""

import argparse
import yfinance as yf
import json
import pandas as pd
import datetime as dt
from pathlib import Path

app_path = Path(__file__).parent.parent


class RTStockData:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_historical_ticker_data(self, ticker, period):
        all_ticker_data = []

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

            all_ticker_data.append(ticker_combined_data)

            # Save Machine Readable DataFrame for future ML training
            ticker_df = pd.DataFrame([ticker_combined_data])
            ticker_df.to_json(
                f"{app_path}/scrape/output/machine_readable/historical_mr_scraped_data_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json"
            )

    def get_last_ten_days(self):
        # Create a dictionary to store the last ten days of stock data for each ticker
        last_ten_days_data = {}

        for ticker in self.tickers:
            # Load historical data for the ticker
            filename = f"{app_path}/scrape/output/machine_readable/historical_mr_scraped_data_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json"
            with open(filename, "r") as file:
                historical_data = json.load(file)

            # Extract the last ten days data and convert it to a DataFrame
            ticker_price_data = historical_data["Ticker_Price_Data"]["0"]
            last_ten_days_data[ticker] = ticker_price_data[-10:]

        # Save last ten days data into separate dataframes for each stock
        for ticker, data in last_ten_days_data.items():
            # Create a dictionary to store the last ten days of stock data for each ticker
            last_ten_days_data = {
                "Ticker": ticker,
                "Ticker_Price_Data": data,
                "Date Collected": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

            # Save Machine Readable DataFrame for future ML training
            ticker_df = pd.DataFrame([last_ten_days_data])
            ticker_df.to_json(
                f"{app_path}/scrape/output/machine_readable/last_ten_days_mr_scraped_data_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M')}.json"
            )

        return last_ten_days_data


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", action="store_true", help="Use OpenAI and SerpAPI")
    args = parser.parse_args()

    tickers = ["MSFT", "META", "NVDA", "FUBO", "MBRX"]

    rt_stock_data = RTStockData(tickers)
    rt_stock_data.get_historical_ticker_data(tickers, "1y")
    last_ten_days_data = rt_stock_data.get_last_ten_days()
