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
import secret
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import datetime as dt
import pytz
from pathlib import Path

app_path = Path(__file__).parent.parent


class RTStockData:
    def __init__(self, llm, tools, agent, tickers):
        self.llm = llm
        self.tools = tools
        self.agent = agent
        self.tickers = tickers

    def get_current_ticker_data(self, ticker, period):
        all_ticker_data = []

        for ticker in self.tickers:
            # Scrape ticker price via yfinance API
            ticker_info = yf.Ticker(ticker)
            current_ticker_data = ticker_info.history(period=period)

            # Convert the Ticker_Price_Data to a list of dictionaries
            ticker_price_data = []
            for timestamp, price_data in current_ticker_data.iterrows():
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
                "Date Collected": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }

            all_ticker_data.append(ticker_combined_data)

            # Save Machine Readable DataFrame for future ML training
            ticker_df = pd.DataFrame([ticker_combined_data])
            ticker_df.to_json(
                f"{app_path}/scrape/output/mr_scraped_data_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            )
            # Save Human Readable CSV
            ticker_df.to_csv(
                f"{app_path}/scrape/output/hr_scraped_data_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
                index=False,
            )

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
                "Date Collected": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            all_ticker_data.append(ticker_combined_data)

            # Save Machine Readable DataFrame for future ML training
            ticker_df = pd.DataFrame([ticker_combined_data])
            ticker_df.to_json(
                f"{app_path}/scrape/output/historical_mr_scraped_data_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            )
            # Save Human Readable CSV
            ticker_df.to_csv(
                f"{app_path}/scrape/output/historical_hr_scraped_data_{ticker}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
                index=False,
            )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", action="store_true", help="Use OpenAI and SerpAPI")
    args = parser.parse_args()

    # Set up OpenAI and SerpAPI if the "--ai" argument is passed
    chat = ChatOpenAI(
        temperature=0,
        openai_api_key=secret.OPENAI_API_KEY,
        model="gpt-3.5-turbo",
    )
    tools = load_tools(
        ["serpapi", "llm-math"], llm=chat, serpapi_api_key=secret.SERPAPI_KEY
    )
    agent = initialize_agent(
        tools,
        chat,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    tickers = ["MSFT", "META", "NVDA", "FUBO", "MBRX"]

    rt_stock_data = RTStockData(chat, tools, agent, tickers)
    combined_data = rt_stock_data.get_current_ticker_data(tickers, "1d")
    rt_stock_data.get_historical_ticker_data(tickers, "1y")
