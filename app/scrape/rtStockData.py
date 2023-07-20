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
from pathlib import Path

app_path = Path(__file__).parent.parent


class RTStockData:
    def __init__(self, llm, tools, agent, tickers):
        self.llm = llm
        self.tools = tools
        self.agent = agent
        self.tickers = tickers

    def combine_ticker_data(self):
        all_ticker_data = []

        for ticker in self.tickers:
            # # Scrape ticker news via serpAPI
            # serpapi_prompt = f"Objective: Find the latest popular stock news for {ticker}. Summarize and provide a brief summary of the sentiment surrounding the stock and anything else you would add as a quantitative analyst. Be sure to include current events and any other relevant information. \n\nTicker: {ticker}\n\nQuantitative Analyst Report:"
            # serpapi_response = self.agent(serpapi_prompt)
            # print(serpapi_response)

            # Scrape ticker price via yfinance API
            ticker_info = yf.Ticker(ticker)
            ticker_hist_max = ticker_info.history(period="max")

            # Create a dictionary to store all the data
            ticker_combined_data = {
                "Ticker": ticker,
                # "SerpAPI_Response": f"Current Sentiment: {serpapi_response['output']}",
                "Ticker_Price_Data": ticker_hist_max,
                "Timestamp": dt.datetime.now(),
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

        # CURRENTLY SKIPPING COMBINED DATA
        # Convert the list of dictionaries to a DataFrame
        combined_df = pd.DataFrame(all_ticker_data)

        # Save Machine Readable DataFrame for future ML training
        combined_df.to_json(
            f"{app_path}/scrape/output/mr_collective_data{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        )
        # Save Human Readable CSV
        combined_df.to_csv(
            f"{app_path}/scrape/output/hr_collective_data_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        )

        return combined_df


if __name__ == "__main__":
    chat = ChatOpenAI(
        temperature=0,
        openai_api_key=secret.OPENAI_API_KEY,
        model="gpt-3.5-turbo",
    )
    messages = [
        SystemMessage(
            content="You are a quantitative analyst reporting to your boss. You are tasked with analyzing the sentiment surrounding a company's stock price based strictly on the news you find."
        ),
    ]
    # llm = OpenAI(temperature=0, openai_api_key=secret.OPENAI_API_KEY, model="gpt-3.5-turbo")

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
    combined_data = rt_stock_data.combine_ticker_data()
    print(combined_data)
