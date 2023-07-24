
import pandas as pd

class StockDataProcessor:
    def __init__(self, json_data, threshold=2):
        self.json_data = json_data
        self.threshold = threshold
        self.ticker = None
        self.features = None
        self.ticker_price_df = None
        self.trainable_df = None

    def preprocess_json_data(self):
        # Extract relevant information from the JSON data
        self.ticker = self.json_data["Ticker"]["0"]
        self.features = self.json_data["Features"]
        ticker_price_data = self.json_data["Ticker_Price_Data"]["0"]

        # Convert ticker_price_data into a DataFrame and set the Timestamp as the index
        self.ticker_price_df = pd.DataFrame(ticker_price_data)
        self.ticker_price_df["Timestamp"] = pd.to_datetime(self.ticker_price_df["Timestamp"])
        self.ticker_price_df.set_index("Timestamp", inplace=True)

        # Calculate the percentage change in closing price from the previous day
        self.ticker_price_df["Close_Previous_Day"] = self.ticker_price_df["Close"].shift(1)
        self.ticker_price_df["Price_Change_Percentage"] = (
            self.ticker_price_df["Close"] - self.ticker_price_df["Close_Previous_Day"]
        ) / self.ticker_price_df["Close_Previous_Day"] * 100

        # Function to assign buy/sell/hold labels based on the price change threshold
        def label_buy_sell_hold(row):
            if row["Price_Change_Percentage"] > self.threshold:
                return "Buy"
            elif row["Price_Change_Percentage"] < -self.threshold:
                return "Sell"
            else:
                return "Hold"

        # Generate buy/sell/hold labels for the ticker's price data
        self.ticker_price_df["Label"] = self.ticker_price_df.apply(label_buy_sell_hold, axis=1)

    def create_trainable_dataset(self):
        if self.ticker_price_df is None:
            self.preprocess_json_data()

        # Create the trainable dataset
        trainable_data = []
        for timestamp, feature_data in self.features.items():
            timestamp = pd.to_datetime(timestamp)
            features_row = {"Timestamp": timestamp, "Ticker": self.ticker}
            features_row.update(feature_data)
            features_row.update(
                self.ticker_price_df.loc[timestamp - pd.Timedelta(days=365)]
            )  # Get the price data from the previous year
            features_row["Label"] = self.ticker_price_df.loc[timestamp, "Label"]  # Use the label from the current day
            trainable_data.append(features_row)

        # Convert the list of dictionaries to a DataFrame
        self.trainable_df = pd.DataFrame(trainable_data)

    def save_trainable_dataset(self, filename="trainable_dataset.csv"):
        if self.trainable_df is None:
            self.create_trainable_dataset()

        # Save the dataset to a CSV file for further processing and training
        self.trainable_df.to_csv(filename, index=False)
