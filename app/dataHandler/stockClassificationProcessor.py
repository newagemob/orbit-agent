import pandas as pd
from pathlib import Path

app_path = Path(__file__).parent.parent

import pandas as pd


class StockClassificationDatasetProcessor:
    def __init__(self, json_data):
        self.json_data = json_data
        self.ticker = json_data["Ticker"]["0"]
        self.date_collected = json_data["Date Collected"]["0"]
        self.price_data = json_data["Ticker_Price_Data"]["0"]
        self.features = json_data["Features"]

    def create_dataset(self, window_size=5):
        # Create a DataFrame from the price data
        price_df = pd.DataFrame(self.price_data)
        price_df["Timestamp"] = pd.to_datetime(price_df["Timestamp"])
        price_df.set_index("Timestamp", inplace=True)

        # Calculate the label 'Action' based on price change
        price_df["Next_Close"] = price_df["Close"].shift(-1)
        price_df["Price_Change"] = price_df["Next_Close"] - price_df["Close"]
        price_df["Action"] = price_df["Price_Change"].apply(self.get_action)

        # Calculate rolling window features
        price_df["Rolling_Volatility"] = (
            price_df["Close"].rolling(window=window_size).std()
        )
        price_df["Rolling_Volume"] = (
            price_df["Volume"].rolling(window=window_size).mean()
        )

        # Drop rows with NaN values due to rolling window calculation
        price_df.dropna(inplace=True)

        # Prepare the final dataset with selected features
        dataset = price_df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Rolling_Volatility",
                "Rolling_Volume",
                "Action",
            ]
        ]

        # save the dataset to a csv file
        dataset.to_csv(
            f"{app_path}/dataHandler/output/classificationDatasets/{self.ticker}_{self.date_collected}.csv"
        )

        return dataset

    def get_action(self, price_change):
        threshold = self.features["price_change_volatility"]
        if price_change > threshold:
            return "Buy"
        elif price_change < -threshold:
            return "Sell"
        else:
            return "Hold"

    def combine_datasets(self):
        # Combine all the datasets into a single csv file
        classificationDatasetsPath = (
            f"{app_path}/dataHandler/output/classificationDatasets"
        )
        datasets = []
        for file in Path(classificationDatasetsPath).glob("*.csv"):
            datasets.append(pd.read_csv(file))
        combined_dataset = pd.concat(datasets)
        combined_dataset.to_csv(
            f"{app_path}/dataHandler/output/classificationDatasets/combined_dataset.csv"
        )


if __name__ == "__main__":
    import json

    for file in Path(f"{app_path}/scrape/output/machine_readable").glob("**/*.json"):
        json_data = json.load(file.open())

        data_processor = StockClassificationDatasetProcessor(json_data)
        dataset = data_processor.create_dataset()

    data_processor.combine_datasets()
