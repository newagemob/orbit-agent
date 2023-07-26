import argparse
from app.modelHandler.MLTrainingPipeline import FinancialMLPipeline
from app.scrape.rtStockData import RTStockData
from app.dataHandler.featureEngineering import FeatureEngineering
import json
import datetime as dt
from pathlib import Path

app_path = Path(__file__).parent.parent

if __name__ == "__main__":

    def scrape_tickers():
        # extract tickers from "./tickers.json" (array inside of "tickers" key) as a list
        tickers = []
        with open(f"{app_path}/app/tickers.json", "r") as file:
            for ticker in json.load(file)["tickers"]:
                tickers.append(ticker)

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

    parser = argparse.ArgumentParser(
        description="Financial Analysis and Price Prediction Model Training"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["classification", "closing_price_prediction"],
        required=True,
        help="Specify the type of model to train.",
    )
    parser.add_argument(
        "--test", action="store_true", help="Flag to test the trained model."
    )

    args = parser.parse_args()

    def test_classification():
        pass

    def test_price_prediction():
        pass

    if args.model == "classification":
        scrape_tickers()

        classification_data_path = (
            f"{app_path}/dataHandler/output/classificationDatasets/combined_dataset.csv"
        )
        prediction_data_path = (
            f"{app_path}/dataHandler/output/predictionDatasets/combined_dataset.csv"
        )

        pipeline = FinancialMLPipeline(classification_data_path, prediction_data_path)
        pipeline.train_pipeline()

        if args.test:
            test_classification()

    elif args.model == "closing_price_prediction":
        scrape_tickers()

        classification_data_path = (
            f"{app_path}/dataHandler/output/classificationDatasets/combined_dataset.csv"
        )
        prediction_data_path = (
            f"{app_path}/dataHandler/output/predictionDatasets/combined_dataset.csv"
        )

        pipeline = FinancialMLPipeline(classification_data_path, prediction_data_path)
        pipeline.train_pipeline()

        if args.test:
            test_price_prediction()
