import numpy as np


class FeatureEngineering:
    def __init__(self, tickers):
        self.tickers = tickers

    # historical analysis calculations
    def calculate_features(self, historical_data, current_data):
        features_data = {}
        for ticker in self.tickers:
            # Extract historical and current data for the current ticker
            historical_ticker_data = historical_data[ticker]["Ticker_Price_Data"]["0"]
            current_ticker_data = current_data[ticker]

            # Calculate features for the ticker
            features = {}

            # Price-based features
            prices = [day["Close"] for day in historical_ticker_data]
            current_price = current_ticker_data[-1][
                "Close"
            ]  # Access the last day's price
            prices.append(current_price)  # Add current price to historical prices
            features["average_price"] = np.mean(prices)
            features["price_volatility"] = np.std(prices)

            # Volume-based features
            volumes = [day["Volume"] for day in historical_ticker_data]
            current_volume = current_ticker_data[-1][
                "Volume"
            ]  # Access the last day's volume
            volumes.append(current_volume)  # Add current volume to historical volumes
            features["average_volume"] = np.mean(volumes)
            features["volume_volatility"] = np.std(volumes)

            # Price change-based features
            price_changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
            features["average_price_change"] = np.mean(price_changes)
            features["price_change_volatility"] = np.std(price_changes)

            # Moving average features
            window_size_short = 20
            window_size_long = 50
            sma_short = np.mean(prices[-window_size_short:])
            sma_long = np.mean(prices[-window_size_long:])
            features["sma_ratio"] = sma_short / sma_long

            # Exponential moving average (EMA) feature
            alpha = 2 / (window_size_short + 1)
            ema_short = prices[-1]
            for i in range(1, window_size_short + 1):
                ema_short = alpha * prices[-i] + (1 - alpha) * ema_short
            features["ema_ratio"] = ema_short / sma_long

            # Relative Strength Index (RSI) feature
            rsi_window = 14
            rsi = self.calculate_rsi(prices[-rsi_window:])
            features["rsi"] = rsi

            features_data[ticker] = features

        return features_data

    def calculate_rsi(self, prices):
        diff = np.diff(prices)
        up_changes = diff[diff >= 0]
        down_changes = -diff[diff < 0]

        avg_up = np.mean(up_changes) if len(up_changes) > 0 else 0
        avg_down = np.mean(down_changes) if len(down_changes) > 0 else 0

        rs = avg_up / (avg_down + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # pattern analysis calculations on last 10 days of data
    def calculate_bullish_engulfing(self, historical_data):
        pattern_data = {}
        for ticker in self.tickers:
            # Extract historical data for the current ticker
            print(historical_data[ticker])
            historical_ticker_data = historical_data[ticker]

            # Calculate Bullish Engulfing pattern for the ticker
            pattern = self.check_bullish_engulfing(historical_ticker_data)
            pattern_data[ticker] = pattern

        return pattern_data

    def check_bullish_engulfing(self, prices_data):
        if len(prices_data) < 2:
            return False

        current_day = prices_data[-1]
        previous_day = prices_data[-2]

        # Check if the current day is a bullish engulfing pattern
        if (
            current_day["Close"] > current_day["Open"]
            and previous_day["Close"] < previous_day["Open"]
            and current_day["Open"] < previous_day["Close"]
            and current_day["Close"] > previous_day["Open"]
        ):
            return True
        else:
            return False

    def calculate_pattern_similarity(self, historical_data):
        pattern_similarity_data = {}
        for ticker in self.tickers:
            # Extract the last 10 days of historical data for the current ticker
            historical_ticker_data = historical_data[ticker]
            last_ten_days_prices = [
                day["Close"] for day in historical_ticker_data[-10:]
            ]

            # Calculate pattern similarity as the correlation between the last 10 days and a reference pattern
            # You can define the reference pattern based on your specific requirements
            # For example, you can use a simple linear pattern [1, 2, 3, ..., 10]
            reference_pattern = np.arange(1, 11)
            correlation = np.corrcoef(last_ten_days_prices, reference_pattern)[0, 1]
            pattern_similarity_data[ticker] = correlation

        return pattern_similarity_data

    def calculate_pattern_direction(self, historical_data):
        pattern_direction_data = {}
        for ticker in self.tickers:
            # Extract the last 10 days of historical data for the current ticker
            historical_ticker_data = historical_data[ticker]
            last_ten_days_prices = [
                day["Close"] for day in historical_ticker_data[-10:]
            ]

            # Calculate pattern direction as the direction of the linear regression line
            x = np.arange(1, 11)  # Days [1, 2, ..., 10]
            slope, _ = np.polyfit(x, last_ten_days_prices, 1)  # Linear regression
            pattern_direction_data[ticker] = "Up" if slope > 0 else "Down"

        return pattern_direction_data
