import pandas as pd
import numpy as np

def calculate_macd(series: pd.Series, n1: int=12, n2: int=26) -> pd.Series:
    """Calculate MACD"""
    ema_n1 = series.ewm(span=n1, min_periods=n1).mean()
    ema_n2 = series.ewm(span=n2, min_periods=n2).mean()
    return ema_n1 - ema_n2

def calculate_bollinger_bands(series: pd.Series, n: int=20) -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    sma = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return pd.DataFrame({'upper_band': upper_band, 'lower_band': lower_band})



# Convert your data into a DataFrame
df = pd.read_csv("btc_6h.csv")
df['datetime'] = pd.to_datetime(df['datetime'])  # Convert 'datetime' column to datetime objects

# Convert 'close' column to a pandas Series for signal generation
close_series = df['close']

# Calculate signals based on MACD and Bollinger Bands
macd_signal = calculate_macd(close_series)
bollinger_signal = calculate_bollinger_bands(close_series)

# Print or use the generated signals as needed
print("MACD Signal:\n", macd_signal)
print("Bollinger Band Signal:\n", bollinger_signal)