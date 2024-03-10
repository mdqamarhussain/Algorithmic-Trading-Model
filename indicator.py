import pandas as pd
import matplotlib.pyplot as plt



def calculate_simple_moving_average(series: pd.Series, n: int = 20) -> pd.Series:
    """Calculates the simple moving average"""
    return series.rolling(window=n).mean()

def calculate_macd_oscillator(series: pd.Series, n1: int = 12, n2: int = 26) -> pd.Series:
    """
    Calculate the moving average convergence divergence oscillator, given a
    short moving average of length n1 and a long moving average of length n2
    """
    short_ema = series.ewm(span=n1, min_periods=n1).mean()
    long_ema = series.ewm(span=n2, min_periods=n2).mean()
    return short_ema - long_ema

def calculate_bollinger_bands(series: pd.Series, n: int = 20) -> pd.DataFrame:
    """
    Calculates the Bollinger Bands and returns them as a DataFrame
    """
    sma = series.rolling(window=n).mean()
    rolling_std = series.rolling(window=n).std()

    upper_band = sma + 2 * rolling_std
    lower_band = sma - 2 * rolling_std

    bollinger_bands = pd.DataFrame({'upper': upper_band, 'middle': sma, 'lower': lower_band})
    return bollinger_bands

def calculate_money_flow_volume_series(df: pd.DataFrame) -> pd.Series:
    """
    Calculates money flow volume series
    """
    mfv = df['volume'] * (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
    return mfv

def calculate_money_flow_volume(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    Calculates money flow volume, or q_t in our formula
    """
    return calculate_money_flow_volume_series(df).rolling(n).sum()

def calculate_chaikin_money_flow(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    Calculates the Chaikin money flow
    """
    money_flow_volume = calculate_money_flow_volume(df, n)
    volume_sum = df['volume'].rolling(n).sum()

    return money_flow_volume/volume_sum
if __name__ == '__main__':
    # Assuming you've loaded your dataset into a DataFrame named 'df'
    # Replace this line with your dataset loading method
    # data = load_eod_data('AWU')
    data = pd.read_csv('btc_6h.csv')  # Replace 'your_dataset.csv' with your dataset file name or path

    closes = data['close']
    sma = calculate_simple_moving_average(closes, 10)
    macd = calculate_macd_oscillator(closes, 5, 50)

    bollinger_bands = calculate_bollinger_bands(closes, 100)
    bollinger_bands = bollinger_bands.assign(closes=closes)
    bollinger_bands.plot()

    cmf = calculate_chaikin_money_flow(data)
    # cmf.plot()

    import matplotlib.pyplot as plt

    # For the Bollinger Bands plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size
    plt.plot(bollinger_bands.index, bollinger_bands['middle'], label='Middle Band')
    plt.plot(bollinger_bands.index, bollinger_bands['upper'], label='Upper Band')
    plt.plot(bollinger_bands.index, bollinger_bands['lower'], label='Lower Band')
    plt.plot(bollinger_bands.index, bollinger_bands['closes'], label='Close Price')
    plt.legend()
    plt.title('Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

    # For the Chaikin Money Flow (CMF) plot
    plt.figure(figsize=(10, 4))  # Adjust the figure size
    plt.plot(cmf.index, cmf, label='Chaikin Money Flow', color='green')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Add a zero line for reference
    plt.title('Chaikin Money Flow (CMF)')
    plt.xlabel('Date')
    plt.ylabel('CMF Value')
    plt.legend()
    plt.grid(True)
    plt.show()