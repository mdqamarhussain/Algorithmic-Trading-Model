import pandas as pd
import numpy as np
df=pd.read_csv("btc_30m.csv")
# Convert 'datetime' column to datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Converting 'datetime' column to datetime format
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')

# Calculate metrics
df['daily_return'] = df['close'].pct_change()  # Daily returns
df['profit_loss'] = df['close'] - df['open']  # Profit or loss
df['trade_duration'] = df['datetime'].diff().dt.total_seconds() / 3600  # Trade duration in hours

# Metrics calculation
total_closed_trades = len(df)
win_rate = (len(df[df['profit_loss'] > 0]) / total_closed_trades) * 100  # Win Rate
max_drawdown = np.min(df['low'] - df['high'].shift(1))  # Max Drawdown

# Other metrics can be calculated similarly based on their respective formulas

# Displaying the DataFrame with added columns
print(df)

# Displaying calculated metrics
print(f"Total Closed Trades: {total_closed_trades}")
print(f"Win Rate (Profitability %): {win_rate:.2f}%")
print(f"Max Drawdown: {max_drawdown}")
# Display other calculated metrics similarly

# Calculate metrics
df['change'] = df['close'] - df['open']
df['gross_profit'] = np.where(df['change'] > 0, df['change'], 0)
df['gross_loss'] = np.where(df['change'] < 0, abs(df['change']), 0)

winning_trades = df[df['change'] > 0]['change']
losing_trades = df[df['change'] < 0]['change']

average_winning_trade = winning_trades.mean()
average_losing_trade = losing_trades.mean()

buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100

largest_losing_trade = losing_trades.min()
largest_winning_trade = winning_trades.max()

# Assuming risk-free rate as 2% annually
risk_free_rate = 0.02

daily_returns = df['change'] / df['open']
annualized_return = (daily_returns.mean() * 365) / daily_returns.std()
sharpe_ratio = (annualized_return - risk_free_rate) / daily_returns.std()

downside_returns = np.where(df['change'] < 0, df['change'], 0)
downside_deviation = np.std(downside_returns)
sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation

# Assuming datetime column is in datetime format
df['datetime'] = pd.to_datetime(df['datetime'])
df['time_diff'] = df['datetime'].diff().dt.total_seconds() / 60  # in minutes
average_holding_duration = df['time_diff'].mean()

running_profit = df['change'].cumsum()
running_max_dip = (running_profit - running_profit.cummax()).min()
average_dip = (running_profit - running_profit.cummax()).mean()

# Display calculated metrics
print("Gross Loss:", df['gross_loss'].sum())
print("Average Winning Trade (in USDT):", average_winning_trade)
print("Average Losing Trade (in USDT):", average_losing_trade)
print("Buy and Hold Return of BTC:", buy_hold_return)
print("Largest Losing Trade (in USDT):", largest_losing_trade)
print("Largest Winning Trade (in USDT):", largest_winning_trade)
print("Sharpe Ratio:", sharpe_ratio)
print("Sortino Ratio:", sortino_ratio)
print("Average Holding Duration per Trade (in minutes):", average_holding_duration)
print("Max Dip in Running Trade:", running_max_dip)
print("Average Dip in Running Trade:", average_dip)
# Assuming your data is stored in a DataFrame named 'df'
import pandas as pd

# Sample data (replace this with your actual DataFrame)
data = {
    'datetime': pd.to_datetime(['2018-01-01 05:30:00', '2018-01-01 06:00:00', '2018-01-01 06:30:00']),
    'open': [13715.65, 13500.00, 13528.99],
    'close': [13621.12, 13529.01, 13560.00],
    'profit_loss': [-194.53, 29.01, 31.01],
    'trade_duration': [None, 0.5, 0.5]
}



# Calculate total profit
total_profit = sum(df['profit_loss'])

print("Total Profit:", total_profit)


# Assuming the benchmark return is the "Buy and Hold Return of BTC"
benchmark_return = 178.64555598944463

# Total profit from the provided data
total_profit = sum(df['profit_loss'])

# Net profit exceeding the benchmark return
net_profit_exceeding_benchmark = total_profit - benchmark_return

print("Net Profit Exceeding Benchmark Return:", net_profit_exceeding_benchmark)

# Calculate Risk-Reward Ratio
average_winning_trade = df[df['profit_loss'] > 0]['profit_loss'].mean()
average_losing_trade = df[df['profit_loss'] < 0]['profit_loss'].mean()

risk_reward_ratio = abs(average_winning_trade / average_losing_trade) if average_losing_trade != 0 else None

print("Risk-Reward Ratio:", risk_reward_ratio)

# Find the maximum duration time
max_duration_time = df['trade_duration'].max()

print("Max Duration Time of a Single Trade:", max_duration_time)