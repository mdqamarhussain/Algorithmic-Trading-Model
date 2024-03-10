import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold

def calculate_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Given a dataframe with predictor columns, cross-validated and fit a classifier. Print diagnostics.
    """
    classifier = RandomForestClassifier(n_estimators=100)

    # Separate data
    predictor_columns = ['open', 'high', 'low', 'close', 'volume']
    X = df[predictor_columns]

    # Define the target variable
    df['y'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df = df.dropna()  # Drop the last row as it will have a NaN in 'y'
    y = df['y']

    # Fit cross-validation
    scores = repeated_k_fold(classifier, X, y)

    # Get a full dataset fit for importance scores
    classifier.fit(X, y)

    # Compute diagnostics
    imp = classifier.feature_importances_
    importance_series = pd.Series(imp, index=predictor_columns)
    importance_series = importance_series.sort_values(ascending=False)

    # Compute baseline accuracy
    baseline = np.max(y.value_counts() / y.shape[0])

    # Compute confidence interval for the improvement
    mean_score = scores.mean()
    std_score = scores.std()

    upper_bound = mean_score + 2 * std_score
    lower_bound = mean_score - 2 * std_score
    ibounds = (lower_bound - baseline, upper_bound - baseline)

    print()
    print('Feature importances')
    for col, imp in importance_series.items():
        print(f'{col:24} {imp:>.3f}')
    print()

    print('Cross validation scores')
    print(np.round(100 * scores, 1))
    print()

    print(f'Baseline accuracy {100 * baseline:.1f}%')
    print(f'OOS accuracy {100 * mean_score:.1f}% +/- {200 * std_score:.1f}%')
    print(f'Improvement {100 * ibounds[0]:.1f} to {100 * ibounds[1]:.1f}%')
    print()

    return classifier

# Function for repeated k-fold cross-validation
def repeated_k_fold(classifier, X, y) -> np.ndarray:
    """
    Perform repeated k-fold cross-validation on a classifier.
    """
    n_splits = 5
    n_repeats = 4

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

    scores = []
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        scores.append(score)

    return np.array(scores)

def add_target_column(df):
    # Assuming you're classifying based on 'close' price increase or decrease
    df['y'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df = df.dropna()  # Drop the last row as it will have a NaN in 'y'
    return df

# Load data
df = pd.read_csv("btc_6h.csv")

# Assuming you have loaded your data into a DataFrame called 'df'
# This is where you'd add the function call
df = add_target_column(df)

# Then call your model fitting function
calculate_model(df)
# Assuming calculate_model returns the trained model

def backtest_strategy(model, data):
    # Assuming data has columns: 'open', 'high', 'low', 'close', 'volume'

    # Prepare data for predictions
    predictor_columns = ['open', 'high', 'low', 'close', 'volume']
    X = data[predictor_columns]

    # Use the model to predict 'y' values
    data['predicted_signal'] = model.predict(X)
    data['actual_signal'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)

    # Calculate transaction costs
    transaction_cost_rate = 0.0015  # 0.15% transaction cost
    data['transaction_costs'] = transaction_cost_rate * np.abs(data['predicted_signal'].diff())

    # Calculate the P&L (Profit and Loss)
    data['pnl'] = data['actual_signal'] * data['close'].pct_change() - data['transaction_costs']

    # Calculate other performance metrics (Sharpe ratio, annualized returns, maximum drawdown, etc.)

    # For example:
    # Sharpe ratio
    sharpe_ratio = (data['pnl'].mean() / data['pnl'].std()) * np.sqrt(24 * 365)  # Assuming daily data
    print(f"Sharpe Ratio: {sharpe_ratio}")

    # Annualized returns
    annualized_returns = ((data['pnl'].mean() + 1) ** (365 / len(data))) - 1
    print(f"Annualized Returns: {annualized_returns * 100}%")

    # Maximum Drawdown
    cumulative_returns = (data['pnl'] + 1).cumprod()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    print(f"Maximum Drawdown: {max_drawdown * 100}%")

    # Other metrics and visualizations (equity curve, trade history, etc.) can also be computed/created here

    return data

# Load data
df = pd.read_csv("btc_1h.csv")  # Assuming you have 4-year historical data for BTC/USDT

# Assuming you have trained your model using calculate_model
trained_model = calculate_model(df)

# Perform backtesting
backtest_results = backtest_strategy(trained_model, df)
print(backtest_strategy(trained_model, df) )
