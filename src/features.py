import pandas as pd
import numpy as np

def create_features(series: pd.Series) -> pd.DataFrame:
    """
    Generates calendar, lag, and rolling features from a pandas Series.

    Args:
        series (pd.Series): The input time series data (e.g., daily_energy['load']),
                            with a DatetimeIndex.

    Returns:
        pd.DataFrame: A DataFrame containing the generated features.
    """
    df = pd.DataFrame(index=series.index)

    # Calendar features
    df["dayofweek"] = df.index.dayofweek  # 0 = Monday
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
    df["dayofyear"] = df.index.dayofyear
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["weekofyear"] = df.index.isocalendar().week.astype(int)

    # Lag features
    lags = [1, 7, 14, 28]
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)

    # Rolling features
    df["rolling_7_mean"] = series.rolling(7).mean()
    df["rolling_14_mean"] = series.rolling(14).mean()
    df["rolling_7_std"] = series.rolling(7).std()

    return df.dropna()
