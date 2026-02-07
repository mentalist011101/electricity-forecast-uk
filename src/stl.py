import pandas as pd
from statsmodels.tsa.seasonal import STL

def perform_stl_decomposition(series: pd.Series, period: int = 7):
    """
    Performs Seasonal-Trend decomposition using LOESS (STL) on a time series.

    Args:
        series (pd.Series): The input time series data with a DatetimeIndex.
        period (int): The period of the seasonality. Default is 7 for weekly seasonality.

    Returns:
        tuple: A tuple containing (trend, seasonal, residual) components.
    """
    # Drop any NaN values before performing decomposition
    series_cleaned = series.dropna()
    
    stl = STL(
        series_cleaned,
        period=period,
        robust=True
    )
    res = stl.fit()

    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid

    return trend, seasonal, residual
