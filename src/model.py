import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

def train_single_step_ridge(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0):
    """
    Trains a single-step Ridge regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        alpha (float): Regularization strength. Defaults to 1.0.

    Returns:
        Ridge: The trained Ridge model.
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def recursive_forecast(model, last_row: pd.Series, horizon: int = 7) -> list:
    """
    Performs multi-step forecasting recursively by predicting and updating features.

    Args:
        model (Ridge): The trained Ridge model.
        last_row (pd.Series): The last row of features from the historical data to start forecasting.
        horizon (int): The number of steps to forecast. Defaults to 7.

    Returns:
        list: A list of predicted values for the given horizon.
    """
    preds = []
    current = last_row.copy()

    for _ in range(horizon):
        # Predict the next step
        y_hat = model.predict(current.values.reshape(1, -1))[0]
        preds.append(y_hat)

        # Update lag_1 for the next prediction
        current["lag_1"] = y_hat
        
        # Update rolling_7_mean (assuming it's a simple moving average of the last 7 values)
        # This update logic is a simplification. In a real scenario, you'd need the actual 6 previous values.
        # For demonstration, we'll use a simplified update based on the notebook's example structure.
        if "rolling_7_mean" in current and "lag_7" in current: # This check is to prevent errors if features are not present
            # A more robust rolling mean update would involve managing a window of values.
            # Based on the notebook's approach, let's assume a conceptual update for demonstration.
            # The notebook had: current["rolling_7_mean"] = (current["rolling_7_mean"] * 6 + y_hat) / 7
            # This implies `current["rolling_7_mean"]` actually stores the sum or average over a specific window.
            # For a pure rolling mean on features, we'd need to shift previous values.
            # For simplicity, we'll mimic the notebook's provided (though simplified) update for the recursive approach.
            # A real implementation would need to track the last 7 predictions/actuals to compute a true rolling mean.
            # As per the notebook's recursive function, only lag_1 and rolling_7_mean were explicitly updated.
            # Assuming `rolling_7_mean` needs to include `y_hat` in its next calculation, if it's the most recent item.
            # The example in the notebook for recursive forecast was very simplified for `rolling_7_mean` update.
            # For `rolling_7_mean`, we'd typically need the actual values to compute it correctly.
            # Since we only have `y_hat`, we can only approximate. The notebook example was:
            # current["rolling_7_mean"] = (current["rolling_7_mean"] * 6 + y_hat) / 7
            # This means the current `rolling_7_mean` would represent the sum of the last 7, and we are updating that sum
            # by removing the oldest (lag_7) and adding the newest (y_hat).
            # However, `last_row` is just one row. A complete recursive update of rolling features requires more context.
            # Sticking to the most direct interpretation of the notebook's `recursive_forecast`:
            current["rolling_7_mean"] = (current["rolling_7_mean"] * 6 + y_hat) / 7 # simplified as in notebook
            # Note: A proper rolling update requires managing a queue of the last N values.
            # The current approach is a direct translation of the notebook's (potentially simplified) example.

    return preds

def train_direct_ridge_models(df_daily_with_features: pd.DataFrame, split_date: str, H: int = 7, alpha: float = 1.0):
    """
    Trains independent Ridge models for each horizon in a direct forecasting strategy.

    Args:
        df_daily_with_features (pd.DataFrame): DataFrame with original series and features.
        split_date (str): Date string to split training and testing data.
        H (int): Maximum forecast horizon. Defaults to 7.
        alpha (float): Regularization strength for Ridge models. Defaults to 1.0.

    Returns:
        dict: A dictionary where keys are horizons (1 to H) and values are trained Ridge models.
    """
    df_direct = df_daily_with_features.copy()

    # Prepare target columns for each horizon
    for h in range(1, H + 1):
        df_direct[f"y_t+{h}"] = df_direct["y"].shift(-h)

    df_direct = df_direct.dropna() # Drop rows with NaNs created by shifting

    models = {}
    # Features used for direct models (excluding 'y', 'day', and horizon targets)
    # Assuming 'y' is the original target, and 'day' is a leftover column that should not be a feature.
    # Also exclude already created y_t+h columns from features.
    feature_cols = [col for col in df_direct.columns if not col.startswith('y_t+') and col not in ['y', 'day']]

    for h in range(1, H + 1):
        X = df_direct[feature_cols]
        y_h = df_direct[f"y_t+{h}"]

        X_train_h = X[X.index < split_date]
        y_train_h = y_h[y_h.index < split_date]

        model_h = Ridge(alpha=alpha)
        model_h.fit(X_train_h, y_train_h)
        models[h] = model_h

    return models

def direct_forecast(models: dict, X_test: pd.DataFrame) -> dict:
    """
    Generates multi-step predictions using a dictionary of direct models.

    Args:
        models (dict): Dictionary of trained Ridge models, one for each horizon.
        X_test (pd.DataFrame): Test features to make predictions on.

    Returns:
        dict: A dictionary where keys are horizons and values are arrays of predicted values.
    """
    preds_direct = {
        h: models[h].predict(X_test)
        for h in models
    }
    return preds_direct

def train_and_forecast_residual_models_h(
    train_df_ml: pd.DataFrame,
    test_df_ml: pd.DataFrame,
    features: list,
    horizons: list,
    alpha: float = 0.1
) -> tuple:
    """
    Trains Ridge models for residual series for each horizon and forecasts them.

    Args:
        train_df_ml (pd.DataFrame): Training data for residual models.
        test_df_ml (pd.DataFrame): Test data for residual models.
        features (list): List of feature column names.
        horizons (list): List of forecast horizons (e.g., [1, 2, ..., 7]).
        alpha (float): Regularization strength for Ridge models. Defaults to 0.1.

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary of trained models, keyed by horizon.
            - dict: Dictionary of predicted residual arrays, keyed by horizon.
    """
    residual_models = {}
    predicted_residuals = {}

    for h in horizons:
        model = Ridge(alpha=alpha)

        X_train = train_df_ml[features]
        y_train = train_df_ml[f"y_h{h}"]

        X_test = test_df_ml[features]
        # y_test = test_df_ml[f"y_h{h}"] # Not used for prediction, only for evaluation

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        residual_models[h] = model
        predicted_residuals[h] = preds

    return residual_models, predicted_residuals

def reconstruct_forecast_with_stl_components(
    pred_residuals_dict: dict,
    trend_series: pd.Series,
    seasonal_series: pd.Series,
    test_index: pd.DatetimeIndex,
    horizons: list
) -> dict:
    """
    Reconstructs the final forecasts by combining predicted residuals with trend and seasonal components.

    Args:
        pred_residuals_dict (dict): Dictionary of predicted residual arrays, keyed by horizon.
        trend_series (pd.Series): The trend component from STL decomposition.
        seasonal_series (pd.Series): The seasonal component from STL decomposition.
        test_index (pd.DatetimeIndex): The datetime index of the test set for alignment.
        horizons (list): List of forecast horizons.

    Returns:
        dict: A dictionary where keys are horizons and values are reconstructed final forecast Series.
    """
    final_forecasts = {}
    for h in horizons:
        pred_resid = pred_residuals_dict[h]

        # Align trend and seasonal components to the corresponding future dates
        # For a forecast at 'test_index' for horizon 'h', we need trend/seasonal at 'test_index + h days'.
        # test_index corresponds to the start of the prediction for each 'h'.
        # E.g., if test_index is '2019-01-01', for h=1, we want trend/seasonal for '2019-01-02'.
        # So we need to shift the test_index by 'h' days to get the correct dates for trend/seasonal.
        
        # Create the target dates for reconstruction
        reconstruction_dates = test_index + pd.to_timedelta(h, unit='D')
        
        # Interpolate if needed, or get closest available if trend/seasonal is not exactly aligned (e.g. from original series)
        # Given `trend` and `seasonal` are daily series derived from `daily_energy`, direct indexing should work.
        trend_for_forecast = trend_series.reindex(reconstruction_dates).values
        seasonal_for_forecast = seasonal_series.reindex(reconstruction_dates).values

        # Ensure we handle potential NaNs if reconstruction_dates go beyond trend/seasonal series end
        # The `reindex` above will introduce NaNs for dates outside the `trend_series` or `seasonal_series` range.
        # Let's align the predicted residuals with these dates as well for consistency.
        
        # Create a DataFrame to manage alignment and NaNs
        temp_df = pd.DataFrame({
            'pred_resid': pred_resid,
            'trend': trend_for_forecast,
            'seasonal': seasonal_for_forecast
        }, index=test_index)
        
        # The reconstructed series should have the `reconstruction_dates` as its index
        reconstructed_series = pd.Series(
            temp_df['pred_resid'] + temp_df['trend'] + temp_df['seasonal'],
            index=reconstruction_dates
        )
        
        # Drop any NaNs that resulted from `reindex` if the `reconstruction_dates` went out of bounds
        final_forecasts[h] = reconstructed_series.dropna()

    return final_forecasts
