"""Linear regression and XGBoost forecasts from the same inputs."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from volforecast.evaluate import Split, training_rows

# Default inputs: realized volatility over the last day, week and month, and the last VIX close
FEATURES = ("rv_1d", "rv_5d", "rv_21d", "vix")

# Every input build_dataset makes, in display order. USD/TRY data starts in 2005.
ALL_FEATURES = ("rv_1d", "rv_5d", "rv_21d", "rv_63d", "downside_21d", "vix", "usdtry_21d")


def make_xgboost(n_estimators: int, max_depth: int, learning_rate: float) -> XGBRegressor:
    """Gradient-boosted trees, with row sampling and a fixed seed so runs repeat exactly."""
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        random_state=0,
    )


def fit_predict(
    model: LinearRegression | XGBRegressor, data: pd.DataFrame, features: list[str], split: Split
) -> pd.Series:
    """Fit on the training rows, then forecast every date that has all the inputs."""
    train = training_rows(data[[*features, "target"]], split)
    model.fit(train[features], train["target"])
    inputs = data[features].dropna()
    return pd.Series(model.predict(inputs), index=inputs.index)
