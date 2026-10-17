"""The two machine-learning models, linear regression and XGBoost, on the same inputs."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from volforecast.transform import Split, training_rows


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
    model: LinearRegression | XGBRegressor, dataset: pd.DataFrame, features: list[str], split: Split
) -> pd.Series:
    """Fit on the training rows, then forecast every date that has all the inputs."""
    train = training_rows(dataset[[*features, "target"]], split)
    model.fit(train[features], train["target"])
    inputs = dataset[features].dropna()
    return pd.Series(model.predict(inputs), index=inputs.index)
