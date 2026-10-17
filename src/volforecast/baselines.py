"""The four baselines: historical average, naive, VIX rescaled to BIST, and GARCH(1,1)."""

from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model

from volforecast.transform import Split, realized_volatility, training_rows


def historical_average(dataset: pd.DataFrame, split: Split) -> pd.Series:
    """The average target over the training years, as a constant forecast."""
    average = training_rows(dataset[["target"]], split)["target"].mean()
    return pd.Series(average, index=dataset.index)


def naive(returns: pd.Series, split: Split) -> pd.Series:
    """Realized volatility over the last `horizon` days, as the forecast for the next ones."""
    return realized_volatility(returns, split.horizon)


def rescaled_vix(dataset: pd.DataFrame, split: Split) -> pd.Series:
    """VIX rescaled to BIST volatility with a linear fit on the training years."""
    train = training_rows(dataset[["target", "vix_close"]], split)
    slope, intercept = np.polyfit(train["vix_close"], train["target"], 1)
    return intercept + slope * dataset["vix_close"]


def garch(
    returns: pd.Series, split: Split, distribution: Literal["normal", "t"] = "t"
) -> pd.Series:
    """GARCH(1,1) with Student-t (or normal) errors, estimated once on the training years.

    The parameters stay fixed after the forecast start, but each forecast uses the returns
    up to its own date.
    """
    model = arch_model(returns * 100, mean="Zero", vol="GARCH", p=1, q=1, dist=distribution)
    fit = model.fit(first_obs=split.train_start, last_obs=split.forecast_start, disp="off")
    forecast = fit.forecast(horizon=split.horizon, start=split.forecast_start, reindex=False)
    # Daily variances of % returns: average over the horizon, then annualize
    return (forecast.variance.mean(axis=1) * 252) ** 0.5
