"""Baseline forecasts of realized volatility over the next `horizon` trading days."""

from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model

from volforecast.evaluate import Split, training_rows


def historical_average(data: pd.DataFrame, split: Split) -> pd.Series:
    """The average target over the training years, as a constant forecast."""
    average = training_rows(data[["target"]], split)["target"].mean()
    return pd.Series(average, index=data.index)


def rescaled_vix(data: pd.DataFrame, split: Split) -> pd.Series:
    """VIX rescaled to BIST volatility with a linear fit on the training years."""
    train = training_rows(data[["target", "vix"]], split)
    slope, intercept = np.polyfit(train["vix"], train["target"], 1)
    return intercept + slope * data["vix"]


def garch(returns: pd.Series, split: Split, dist: Literal["normal", "t"] = "t") -> pd.Series:
    """GARCH(1,1) with Student-t (or normal) errors, estimated once on the training years.

    The parameters stay fixed after the cutoff, but each forecast uses the returns up to
    its own date.
    """
    model = arch_model(returns * 100, mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
    fit = model.fit(first_obs=split.train_start, last_obs=split.cutoff, disp="off")
    forecast = fit.forecast(horizon=split.horizon, start=split.cutoff, reindex=False)
    # Daily variances of % returns: average over the horizon, then annualize
    return (forecast.variance.mean(axis=1) * 252) ** 0.5
