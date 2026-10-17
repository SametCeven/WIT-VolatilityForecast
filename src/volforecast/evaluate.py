"""How forecasts are tested: the train/test split, scores and significance tests."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON = 21  # default forecast horizon in trading days, about one month
TEST_START = "2018-01-01"  # default start of the test period


@dataclass(frozen=True)
class Split:
    """Train from `train_start`, forecast from `cutoff` on, `horizon` trading days ahead."""

    train_start: str
    cutoff: str
    horizon: int


def training_rows(data: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Complete rows from the training start to `horizon` trading days before the cutoff.

    The target at each date covers the next `horizon` days, so the rows closer to the
    cutoff would include returns from the forecast period.
    """
    dates = data.index
    window = (dates >= pd.Timestamp(split.train_start)) & (dates < pd.Timestamp(split.cutoff))
    gap = split.horizon
    return data.loc[window].iloc[:-gap].dropna()


def losses(forecast: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Per-date losses: absolute error, squared error and QLIKE (on variances)."""
    error = forecast - target
    ratio = ((target / forecast) ** 2).to_numpy()
    columns = {
        "MAE": error.abs(),
        "MSE": error**2,
        "QLIKE": pd.Series(ratio - np.log(ratio) - 1, index=error.index),
    }
    return pd.DataFrame(columns)


def score(forecast: pd.Series, target: pd.Series) -> dict[str, float]:
    """Bias, MAE and RMSE in volatility points, and QLIKE on variances."""
    loss = losses(forecast, target)
    return {
        "bias": (forecast - target).mean(),
        "MAE": loss["MAE"].mean(),
        "RMSE": loss["MSE"].mean() ** 0.5,
        "QLIKE": loss["QLIKE"].mean(),
    }


def score_table(forecasts: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Score each forecast column on the dates where all forecasts and the target exist."""
    test = forecasts.join(target.rename("target")).dropna()
    rows = {}
    for name, forecast in test.drop(columns="target").items():
        rows[name] = score(forecast, test["target"])
    return pd.DataFrame(rows).T


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int) -> tuple[float, float]:
    """Diebold–Mariano test of equal average loss: the statistic and its p-value.

    A positive statistic means `loss_a` is higher on average. Targets covering the next
    `horizon` days overlap, so neighboring errors are correlated; the variance uses
    Newey–West weights over `horizon - 1` lags to account for that.
    """
    d = (loss_a - loss_b).dropna().to_numpy()
    n = len(d)
    centered = d - d.mean()
    variance = float(centered @ centered) / n
    for lag in range(1, horizon):
        weight = 1 - lag / horizon
        variance += 2 * weight * float(centered[lag:] @ centered[:-lag]) / n
    stat = float(d.mean()) / math.sqrt(variance / n)
    # Two-sided p-value from the standard normal distribution
    return stat, math.erfc(abs(stat) / math.sqrt(2))


def dm_table(
    forecasts: pd.DataFrame, target: pd.Series, benchmark: str, horizon: int
) -> pd.DataFrame:
    """Diebold–Mariano tests of every other forecast against `benchmark`, per loss.

    A positive statistic means the other forecast has the higher loss. The p-value is
    the chance of a gap at least this large if both forecasts were equally good.
    """
    test = forecasts.join(target.rename("target")).dropna()
    base = losses(test[benchmark], test["target"])
    rows = {}
    for name, forecast in test.drop(columns=["target", benchmark]).items():
        loss = losses(forecast, test["target"])
        row = {}
        for column in ["MAE", "MSE", "QLIKE"]:
            stat, p_value = diebold_mariano(loss[column], base[column], horizon)
            row[f"{column} stat"] = stat
            row[f"{column} p"] = p_value
        rows[name] = row
    return pd.DataFrame(rows).T
