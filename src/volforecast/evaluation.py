"""How good the forecasts are: scores, and Diebold–Mariano tests of the gaps between them."""

import math

import numpy as np
import pandas as pd


def losses(forecast: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Per-date losses behind MAE, MSE (RMSE squared) and QLIKE (on variances)."""
    error = forecast - target
    ratio = ((target / forecast) ** 2).to_numpy()
    columns = {
        "mae": error.abs(),
        "mse": error**2,
        "qlike": pd.Series(ratio - np.log(ratio) - 1, index=error.index),
    }
    return pd.DataFrame(columns)


def score_forecast(forecast: pd.Series, target: pd.Series) -> dict[str, float]:
    """Bias, MAE and RMSE in volatility points, and QLIKE on variances."""
    loss = losses(forecast, target)
    return {
        "bias": (forecast - target).mean(),
        "mae": loss["mae"].mean(),
        "rmse": loss["mse"].mean() ** 0.5,
        "qlike": loss["qlike"].mean(),
    }


def score_forecasts(forecasts: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Score each forecast column on the dates where all forecasts and the target exist."""
    test = forecasts.join(target.rename("target")).dropna()
    rows = {}
    for model, forecast in test.drop(columns="target").items():
        rows[model] = score_forecast(forecast, test["target"])
    return pd.DataFrame(rows).T.rename_axis("model")


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int) -> tuple[float, float]:
    """Diebold–Mariano test of equal average loss: the statistic and its p-value.

    A positive statistic means `loss_a` is higher on average. Targets covering the next
    `horizon` days overlap, so neighboring errors are correlated; the variance uses
    Newey–West weights over `horizon - 1` lags to account for that.
    """
    loss_difference = (loss_a - loss_b).dropna().to_numpy()
    n = len(loss_difference)
    centered = loss_difference - loss_difference.mean()
    variance = float(centered @ centered) / n
    for lag in range(1, horizon):
        weight = 1 - lag / horizon
        variance += 2 * weight * float(centered[lag:] @ centered[:-lag]) / n
    stat = float(loss_difference.mean()) / math.sqrt(variance / n)
    # Two-sided p-value from the standard normal distribution
    return stat, math.erfc(abs(stat) / math.sqrt(2))


def diebold_mariano_table(
    forecasts: pd.DataFrame, target: pd.Series, benchmark: str, horizon: int
) -> pd.DataFrame:
    """Diebold–Mariano tests of every other forecast against `benchmark`, per loss.

    A positive statistic means the other forecast has the higher loss. The p-value is
    the chance of a gap at least this large if both forecasts were equally good.
    """
    test = forecasts.join(target.rename("target")).dropna()
    benchmark_losses = losses(test[benchmark], test["target"])
    rows = {}
    for model, forecast in test.drop(columns=["target", benchmark]).items():
        model_losses = losses(forecast, test["target"])
        row = {}
        for loss in ["mae", "mse", "qlike"]:
            stat, p_value = diebold_mariano(model_losses[loss], benchmark_losses[loss], horizon)
            row[f"{loss}_stat"] = stat
            row[f"{loss}_p"] = p_value
        rows[model] = row
    return pd.DataFrame(rows).T.rename_axis("model")
