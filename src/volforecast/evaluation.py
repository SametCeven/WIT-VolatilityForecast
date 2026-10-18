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


def long_run_variance(centered: np.ndarray, lags: int, bartlett: bool = False) -> float:
    """n times the variance of a series' mean, counting its autocovariances up to `lags`.

    By default each lag counts in full. Bartlett (Newey–West) weights shrink linearly with
    the lag instead: they count less of the correlation, but never give a negative variance.
    """
    n = len(centered)
    variance = float(centered @ centered) / n
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1) if bartlett else 1.0
        variance += 2 * weight * float(centered[lag:] @ centered[:-lag]) / n
    return variance


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int) -> tuple[float, float]:
    """Diebold–Mariano test of equal average loss: the statistic and its p-value.

    A positive statistic means `loss_a` is higher on average. Targets covering the next
    `horizon` days overlap, so loss differences up to `horizon - 1` days apart are correlated.
    As in Diebold and Mariano (1995), the variance counts those autocovariances in full, and
    falls back to Bartlett weights in the rare case that comes out negative. Without at least
    two dates, or with identical losses, there's nothing to test and both values are NaN.
    """
    loss_difference = (loss_a - loss_b).dropna().to_numpy()
    n = len(loss_difference)
    if n < 2:
        return math.nan, math.nan
    centered = loss_difference - loss_difference.mean()
    variance = long_run_variance(centered, horizon - 1)
    if variance <= 0:
        variance = long_run_variance(centered, horizon - 1, bartlett=True)
    if variance <= 0:
        return math.nan, math.nan
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
