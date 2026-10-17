"""Turn daily prices into returns, realized volatility and the model dataset, and split it."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns, ln(close / previous close). The first value is NaN."""
    # numpy's type stubs say np.log returns an ndarray, but on a Series it returns a Series
    return cast(pd.Series, np.log(close)).diff()


def realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility in % over the `window` days ending at each date."""
    # Root mean square of returns, not demeaned: the usual realized-volatility definition
    return ((returns**2).rolling(window).mean() * 252) ** 0.5 * 100


def forward_realized_volatility(returns: pd.Series, horizon: int) -> pd.Series:
    """Realized volatility over the `horizon` days after each date: the forecast target."""
    return realized_volatility(returns, horizon).shift(-horizon)


def downside_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Realized volatility from the down days only: up days count as zero returns."""
    return realized_volatility(returns.clip(upper=0), window)


def last_value_before(series: pd.Series, index: pd.Index) -> pd.Series:
    """The last value of `series` strictly before each date in `index`.

    VIX closes hours after BIST, so a forecast made at BIST's close can only use the
    previous US session's VIX. Using the same day's value would leak future information.
    """
    filled = series.reindex(series.index.union(index)).ffill()
    # After shift(1), each date holds the latest value from a strictly earlier date
    return filled.shift(1).reindex(index)


def build_dataset(
    returns: pd.Series, vix_close: pd.Series, horizon: int, usdtry_close: pd.Series | None = None
) -> pd.DataFrame:
    """The target and every possible model input for each BIST date.

    Inputs only use what's known at BIST's close on that date: realized volatility over
    the last day, week, month and quarter, the down days' volatility over the last month,
    the last VIX close before it and, if given, USD/TRY volatility over the last month.
    """
    columns = {
        "target": forward_realized_volatility(returns, horizon),
        "rv_1d": realized_volatility(returns, 1),
        "rv_5d": realized_volatility(returns, 5),
        "rv_21d": realized_volatility(returns, 21),
        "rv_63d": realized_volatility(returns, 63),
        "rv_down_21d": downside_volatility(returns, 21),
        "vix_close": last_value_before(vix_close, returns.index),
    }
    if usdtry_close is not None:
        usdtry_volatility = realized_volatility(log_returns(usdtry_close).dropna(), 21)
        columns["rv_usdtry_21d"] = last_value_before(usdtry_volatility, returns.index)
    return pd.DataFrame(columns)


@dataclass(frozen=True)
class Split:
    """Where training starts, where forecasts start, and how many days each target covers."""

    train_start: str
    forecast_start: str
    horizon: int


def training_rows(dataset: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Complete rows from the training start to `horizon` trading days before the forecasts.

    The target at each date covers the next `horizon` days, so the rows closer to the
    forecast start would include returns from the forecast period.
    """
    start, end = pd.Timestamp(split.train_start), pd.Timestamp(split.forecast_start)
    window = (dataset.index >= start) & (dataset.index < end)
    gap = split.horizon
    return dataset.loc[window].iloc[:-gap].dropna()
