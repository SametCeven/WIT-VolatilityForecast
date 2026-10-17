"""Run every forecast for one set of settings, and save the results for the app."""

from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from volforecast import baselines, models
from volforecast.data import DATA_DIR, load
from volforecast.evaluate import HORIZON, TEST_START, Split, dm_table, score_table
from volforecast.features import build_dataset, log_returns, realized_vol

RUNS_DIR = DATA_DIR.parent / "output" / "runs"

# How each saved table is read back: its index columns (a Date column holds dates)
INDEX_COLUMNS = {
    "forecasts": ["scenario", "Date"],
    "scores": ["scenario", "model"],
    "dm_tests": ["model"],
    "learned": ["input"],
    "latest": ["model"],
}

# Runs saved before a setting existed used these values for it
EARLIER_VALUES = {"refit": "once"}


def scenario_label(train_start: str) -> str:
    """How a training start is labeled in the results, for example "from 2003"."""
    return f"from {train_start[:4]}"


@dataclass(frozen=True)
class Config:
    """Every setting the pipeline uses. The notebooks use the same defaults but fit once."""

    horizon: int = HORIZON
    test_start: str = TEST_START
    train_start: str = "2003-01-01"  # main scenario: tests, learned inputs, latest forecast
    compare_start: str = "1997-01-01"  # second training start, scored for comparison
    refit: Literal["once", "yearly"] = "yearly"  # when the models are fitted again
    features: tuple[str, ...] = models.FEATURES
    garch_dist: Literal["normal", "t"] = "t"
    xgb_trees: int = 300
    xgb_depth: int = 3
    xgb_learning_rate: float = 0.05

    @property
    def main_scenario(self) -> str:
        return scenario_label(self.train_start)


def forecast_all(
    data: pd.DataFrame, returns: pd.Series, split: Split, config: Config
) -> tuple[pd.DataFrame, LinearRegression, XGBRegressor]:
    """All six forecasts from the cutoff on, plus the two fitted models."""
    features = list(config.features)
    linear = LinearRegression()
    xgb = models.make_xgboost(config.xgb_trees, config.xgb_depth, config.xgb_learning_rate)
    columns = {
        "historical average": baselines.historical_average(data, split),
        "naive": realized_vol(returns, split.horizon),
        "VIX": baselines.rescaled_vix(data, split),
        "GARCH": baselines.garch(returns, split, config.garch_dist),
        "linear regression": models.fit_predict(linear, data, features, split),
        "XGBoost": models.fit_predict(xgb, data, features, split),
    }
    cutoff = split.cutoff
    return pd.DataFrame(columns).loc[cutoff:], linear, xgb


def backtest(
    data: pd.DataFrame, returns: pd.Series, train_start: str, config: Config
) -> pd.DataFrame:
    """All six forecasts for the test period.

    With `refit="once"`, every model is fitted on the years before the test period. With
    "yearly", they are refitted at the start of each test year on everything before it,
    which is how the forecasts would be made in practice.
    """
    if config.refit == "once":
        split = Split(train_start, config.test_start, config.horizon)
        return forecast_all(data, returns, split, config)[0]
    frames = []
    for year in range(int(config.test_start[:4]), data.index[-1].year + 1):
        cutoff = max(config.test_start, f"{year}-01-01")
        split = Split(train_start, cutoff, config.horizon)
        frame = forecast_all(data, returns, split, config)[0]
        end = f"{year}-12-31"
        frames.append(frame.loc[cutoff:end])
    return pd.concat(frames)


def run(config: Config) -> dict[str, pd.DataFrame]:
    """Score every model on the test period and forecast from the latest close."""
    xu100, vix, usdtry = load("xu100"), load("vix"), load("usdtry")
    returns = log_returns(xu100["Close"]).dropna()
    data = build_dataset(returns, vix["Close"], config.horizon, usdtry["Close"])
    target = data["target"]

    starts = {scenario_label(start): start for start in (config.train_start, config.compare_start)}
    main = config.main_scenario
    forecasts: dict[str, pd.DataFrame] = {}
    scores: dict[str, pd.DataFrame] = {}
    for scenario, start in starts.items():
        frame = backtest(data, returns, start, config)
        forecasts[scenario] = frame.join(target)
        scores[scenario] = score_table(frame, target)

    dm_tests = dm_table(
        forecasts[main].drop(columns="target"), target, "linear regression", config.horizon
    )

    # Refit on everything up to the latest close, for today's forecast and the learned inputs
    as_of = f"{data.index[-1]:%Y-%m-%d}"
    live = Split(config.train_start, as_of, config.horizon)
    frame, linear, xgb = forecast_all(data, returns, live, config)
    latest = frame.iloc[-1].rename_axis("model").to_frame("forecast")
    latest.insert(0, "as of", as_of)

    features = list(config.features)
    learned_columns = {
        "linear coefficient": pd.Series(linear.coef_, index=features),
        "XGBoost importance": pd.Series(xgb.feature_importances_, index=features),
    }
    learned = pd.DataFrame(learned_columns)
    learned.loc["intercept", "linear coefficient"] = linear.intercept_

    return {
        "forecasts": pd.concat(forecasts, names=["scenario"]),
        "scores": pd.concat(scores, names=["scenario", "model"]),
        "dm_tests": dm_tests.rename_axis("model"),
        "learned": learned.rename_axis("input"),
        "latest": latest,
    }


def save(results: dict[str, pd.DataFrame], config: Config) -> Path:
    """Write the tables and settings to a new folder in data/output/runs/, named by time."""
    folder = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder.mkdir(parents=True)
    for name, table in results.items():
        table.to_csv(folder / f"{name}.csv")
    settings = asdict(config)
    settings["features"] = ", ".join(config.features)
    pd.DataFrame([settings]).to_csv(folder / "config.csv", index=False)
    return folder


def list_runs() -> list[Path]:
    """Saved run folders, newest first."""
    if not RUNS_DIR.exists():
        return []
    return sorted((path for path in RUNS_DIR.iterdir() if path.is_dir()), reverse=True)


def read_config(folder: Path) -> Config:
    """The settings a saved run used."""
    row = pd.read_csv(folder / "config.csv", dtype=str).iloc[0]
    defaults = Config()
    values: dict[str, Any] = dict(EARLIER_VALUES)
    for setting in fields(Config):
        # Settings added after this run was saved keep their earlier value or default
        if setting.name not in row:
            continue
        default = getattr(defaults, setting.name)
        text = row[setting.name]
        values[setting.name] = (
            tuple(text.split(", ")) if isinstance(default, tuple) else type(default)(text)
        )
    return Config(**values)


def read_table(folder: Path, name: str) -> pd.DataFrame:
    """One saved results table, with its index restored."""
    index = INDEX_COLUMNS[name]
    dates = ["Date"] if "Date" in index else False
    return pd.read_csv(folder / f"{name}.csv", index_col=index, parse_dates=dates)


def load_run(folder: Path) -> tuple[Config, dict[str, pd.DataFrame]]:
    """A saved run's settings and all its tables."""
    return read_config(folder), {name: read_table(folder, name) for name in INDEX_COLUMNS}


def main() -> None:
    config = Config()
    results = run(config)
    folder = save(results, config)
    print("Scores on the test period:")
    print(results["scores"].round(3).to_string())
    print(f"\nVolatility forecast for the next {config.horizon} trading days (annualized %):")
    print(results["latest"].round(1).to_string())
    print(f"\nSaved to {folder}")


if __name__ == "__main__":
    main()
