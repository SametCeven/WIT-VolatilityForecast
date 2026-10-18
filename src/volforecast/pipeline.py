"""One run: every model's forecasts, the backtest, scores, tests and the latest forecast.

Also saves each run to data/output/runs/<date_time>/ and reads saved runs back for the app.
"""

from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from volforecast import baselines, models
from volforecast.config import BASE_RUN, Config
from volforecast.evaluation import diebold_mariano_table, score_forecasts
from volforecast.extract import RAW_DIR, load_prices
from volforecast.transform import Split, build_dataset, log_returns

RUNS_DIR = RAW_DIR.parent / "output" / "runs"

# The index columns of each saved table, to read it back the way it was written
TABLE_INDEX_COLUMNS = {
    "forecasts": ["scenario", "date"],
    "scores": ["scenario", "model"],
    "significance_tests": ["model"],
    "model_weights": ["feature"],
    "latest_forecast": ["model"],
}


class Forecasts(NamedTuple):
    """Every model's forecasts, plus the two fitted machine-learning models."""

    table: pd.DataFrame
    linear: LinearRegression
    xgboost: XGBRegressor


def forecast_all_models(
    dataset: pd.DataFrame, returns: pd.Series, split: Split, config: Config
) -> Forecasts:
    """All six forecasts from the forecast start on."""
    features = list(config.features)
    linear = LinearRegression()
    xgboost = models.make_xgboost(
        config.xgboost_trees, config.xgboost_max_depth, config.xgboost_learning_rate
    )
    columns = {
        "historical_average": baselines.historical_average(dataset, split),
        "naive": baselines.naive(returns, split),
        "vix_rescaled": baselines.vix_rescaled(dataset, split),
        "garch": baselines.garch(returns, split, config.garch_distribution),
        "linear_regression": models.fit_predict(linear, dataset, features, split),
        "xgboost": models.fit_predict(xgboost, dataset, features, split),
    }
    start = split.forecast_start
    return Forecasts(pd.DataFrame(columns).loc[start:], linear, xgboost)


def backtest(
    dataset: pd.DataFrame, returns: pd.Series, train_start: str, config: Config
) -> pd.DataFrame:
    """All six forecasts for the test period.

    With `refit="once"`, every model is fitted on the years before the test period. With
    "yearly", they are refitted at the start of each test year on everything before it,
    which is how the forecasts would be made in practice.
    """
    if config.refit == "once":
        split = Split(train_start, config.test_start, config.horizon)
        return forecast_all_models(dataset, returns, split, config).table
    yearly = []
    for year in range(int(config.test_start[:4]), dataset.index[-1].year + 1):
        start = max(config.test_start, f"{year}-01-01")
        split = Split(train_start, start, config.horizon)
        table = forecast_all_models(dataset, returns, split, config).table
        end = f"{year}-12-31"
        yearly.append(table.loc[start:end])
    return pd.concat(yearly)


def latest_forecast(dataset: pd.DataFrame, returns: pd.Series, config: Config) -> Forecasts:
    """Every model refitted on all the data up to the latest close, and its forecast from there.

    The table has one row per model, with the date of the close the forecast is made at.
    """
    as_of = f"{dataset.index[-1]:%Y-%m-%d}"
    split = Split(config.train_start, as_of, config.horizon)
    forecasts = forecast_all_models(dataset, returns, split, config)
    table = forecasts.table.iloc[-1].rename_axis("model").to_frame("forecast")
    table.insert(0, "as_of", as_of)
    return forecasts._replace(table=table)


def model_weights(
    linear: LinearRegression, xgboost: XGBRegressor, features: list[str]
) -> pd.DataFrame:
    """The linear regression's coefficients and XGBoost's importances, per feature."""
    columns = {
        "linear_coefficient": pd.Series(linear.coef_, index=features),
        "xgboost_importance": pd.Series(xgboost.feature_importances_, index=features),
    }
    weights = pd.DataFrame(columns).rename_axis("feature")
    weights.loc["intercept", "linear_coefficient"] = linear.intercept_
    return weights


def run(config: Config) -> dict[str, pd.DataFrame]:
    """Score every model on the test period and forecast from the latest close."""
    returns = log_returns(load_prices("xu100")["close"]).dropna()
    vix_close, usdtry_close = load_prices("vix")["close"], load_prices("usdtry")["close"]
    dataset = build_dataset(returns, vix_close, config.horizon, usdtry_close)
    target = dataset["target"]

    backtests = {
        scenario: backtest(dataset, returns, start, config)
        for scenario, start in config.scenarios.items()
    }
    forecasts = {scenario: table.join(target) for scenario, table in backtests.items()}
    scores = {scenario: score_forecasts(table, target) for scenario, table in backtests.items()}
    tests = diebold_mariano_table(
        backtests[config.main_scenario], target, "linear_regression", config.horizon
    )
    latest = latest_forecast(dataset, returns, config)
    return {
        "forecasts": pd.concat(forecasts, names=["scenario"]),
        "scores": pd.concat(scores, names=["scenario", "model"]),
        "significance_tests": tests,
        "model_weights": model_weights(latest.linear, latest.xgboost, list(config.features)),
        "latest_forecast": latest.table,
    }


def save_run(results: dict[str, pd.DataFrame], config: Config) -> Path:
    """Write the tables and settings to a new folder in data/output/runs/, named by time."""
    folder = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder.mkdir(parents=True)
    for name, table in results.items():
        table.to_csv(folder / f"{name}.csv")
    settings = asdict(config)
    settings["features"] = ", ".join(config.features)
    # Written last: list_runs only counts folders that have it, so a save cut short is skipped
    pd.DataFrame([settings]).to_csv(folder / "config.csv", index=False)
    return folder


def is_run(folder: Path) -> bool:
    """Whether a folder holds a complete saved run: save_run writes config.csv last."""
    return (folder / "config.csv").is_file()


def list_runs() -> list[Path]:
    """Saved run folders, newest first."""
    if not RUNS_DIR.exists():
        return []
    return sorted((path for path in RUNS_DIR.iterdir() if is_run(path)), reverse=True)


def read_config(folder: Path) -> Config:
    """The settings a saved run used."""
    row = pd.read_csv(folder / "config.csv", dtype=str).iloc[0]
    defaults = Config()
    values = {}
    for setting in fields(Config):
        # Settings added after this run was saved keep their default
        if setting.name not in row:
            continue
        default = getattr(defaults, setting.name)
        text = row[setting.name]
        values[setting.name] = (
            tuple(text.split(", ")) if isinstance(default, tuple) else type(default)(text)
        )
    return Config(**values)


def base_run() -> Path | None:
    """The run others are compared with: BASE_RUN from config.py, if that run is saved.

    Otherwise it's the newest saved run with the default settings, or None if there's none.
    """
    if BASE_RUN and is_run(RUNS_DIR / BASE_RUN):
        return RUNS_DIR / BASE_RUN
    return next((folder for folder in list_runs() if read_config(folder) == Config()), None)


def read_table(folder: Path, name: str) -> pd.DataFrame:
    """One saved results table, with its index restored."""
    index = TABLE_INDEX_COLUMNS[name]
    dates = ["date"] if "date" in index else False
    return pd.read_csv(folder / f"{name}.csv", index_col=index, parse_dates=dates)


def load_run(folder: Path) -> tuple[Config, dict[str, pd.DataFrame]]:
    """A saved run's settings and all its tables."""
    tables = {name: read_table(folder, name) for name in TABLE_INDEX_COLUMNS}
    return read_config(folder), tables
