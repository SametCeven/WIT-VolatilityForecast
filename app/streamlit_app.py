"""Streamlit app: BIST 100 volatility forecasts, backtests and saved runs.

Start it with `make app`. It reads the prices in data/raw (refresh them with `make data`
or the sidebar) and the saved runs in data/output/runs (made by `make pipeline` or the
sidebar's Run button).
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from volforecast import extract, pipeline
from volforecast.config import ALL_FEATURES, Config
from volforecast.transform import build_dataset, log_returns, realized_volatility

HORIZONS = [5, 10, 21, 63]
LEARNING_RATES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
METRICS = ["rmse", "mae", "qlike", "bias"]
# Words written differently in labels; every other word of a stored name stays as it is
LABEL_WORDS = {
    "vix": "VIX",
    "garch": "GARCH",
    "xgboost": "XGBoost",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "qlike": "QLIKE",
}
INPUTS_HELP = (
    "rv_1d, rv_5d, rv_21d, rv_63d: realized volatility over the last day, week, month and "
    "quarter. rv_down_21d: volatility of the last month's down days. vix_close: the last "
    "VIX close. rv_usdtry_21d: USD/TRY volatility over the last month. Its data starts in "
    "2005, so training starts there when it's chosen."
)
TABLE_LABELS = {
    "xu100": "BIST 100 daily prices",
    "vix": "VIX daily prices",
    "dataset": "Model dataset: target and inputs per date",
    "forecasts": "Forecasts in the test period",
    "scores": "Scores",
    "significance_tests": "Significance tests",
    "model_weights": "What the models learned",
    "latest_forecast": "Forecast from the latest close",
}

st.set_page_config(page_title="BIST 100 volatility", layout="wide")


def label(name: object) -> str:
    """A stored name as people read it, for example "vix_rescaled" -> "VIX rescaled"."""
    return " ".join(LABEL_WORDS.get(word, word) for word in str(name).split("_"))


@st.cache_data
def cached_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily BIST 100 and VIX history from data/raw."""
    return extract.load_prices("xu100"), extract.load_prices("vix")


@st.cache_data
def model_dataset(horizon: int) -> pd.DataFrame:
    """The target and the model inputs for each BIST date."""
    xu100, vix = cached_prices()
    returns = log_returns(xu100["close"]).dropna()
    usdtry_close = extract.load_prices("usdtry")["close"]
    return build_dataset(returns, vix["close"], horizon, usdtry_close)


@st.cache_data(show_spinner="Running every model, which can take up to a minute...")
def run_pipeline(config: Config) -> dict[str, pd.DataFrame]:
    """The pipeline's results for these settings, computed once per setting."""
    return pipeline.run(config)


def line_chart(frame: pd.DataFrame, title: str, log_y: bool = False) -> go.Figure:
    """One line per column, with dates on the x axis."""
    # SVG, not WebGL: browsers allow only about 16 WebGL charts per page
    fig = px.line(frame, log_y=log_y, title=title, render_mode="svg")
    fig.update_layout(
        xaxis_title=None, yaxis_title=None, legend_title=None, showlegend=frame.shape[1] > 1
    )
    return fig


def bar_chart(values: pd.Series, title: str, label_format: str = ".1f") -> go.Figure:
    """One horizontal bar per label, with its value printed on the bar."""
    frame = values.rename_axis("label").reset_index(name="value")
    fig = px.bar(frame, x="value", y="label", orientation="h", title=title)
    fig.update_traces(texttemplate=f"%{{x:{label_format}}}")
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    return fig


def for_display(table: pd.DataFrame) -> pd.DataFrame:
    """The table with its index as ordinary columns, and dates without a time of day."""
    flat = table.reset_index()
    for column, values in list(flat.items()):
        if pd.api.types.is_datetime64_any_dtype(values):
            flat[column] = values.dt.date
    return flat


def settings() -> tuple[Config, str | None]:
    """The sidebar settings as a Config, plus a problem to fix before running, if any."""
    defaults = Config()
    st.sidebar.subheader("Settings")
    train_year = st.sidebar.slider("Train from", 1997, 2020, int(defaults.train_start[:4]))
    compare_year = st.sidebar.slider(
        "Compare with training from", 1997, 2020, int(defaults.comparison_train_start[:4])
    )
    test_year = st.sidebar.slider("Test from", 2008, 2024, int(defaults.test_start[:4]))
    refit = st.sidebar.radio("Refit models", ["yearly", "once"], horizontal=True)
    horizon = st.sidebar.select_slider(
        "Horizon (trading days)", options=HORIZONS, value=defaults.horizon
    )
    chosen = st.sidebar.multiselect(
        "Model inputs", ALL_FEATURES, default=list(defaults.features), help=INPUTS_HELP
    )
    features = [feature for feature in ALL_FEATURES if feature in chosen]
    garch = st.sidebar.radio("GARCH errors", ["t", "normal"], horizontal=True)
    with st.sidebar.expander("XGBoost"):
        trees = st.slider("Trees", 50, 1000, defaults.xgboost_trees, step=50)
        depth = st.slider("Depth", 1, 8, defaults.xgboost_max_depth)
        rate = st.select_slider(
            "Learning rate", options=LEARNING_RATES, value=defaults.xgboost_learning_rate
        )
    config = Config(
        horizon=int(horizon),
        test_start=f"{test_year}-01-01",
        train_start=f"{train_year}-01-01",
        comparison_train_start=f"{compare_year}-01-01",
        refit="once" if refit == "once" else "yearly",
        features=tuple(features),
        garch_distribution="normal" if garch == "normal" else "t",
        xgboost_trees=trees,
        xgboost_max_depth=depth,
        xgboost_learning_rate=float(rate),
    )
    problem = None
    if not features:
        problem = "Choose at least one model input."
    elif max(train_year, compare_year) >= test_year:
        problem = "Both training starts have to come before the test start."
    return config, problem


def describe(config: Config) -> str:
    """The settings of a run in one line."""
    inputs = ", ".join(config.features)
    fits = "refitted every year" if config.refit == "yearly" else "fitted once"
    comparison = config.comparison_train_start[:4]
    return (
        f"trained from {config.train_start[:4]} (compared with {comparison}), "
        f"{fits}, tested from {config.test_start[:4]}, {config.horizon}-day horizon, "
        f"inputs {inputs}, GARCH with {config.garch_distribution} errors, "
        f"XGBoost with {config.xgboost_trees} trees of depth {config.xgboost_max_depth} "
        f"at learning rate {config.xgboost_learning_rate}"
    )


def run_overview(folders: list[Path]) -> pd.DataFrame:
    """One row per saved run: where its data ends, and every setting."""
    rows = {}
    for folder in folders:
        settings_row = asdict(pipeline.read_config(folder))
        settings_row["features"] = ", ".join(settings_row["features"])
        as_of = pipeline.read_table(folder, "latest_forecast")["as_of"].iloc[0]
        rows[folder.name] = {"data_as_of": as_of, **settings_row}
    return pd.DataFrame(rows).T.rename_axis("run")


def find_saved(config: Config, as_of: str) -> str | None:
    """The saved run with these settings and data up to the same date, if there is one."""
    for folder in pipeline.list_runs():
        same_data = pipeline.read_table(folder, "latest_forecast")["as_of"].iloc[0] == as_of
        if same_data and pipeline.read_config(folder) == config:
            return folder.name
    return None


def compare_runs(run_names: list[str], metric: str) -> pd.DataFrame:
    """One column per run: each model's score in that run's main training scenario."""
    columns = {}
    for run_name in run_names:
        folder = pipeline.RUNS_DIR / run_name
        main = pipeline.read_config(folder).main_scenario
        columns[run_name] = pipeline.read_table(folder, "scores").loc[main, metric]
    return pd.DataFrame(columns)


# Sidebar: the action buttons stay on top, so they're visible without scrolling
actions = st.sidebar.container()
config, problem = settings()
run_clicked = actions.button("Run", type="primary", disabled=problem is not None, width="stretch")
refresh_clicked = actions.button(
    "Refresh data", help="Download the latest prices from Yahoo Finance", width="stretch"
)
actions.caption("Every run is saved to data/output/runs and listed in the History tab.")
if problem:
    actions.warning(problem)

if refresh_clicked:
    with st.spinner("Downloading BIST 100, VIX and USD/TRY from Yahoo Finance..."):
        for ticker in extract.TICKERS:
            extract.download_to_csv(ticker)
    st.cache_data.clear()
    actions.success("Data refreshed. Press Run to update the results.")

if run_clicked:
    results = run_pipeline(config)
    as_of = results["latest_forecast"]["as_of"].iloc[0]
    run_name = find_saved(config, as_of) or pipeline.save_run(results, config).name
    st.session_state.current = (run_name, config, results)

# Until something is run, show a saved run
if "current" not in st.session_state:
    runs = pipeline.list_runs()
    if not runs:
        st.info("No saved runs yet. Press Run in the sidebar, or run `make pipeline`.")
        st.stop()
    # Open on the newest run with the default settings, so it matches the sidebar
    defaults = Config()
    folder = next((run for run in runs if pipeline.read_config(run) == defaults), runs[0])
    st.session_state.current = (folder.name, *pipeline.load_run(folder))

shown_name, shown, results = st.session_state.current

st.title("BIST 100 volatility forecast")
st.caption(f"Showing **{shown_name}**: {describe(shown)}")
if config != shown:
    st.info("The sidebar settings differ from the results shown. Press Run to see them.")

h = shown.horizon
xu100, vix = cached_prices()
returns = log_returns(xu100["close"]).dropna()
tabs = ["About", "Forecast", "Data", "Backtest", "Model", "Tables", "History"]
about_tab, forecast_tab, data_tab, backtest_tab, model_tab, tables_tab, history_tab = st.tabs(tabs)

with about_tab:
    st.markdown((Path(__file__).parent / "about.md").read_text(encoding="utf-8"))

with forecast_tab:
    latest = results["latest_forecast"]
    forecast = latest["forecast"]
    best = float(forecast["linear_regression"])
    recent = float(forecast["naive"])
    as_of = latest["as_of"].iloc[0]
    st.subheader(f"Volatility over the next {h} trading days, from the close of {as_of}")
    left, middle, right = st.columns(3)
    left.metric(
        "Linear regression forecast",
        f"{best:.1f}%",
        f"{best - recent:+.1f} vs last {h} days",
        delta_color="inverse",
    )
    middle.metric(f"Typical index move, {h} days", f"±{best * (h / 252) ** 0.5:.1f}%")
    right.metric(f"Realized, last {h} days", f"{recent:.1f}%")
    st.caption(
        "Volatility is annualized, in %. The typical move is one standard deviation of the "
        f"index's return over the next {h} trading days."
    )
    bars, context = st.columns([1, 2])
    by_model = forecast.sort_values().rename(index=label)
    bars.plotly_chart(bar_chart(by_model, "Every model's forecast"))
    recent_vol = realized_volatility(returns, h).dropna()
    start = recent_vol.index[-1] - pd.DateOffset(years=2)
    frame = recent_vol.loc[start:].rename(f"realized volatility, last {h} days").to_frame()
    fig = line_chart(frame, "The last two years, with the linear regression forecast")
    fig.add_hline(y=best, line_dash="dash", annotation_text="forecast")
    context.plotly_chart(fig)

with data_tab:
    st.plotly_chart(line_chart(xu100[["close"]], "BIST 100 close (log scale)", log_y=True))
    daily = returns.rename("log return").to_frame()
    st.plotly_chart(line_chart(daily, "BIST 100 daily log returns"))
    columns = {f"BIST 100, last {h} days": realized_volatility(returns, h), "VIX": vix["close"]}
    vols = pd.DataFrame(columns).dropna()
    st.plotly_chart(line_chart(vols, "Realized volatility vs VIX (annualized %)"))

with backtest_tab:
    forecasts = results["forecasts"]
    scenarios = list(forecasts.index.unique(level="scenario"))
    scenario = str(st.radio("Training start", scenarios, horizontal=True, format_func=label))
    models = [column for column in forecasts.columns if column != "target"]
    default = [model for model in ("garch", "linear_regression", "xgboost") if model in models]
    chosen = st.multiselect("Models", models, default=default, format_func=label)
    in_scenario = forecasts.index.get_level_values("scenario") == scenario
    frame = forecasts.loc[in_scenario].droplevel("scenario")[[*chosen, "target"]]
    frame = frame.rename(columns=label).rename(columns={"target": "realized"})
    fig = line_chart(frame, f"Next-{h}-day volatility: forecasts vs realized (%)")
    fig.update_traces(line_color="black", line_width=2, selector={"name": "realized"})
    st.plotly_chart(fig)
    st.subheader("Scores on the test period")
    scores = results["scores"].rename(index=label, columns=label)
    st.dataframe(scores.round(3), height="content")
    st.caption(
        "Bias, MAE and RMSE are in volatility points, QLIKE is on variances. Lower is "
        "better, and bias should be close to 0."
    )
    st.subheader("Is the linear regression's edge real?")
    tests = results["significance_tests"].rename(index=label, columns=label)
    st.dataframe(tests.round(3), height="content")
    st.caption(
        "Diebold–Mariano tests of each model against the linear regression, training from "
        f"{shown.train_start[:4]}. A positive statistic means the model has the higher loss, "
        "and p below 0.05 means the gap is unlikely to be luck."
    )

with model_tab:
    weights = results["model_weights"]
    coefficients = weights["linear_coefficient"]
    intercept = float(coefficients["intercept"])
    slopes = coefficients.drop("intercept")
    terms = " ".join(f"{value:+.2f}·{feature}" for feature, value in slopes.items())
    st.markdown(f"**Linear regression:** forecast = {intercept:.1f} {terms}")
    left, right = st.columns(2)
    left.plotly_chart(bar_chart(slopes, "Linear regression coefficients", ".2f"))
    importance = weights["xgboost_importance"].dropna()
    right.plotly_chart(bar_chart(importance, "XGBoost importance", ".2f"))
    st.caption(
        "Coefficients are forecast points per point of each input. Importance is each "
        "input's share of XGBoost's total."
    )

with tables_tab:
    inputs = {"xu100": xu100, "vix": vix, "dataset": model_dataset(h)}
    kind = st.radio("Show", ["Inputs", "Outputs"], horizontal=True)
    tables = inputs if kind == "Inputs" else results
    key = str(st.selectbox("Table", list(tables), format_func=lambda name: TABLE_LABELS[name]))
    table = tables[key]
    st.caption(f"{len(table):,} rows")
    st.dataframe(for_display(table), hide_index=True)
    st.download_button("Download as CSV", table.to_csv().encode(), f"{key}.csv", "text/csv")

with history_tab:
    runs = pipeline.list_runs()
    if not runs:
        st.info("No saved runs yet. Press Run in the sidebar, or run `make pipeline`.")
    else:
        st.dataframe(run_overview(runs))
        names = [folder.name for folder in runs]
        compared = st.multiselect("Compare runs", names, default=names[:3])
        metric = str(st.radio("Score", METRICS, horizontal=True, format_func=label))
        if compared:
            st.dataframe(compare_runs(compared, metric).rename(index=label).round(3))
            st.caption(f"{label(metric)} of each model in each run's main training scenario.")
        pick = str(st.selectbox("Show a saved run in the other tabs", names))
        if st.button("Show this run"):
            st.session_state.current = (pick, *pipeline.load_run(pipeline.RUNS_DIR / pick))
            st.rerun()
