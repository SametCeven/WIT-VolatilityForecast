"""Streamlit app: BIST 100 volatility forecasts, backtests and saved runs.

Start it with `make app`. The top bar has the general pages (About, Data, Compare) and the
Run page, whose tabs show one run. Prices are read from data/raw (refresh them with
`make data` or on the Data page) and runs from data/output/runs (made by `make pipeline` or
the Run button).
"""

from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.io.formats.style import Styler

from volforecast import extract, pipeline
from volforecast.config import ALL_FEATURES, HORIZON, Config
from volforecast.transform import build_dataset, log_returns, realized_volatility

HORIZON_CHOICES = [5, 10, 21, 63]
LEARNING_RATE_CHOICES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
METRICS = ["mae", "rmse", "qlike", "bias"]
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
# How the sidebar names each setting, in its order; Compare names them the same way
SETTING_LABELS = {
    "train_start": "Train from",
    "comparison_train_start": "Compare with training from",
    "test_start": "Test from",
    "refit": "Refit models",
    "horizon": "Horizon (trading days)",
    "features": "Model inputs",
    "garch_distribution": "GARCH errors",
    "xgboost_trees": "XGBoost trees",
    "xgboost_max_depth": "XGBoost tree depth",
    "xgboost_learning_rate": "XGBoost learning rate",
}
# What each model input is; all are annualized % and known at BIST's close
FEATURE_HELP = {
    "rv_1d": "Realized volatility of the last trading day.",
    "rv_5d": "Realized volatility of the last 5 trading days, about a week.",
    "rv_21d": "Realized volatility of the last 21 trading days, about a month.",
    "rv_63d": "Realized volatility of the last 63 trading days, about a quarter.",
    "rv_down_21d": "Realized volatility of the last 21 trading days, from the down days only.",
    "vix_close": "The last VIX close before BIST's close, since the US market closes later.",
    "rv_usdtry_21d": (
        "USD/TRY realized volatility over the last 21 trading days. Its data starts in 2005, "
        "so training starts there when it's chosen."
    ),
}
# Tooltips for the settings that need one
SETTING_HELP = {
    "comparison_train_start": "Scored next to the main training start on the Backtest tab.",
    "refit": "Yearly: refit each January on all years before. Once: fit only before the test.",
    "horizon": "How many trading days ahead to forecast. 21 is about a month.",
    "features": "\n\n".join(f"**{name}**: {text}" for name, text in FEATURE_HELP.items()),
    "garch_distribution": "t allows for the fat tails of BIST returns; normal doesn't.",
}
# The Data page's daily table: its columns in order, with a tooltip each
DAILY_HELP = {
    "bist_close": "The BIST 100 at the day's close.",
    "bist_return": "The day's log return in %: ln(close / previous close) × 100.",
    **FEATURE_HELP,
    "target": (
        f"Realized volatility over the next {HORIZON} trading days: what the forecasters "
        f"predict. Empty for the newest {HORIZON} days, which haven't happened yet."
    ),
}
METRIC_HELP = {
    "mae": "The average miss, in volatility points. Lower is better.",
    "rmse": "Like MAE, but big misses count extra. Lower is better.",
    "qlike": "Punishes forecasting too low more than too high. Lower is better.",
    "bias": "Average forecast minus average outcome. Closest to 0 is best.",
}
# The score each Diebold–Mariano loss is behind, to name the tests by
TEST_LOSSES = {"mae": "mae", "mse": "rmse", "qlike": "qlike"}
TABLE_LABELS = {
    "dataset": "Model dataset: target and inputs per date",
    "forecasts": "Forecasts in the test period",
    "scores": "Scores",
    "significance_tests": "Significance tests",
    "model_weights": "What the models learned",
    "latest_forecast": "Forecast from the latest close",
}
# One color per model in every chart, from Streamlit's palette, as on the About page
MODEL_COLORS = {
    "historical_average": "#a3a8b8",
    "naive": "#ffa421",
    "vix_rescaled": "#803df5",
    "garch": "#1c83e1",
    "linear_regression": "#ff4b4b",
    "xgboost": "#21c354",
}
BEST_STYLE = "background-color: rgba(33, 195, 84, 0.2)"
DIFFERENCE_STYLE = "background-color: rgba(255, 164, 33, 0.2)"
# Less space above every page's content: the top bar is 60px high, so 5rem (80px) leaves a gap
PAGE_STYLE = """
<style>
[data-testid="stMain"] [data-testid="stMainBlockContainer"] { padding-top: 5rem; }
</style>
"""

# Tighter headings and model sections on the Compare page, so the settings and a few open
# models fit on one screen
COMPARE_STYLE = """
<style>
.st-key-comparison h3 { padding: 0.25rem 0 0; }
.st-key-model_results summary { padding: 0 8px; min-height: 28px; }
.st-key-model_results [data-testid="stExpanderDetails"] { padding: 0 8px 4px; }
</style>
"""

st.set_page_config(
    page_title="BIST 100 volatility", page_icon=":material/show_chart:", layout="wide"
)
st.html(PAGE_STYLE)


class SavedRun(NamedTuple):
    """A run as the Run page shows it: its folder name, settings and result tables."""

    name: str
    config: Config
    results: dict[str, pd.DataFrame]


def label(name: object) -> str:
    """A stored name as people read it, for example "vix_rescaled" -> "VIX rescaled"."""
    return " ".join(LABEL_WORDS.get(word, word) for word in str(name).split("_"))


def run_label(run_name: str) -> str:
    """A run folder's name as a date and time, for example "2026-10-17 19:03:11"."""
    day, time = run_name.split("_")
    return f"{day} {time.replace('-', ':')}"


@st.cache_data
def load_all_prices() -> dict[str, pd.DataFrame]:
    """Daily prices of every ticker, from data/raw."""
    return {ticker: extract.load_prices(ticker) for ticker in extract.TICKERS}


def bist_returns() -> pd.Series:
    """Daily BIST 100 log returns."""
    return log_returns(load_all_prices()["xu100"]["close"]).dropna()


@st.cache_data
def model_dataset(horizon: int) -> pd.DataFrame:
    """The target and every model input, per BIST date."""
    prices = load_all_prices()
    vix_close, usdtry_close = prices["vix"]["close"], prices["usdtry"]["close"]
    return build_dataset(bist_returns(), vix_close, horizon, usdtry_close)


@st.cache_data
def daily_table() -> pd.DataFrame:
    """Per BIST date: the close, the log return, every model input and the target."""
    columns = {"bist_close": load_all_prices()["xu100"]["close"], "bist_return": bist_returns()}
    table = pd.DataFrame(columns).join(model_dataset(HORIZON), how="inner")
    table["bist_return"] *= 100
    return table[list(DAILY_HELP)]


@st.cache_data(show_spinner="Running every model, which can take up to a minute...")
def run_pipeline(config: Config) -> dict[str, pd.DataFrame]:
    """The pipeline's results for these settings, computed once per setting."""
    return pipeline.run(config)


def data_as_of(folder: Path) -> str:
    """The date of the last close in a saved run's data."""
    return str(pipeline.read_table(folder, "latest_forecast")["as_of"].iloc[0])


def open_run(run_name: str) -> SavedRun:
    """A saved run, read from its folder."""
    config, results = pipeline.load_run(pipeline.RUNS_DIR / run_name)
    return SavedRun(run_name, config, results)


def find_saved_run(config: Config, as_of: str) -> str | None:
    """The saved run with these settings and data up to the same date, if there is one."""
    for folder in pipeline.list_runs():
        if data_as_of(folder) == as_of and pipeline.read_config(folder) == config:
            return folder.name
    return None


def shown_run() -> SavedRun | None:
    """The run on the Run page: the last one run or opened in this session.

    Until then, it's the newest saved run with the default settings.
    """
    if "shown_run" not in st.session_state:
        runs = pipeline.list_runs()
        if not runs:
            return None
        defaults = Config()
        folder = next((run for run in runs if pipeline.read_config(run) == defaults), runs[0])
        st.session_state.shown_run = open_run(folder.name)
    return st.session_state.shown_run


def chart_colors() -> dict[str, str]:
    """Each model's color by its label, and black (white on a dark theme) for what happened."""
    colors = {label(model): color for model, color in MODEL_COLORS.items()}
    colors["realized"] = "white" if st.context.theme.type == "dark" else "black"
    return colors


def line_chart(
    frame: pd.DataFrame, title: str, log_y: bool = False, value_format: str = ".1f"
) -> go.Figure:
    """One line per column, with dates on the x axis and each model in its own color."""
    colors = chart_colors()
    # Only models get fixed colors; other lines keep the theme's colors
    fixed_colors = {column: colors[column] for column in frame.columns if column in colors}
    # SVG, not WebGL: browsers allow only about 16 WebGL charts per page
    fig = px.line(
        frame, log_y=log_y, title=title, render_mode="svg", color_discrete_map=fixed_colors or None
    )
    fig.update_traces(line_width=2, selector={"name": "realized"})
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        legend_title=None,
        showlegend=frame.shape[1] > 1,
        hovermode="x unified",
    )
    fig.update_yaxes(hoverformat=value_format)
    return fig


def bar_chart(
    values: pd.Series, title: str, value_format: str = ".1f", colors: dict[str, str] | None = None
) -> go.Figure:
    """One horizontal bar per label, with its value printed on the bar."""
    frame = values.rename_axis("label").reset_index(name="value")
    fig = px.bar(
        frame,
        x="value",
        y="label",
        color="label" if colors else None,
        color_discrete_map=colors,
        orientation="h",
        title=title,
    )
    fig.update_traces(texttemplate=f"%{{x:{value_format}}}")
    fig.update_layout(xaxis_title=None, yaxis_title=None, showlegend=False)
    return fig


def forecast_chart(horizon: int, forecast: float) -> go.Figure:
    """The past year's realized volatility, and the forecast drawn over the coming days."""
    realized = realized_volatility(bist_returns(), horizon).dropna()
    last_date = realized.index[-1]
    start = last_date - pd.DateOffset(years=1)
    frame = realized.loc[start:].rename("realized").to_frame()
    title = f"The past year's volatility, and the forecast for the next {horizon} days (%)"
    fig = line_chart(frame, title)
    days = pd.bdate_range(last_date, periods=horizon + 1)
    line = {"color": MODEL_COLORS["linear_regression"], "dash": "dash", "width": 3}
    fig.add_scatter(
        x=[days[1], days[-1]], y=[forecast, forecast], mode="lines", name="forecast", line=line
    )
    fig.update_layout(showlegend=True)
    return fig


def mark_best(values: pd.Series, closest_to_zero: bool = False) -> list[str]:
    """Highlights the best value: the lowest, or the one closest to 0."""
    distance = values.abs() if closest_to_zero else values
    return [BEST_STYLE if value == distance.min() else "" for value in distance]


def mark_differences(values: pd.Series) -> list[str]:
    """Highlights a whole row when the runs don't all have the same value."""
    style = DIFFERENCE_STYLE if values.nunique() > 1 else ""
    return [style] * len(values)


def for_display(table: pd.DataFrame) -> pd.DataFrame:
    """The table with its index as ordinary columns, and dates without a time of day."""
    flat = table.reset_index()
    for column, values in list(flat.items()):
        if pd.api.types.is_datetime64_any_dtype(values):
            flat[column] = values.dt.date
    return flat


def verdict(stat: float, p_value: float) -> str:
    """One Diebold–Mariano test in words, from the side of the model tested."""
    # Three significant digits, so a p just below 0.05 doesn't show as 0.050
    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3g}"
    if p_value >= 0.05:
        return f"no clear difference ({p_text})"
    return f"{'worse' if stat > 0 else 'better'} ({p_text})"


def test_verdicts(tests: pd.DataFrame) -> pd.DataFrame:
    """Each model against the linear regression, per score, in words."""
    columns = {}
    for loss, metric in TEST_LOSSES.items():
        pairs = zip(tests[f"{loss}_stat"], tests[f"{loss}_p"], strict=True)
        columns[label(metric)] = [verdict(stat, p_value) for stat, p_value in pairs]
    return pd.DataFrame(columns, index=tests.index.map(label))


def settings_table(folders: list[Path]) -> pd.DataFrame:
    """One column per run and one row per setting, named as in the sidebar."""
    rows = {"data_as_of": "Data up to", **SETTING_LABELS}
    columns = {}
    for folder in folders:
        config = pipeline.read_config(folder)
        values = {name: str(value) for name, value in asdict(config).items()}
        values["features"] = ", ".join(config.features)
        for name in ("train_start", "comparison_train_start", "test_start"):
            values[name] = values[name][:4]
        values["data_as_of"] = data_as_of(folder)
        columns[run_label(folder.name)] = values
    return pd.DataFrame(columns).reindex(list(rows)).rename(index=rows)


def results_table(folders: list[Path]) -> pd.DataFrame:
    """One column per run, one row per model and result: the scores and the latest forecast.

    The scores are the ones with each run's main training start.
    """
    columns = {}
    for folder in folders:
        main = pipeline.read_config(folder).main_scenario
        scores = pipeline.read_table(folder, "scores")
        in_main = scores.index.get_level_values("scenario") == main
        table = scores.loc[in_main].droplevel("scenario")[METRICS]
        table["latest_forecast"] = pipeline.read_table(folder, "latest_forecast")["forecast"]
        columns[run_label(folder.name)] = table.stack()
    return pd.DataFrame(columns).rename_axis(["model", "result"])


def model_results(results: pd.DataFrame, model: str) -> Styler:
    """One model's results, one column per run, with its best run on each score in green."""
    in_model = results.index.get_level_values("model") == model
    table = results[in_model].droplevel("model").rename_axis(None)
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    text = table.astype(object)
    for result, values in table.iterrows():
        # Only scores have a best run: what follows the latest forecast isn't known yet
        if result in METRICS and len(table.columns) > 1:
            styles.loc[result] = mark_best(values, closest_to_zero=result == "bias")
        digits = 1 if result == "latest_forecast" else 3
        text.loc[result] = [f"{value:.{digits}f}" for value in values]
    return text.rename(index=label).style.apply(lambda _: styles.rename(index=label), axis=None)


def compact(styled: Styler) -> Styler:
    """Shorter rows than st.table's default, and the same column widths in every table."""
    return styled.set_table_styles(
        [
            {"selector": "", "props": "table-layout: fixed"},
            {"selector": "th:first-child", "props": "width: 25%"},
            # Both the cells and the text inside them set the row height
            {"selector": "td", "props": "padding: 1px 6px; font-size: 13px; line-height: 1.25"},
            {"selector": "th", "props": "padding: 1px 6px; font-size: 13px; line-height: 1.25"},
            {"selector": "p", "props": "font-size: 13px; line-height: 1.25"},
        ]
    )


def sidebar_settings(start: Config) -> Config | None:
    """The settings form in the sidebar, filled in with `start`.

    Returns the new settings when Run is pressed and they work together, otherwise None.
    """
    names = SETTING_LABELS
    refit_choices, garch_choices = ["yearly", "once"], ["t", "normal"]
    with st.sidebar.form("settings", border=False):
        run_pressed = st.form_submit_button("Run", type="primary", width="stretch")
        train_year = st.slider(names["train_start"], 1997, 2020, int(start.train_start[:4]))
        compare_year = st.slider(
            names["comparison_train_start"],
            1997,
            2020,
            int(start.comparison_train_start[:4]),
            help=SETTING_HELP["comparison_train_start"],
        )
        test_year = st.slider(names["test_start"], 2008, 2024, int(start.test_start[:4]))
        refit = st.radio(
            names["refit"],
            refit_choices,
            index=refit_choices.index(start.refit),
            horizontal=True,
            help=SETTING_HELP["refit"],
        )
        horizon = st.select_slider(
            names["horizon"], HORIZON_CHOICES, start.horizon, help=SETTING_HELP["horizon"]
        )
        chosen = st.multiselect(
            names["features"], ALL_FEATURES, list(start.features), help=SETTING_HELP["features"]
        )
        garch = st.radio(
            names["garch_distribution"],
            garch_choices,
            index=garch_choices.index(start.garch_distribution),
            horizontal=True,
            help=SETTING_HELP["garch_distribution"],
        )
        with st.expander("XGBoost"):
            trees = st.slider(names["xgboost_trees"], 50, 1000, start.xgboost_trees, step=50)
            depth = st.slider(names["xgboost_max_depth"], 1, 8, start.xgboost_max_depth)
            rate = st.select_slider(
                names["xgboost_learning_rate"], LEARNING_RATE_CHOICES, start.xgboost_learning_rate
            )
    if not run_pressed:
        return None
    features = tuple(feature for feature in ALL_FEATURES if feature in chosen)
    problem = None
    if not features:
        problem = "Choose at least one model input."
    elif max(train_year, compare_year) >= test_year:
        problem = "Both training starts have to come before the test start."
    if problem:
        st.toast(problem, icon=":material/warning:")
        return None
    return Config(
        horizon=int(horizon),
        test_start=f"{test_year}-01-01",
        train_start=f"{train_year}-01-01",
        comparison_train_start=f"{compare_year}-01-01",
        refit="once" if refit == "once" else "yearly",
        features=features,
        garch_distribution="normal" if garch == "normal" else "t",
        xgboost_trees=trees,
        xgboost_max_depth=depth,
        xgboost_learning_rate=float(rate),
    )


def show_forecast(shown: SavedRun) -> None:
    """The Forecast tab: every model's forecast from the latest close."""
    horizon = shown.config.horizon
    forecast = shown.results["latest_forecast"]["forecast"]
    linear_forecast = float(forecast["linear_regression"])
    recent_volatility = float(forecast["naive"])
    st.subheader(f"Volatility over the next {horizon} trading days")
    left, middle, right = st.columns(3)
    left.metric(
        "Linear regression forecast",
        f"{linear_forecast:.1f}%",
        f"{linear_forecast - recent_volatility:+.1f} vs the last {horizon} days",
        delta_color="inverse",
        border=True,
    )
    move = linear_forecast * (horizon / 252) ** 0.5
    middle.metric(
        f"Typical move of the index over {horizon} days",
        f"±{move:.1f}%",
        help="About two times in three, the index ends the period within this range.",
        border=True,
    )
    right.metric(f"Realized, last {horizon} days", f"{recent_volatility:.1f}%", border=True)
    chart_column, bars_column = st.columns([3, 2])
    chart_column.plotly_chart(forecast_chart(horizon, linear_forecast))
    by_model = forecast.sort_values().rename(index=label)
    bars = bar_chart(by_model, "Every model's forecast (%)", colors=chart_colors())
    bars_column.plotly_chart(bars)


def show_backtest(shown: SavedRun) -> None:
    """The Backtest tab: forecasts against what happened, scores and significance tests."""
    config, results = shown.config, shown.results
    forecasts = results["forecasts"]
    scenarios = list(forecasts.index.unique(level="scenario"))
    main = config.main_scenario
    scenario = st.segmented_control(
        "Training start", scenarios, default=main, required=True, format_func=label
    )
    models = [column for column in forecasts.columns if column != "target"]
    default = [model for model in ("garch", "linear_regression", "xgboost") if model in models]
    chosen = st.multiselect("Models", models, default=default, format_func=label)
    in_scenario = forecasts.index.get_level_values("scenario") == scenario
    frame = forecasts.loc[in_scenario].droplevel("scenario")[[*chosen, "target"]]
    frame = frame.rename(columns=label).rename(columns={"target": "realized"})
    title = f"Forecasts vs realized volatility over the next {config.horizon} days (%)"
    st.plotly_chart(line_chart(frame, title))

    st.subheader("Scores on the test period")
    scores = results["scores"]
    in_scenario = scores.index.get_level_values("scenario") == scenario
    table = scores.loc[in_scenario].droplevel("scenario")[METRICS].rename(index=label)
    styled = table.style.apply(mark_best, subset=["mae", "rmse", "qlike"])
    styled = styled.apply(mark_best, subset=["bias"], closest_to_zero=True).format(precision=3)
    columns = {
        metric: st.column_config.Column(label(metric), help=METRIC_HELP[metric])
        for metric in METRICS
    }
    st.dataframe(styled, column_config=columns, height="content")

    test_help = (
        f"Diebold–Mariano tests, training from {config.train_start[:4]}. Each model's misses "
        "are compared with the linear regression's; with p below 0.05, the gap is unlikely "
        "to be luck."
    )
    st.subheader("Compared with the linear regression", help=test_help)
    st.dataframe(test_verdicts(results["significance_tests"]), height="content")


def show_model(shown: SavedRun) -> None:
    """The Model tab: what the linear regression and XGBoost learned from all the data."""
    weights = shown.results["model_weights"]
    coefficients = weights["linear_coefficient"]
    intercept = float(coefficients["intercept"])
    slopes = coefficients.drop("intercept")
    terms = " ".join(f"{value:+.2f}·{feature}" for feature, value in slopes.items())
    st.markdown(f"**Linear regression:** forecast = {intercept:.1f} {terms}")
    linear_chart = bar_chart(slopes, "Linear regression: weight of each input", ".2f")
    linear_chart.update_traces(marker_color=MODEL_COLORS["linear_regression"])
    importance = weights["xgboost_importance"].dropna()
    xgboost_chart = bar_chart(importance, "XGBoost: share of each input", ".2f")
    xgboost_chart.update_traces(marker_color=MODEL_COLORS["xgboost"])
    left, right = st.columns(2)
    left.plotly_chart(linear_chart)
    right.plotly_chart(xgboost_chart)


def show_tables(shown: SavedRun) -> None:
    """The Tables tab: the run's dataset and every result table, to look at or download."""
    tables = {"dataset": model_dataset(shown.config.horizon), **shown.results}
    name = str(st.selectbox("Table", list(tables), format_func=lambda key: TABLE_LABELS[key]))
    table = tables[name]
    st.dataframe(for_display(table), hide_index=True)
    st.download_button("Download as CSV", table.to_csv().encode(), f"{name}.csv", "text/csv")


def run_header(shown: SavedRun) -> SavedRun:
    """A picker of every saved run, starting at the shown one, with badges for the one picked.

    Returns the run to show.
    """
    names = [folder.name for folder in pipeline.list_runs()]
    index = names.index(shown.name) if shown.name in names else None
    picker_column, badges_column = st.columns([1, 2], vertical_alignment="bottom")
    picked = picker_column.selectbox("Run", names, index=index, format_func=run_label)
    if picked is not None and picked != shown.name:
        shown = open_run(picked)
        st.session_state.shown_run = shown
    as_of = shown.results["latest_forecast"]["as_of"].iloc[0]
    settings = "Default settings" if shown.config == Config() else "Custom settings"
    badges_column.markdown(f":gray-badge[Data up to {as_of}] :gray-badge[{settings}]")
    return shown


def run_page() -> None:
    """One run: pick a saved one or make a new one in the sidebar, and see it in tabs."""
    st.title("BIST 100 volatility forecast")
    shown = shown_run()
    # The picker comes before the sidebar, so the sidebar shows the picked run's settings
    if shown is not None:
        shown = run_header(shown)
    config = sidebar_settings(shown.config if shown else Config())
    if config is not None:
        results = run_pipeline(config)
        as_of = str(results["latest_forecast"]["as_of"].iloc[0])
        run_name = find_saved_run(config, as_of) or pipeline.save_run(results, config).name
        st.session_state.shown_run = SavedRun(run_name, config, results)
        # Start over, so the picker and the sidebar show the new run
        st.rerun()
    if shown is None:
        st.info("No saved runs yet. Press Run in the sidebar.")
        return

    forecast_tab, backtest_tab, model_tab, tables_tab = st.tabs(
        ["Forecast", "Backtest", "Model", "Tables"]
    )
    with forecast_tab:
        show_forecast(shown)
    with backtest_tab:
        show_backtest(shown)
    with model_tab:
        show_model(shown)
    with tables_tab:
        show_tables(shown)


def data_page() -> None:
    """The prices behind every run: charts, the daily table, and a button to refresh them."""
    title_column, button_column = st.columns([4, 1], vertical_alignment="bottom")
    title_column.title("Data")
    refresh = button_column.button(
        "Refresh data",
        icon=":material/refresh:",
        help="Download the latest prices from Yahoo Finance.",
        width="stretch",
    )
    if refresh:
        with st.spinner("Downloading BIST 100, VIX and USD/TRY from Yahoo Finance..."):
            for ticker in extract.TICKERS:
                extract.download_to_csv(ticker)
        st.cache_data.clear()
        st.success("Data refreshed. Press Run on the Run page to update the forecasts.")
    prices = load_all_prices()
    returns = bist_returns()
    last_date = f"{prices['xu100'].index[-1]:%Y-%m-%d}"
    st.markdown(f":gray-badge[Daily closes from Yahoo Finance] :gray-badge[Up to {last_date}]")

    scale = st.segmented_control(
        "Scale",
        ["Log scale", "Regular scale"],
        default="Log scale",
        required=True,
        label_visibility="collapsed",
    )
    close = prices["xu100"][["close"]]
    log_y = scale == "Log scale"
    st.plotly_chart(line_chart(close, "BIST 100 close", log_y=log_y, value_format=",.0f"))
    daily = (returns * 100).rename("log return").to_frame()
    st.plotly_chart(line_chart(daily, "BIST 100 daily log returns (%)", value_format=".2f"))
    columns = {
        f"BIST 100, last {HORIZON} days": realized_volatility(returns, HORIZON),
        "VIX close": prices["vix"]["close"],
    }
    volatility = pd.DataFrame(columns).dropna()
    title = "BIST 100 realized volatility vs VIX (annualized %)"
    st.plotly_chart(line_chart(volatility, title))

    st.subheader("Daily data", help="Hover over a column name to see what it is.")
    table = daily_table()
    table_columns = {
        name: st.column_config.NumberColumn(help=text, format="%.2f")
        for name, text in DAILY_HELP.items()
    }
    st.dataframe(for_display(table), hide_index=True, column_config=table_columns)
    st.download_button("Download as CSV", table.to_csv().encode(), "daily_data.csv", "text/csv")


def compare_page() -> None:
    """Saved runs side by side: what each one was set to, and what it got."""
    st.title("Compare runs")
    runs = pipeline.list_runs()
    if not runs:
        st.info("No saved runs yet. Press Run on the Run page, or run `make pipeline`.")
        return
    names = [folder.name for folder in runs]
    compared = st.multiselect("Runs to compare", names, default=names[:3], format_func=run_label)
    if not compared:
        return
    folders = [pipeline.RUNS_DIR / name for name in compared]
    results_help = (
        "Green: each model's best run on a score, with the run's main training start. The "
        "latest forecast is each model's forecast from the latest close: not a score, since "
        "what happens next isn't known yet."
    )
    st.html(COMPARE_STYLE)
    # Static tables: unlike st.dataframe, their column widths can match from table to table
    with st.container(key="comparison"):
        st.subheader("Settings", help="Amber: a setting that differs between the runs.")
        settings = settings_table(folders).style.apply(mark_differences, axis=1)
        st.table(compact(settings))
        st.subheader("Results", help=results_help)
        results = results_table(folders)
        with st.container(key="model_results", gap="xxsmall"):
            for model in results.index.unique(level="model"):
                with st.expander(label(model)):
                    st.table(compact(model_results(results, str(model))))


pages = [
    st.Page(Path(__file__).parent / "about.py", title="About", icon=":material/info:"),
    st.Page(data_page, title="Data", icon=":material/database:", url_path="data"),
    st.Page(run_page, title="Run", icon=":material/query_stats:", url_path="run"),
    st.Page(compare_page, title="Compare", icon=":material/compare_arrows:", url_path="compare"),
]
st.navigation(pages, position="top").run()
