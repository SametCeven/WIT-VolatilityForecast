"""The About page: the question, how the forecasts are made and tested, and what we found.

The Approach section follows the notebooks (01_eda, 02_baselines, 03_models, 04_inputs). Its
numbers are fixed, from the notebooks and the base run with data up to 2026-10-16; the Run
and Compare pages show the saved runs.
"""

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

from volforecast.config import HORIZON, Config
from volforecast.extract import TICKERS, load_prices
from volforecast.transform import log_returns, realized_volatility

# Each card is a title (with an icon or a color), a short text and any LaTeX formulas
SUMMARY = [
    (
        ":material/help: **The question**",
        "How far will the BIST 100 swing over the next 21 trading days, about a month? Not "
        "which way it goes, only how much it moves. The Run page can look 5, 10 or 63 days "
        "ahead instead.",
    ),
    (
        ":material/model_training: **The approach**",
        "Six forecasters, from simple rules to machine learning, learn from 2003–2017. Then "
        "they forecast every day of 2018–2026, and each forecast is checked against what "
        "actually happened.",
    ),
    (
        ":material/emoji_events: **The answer**",
        "A simple linear regression does best: on average, it misses by 6.8 points when "
        "volatility is around 25%. But the tests can't tell it apart from GARCH or the "
        "long-run average: next month's swings are only a little predictable.",
    ),
]
DATA_CONCEPTS = [
    (
        ":material/account_balance: **BIST 100**",
        "The index of the 100 largest companies on Borsa Istanbul, Turkey's stock exchange. "
        "When the news says the Turkish market rose 2% today, it usually means this index. "
        "Bigger companies weigh more: with p a company's price and q its shares that trade "
        "freely,",
        r"\text{BIST 100} \propto \sum_{i=1}^{100} p_i \, q_i",
    ),
    (
        ":material/swap_vert: **Daily return**",
        "How much the index moved in a day, in %. The app uses log returns: on normal days "
        "they're almost the same as % changes, and they add up neatly over many days. With "
        "$P_t$ the close on day $t$:",
        r"r_t = 100 \times \ln \frac{P_t}{P_{t-1}}",
    ),
    (
        ":material/stacked_line_chart: **Log scale**",
        "On a log scale, equal steps are equal % moves. The BIST 100 rose from under 20 in "
        "1997 to over 10,000, mostly through inflation, so on a regular scale the early years "
        "look flat. A doubling looks the same size everywhere:",
        r"\log \frac{200}{100} = \log \frac{10{,}000}{5{,}000}",
    ),
]
TABLE_NAMES = {"xu100": "BIST 100", "vix": "VIX", "usdtry": "USD/TRY"}
# The columns every downloaded price table has
RAW_COLUMNS = pd.DataFrame(
    {
        "what it holds": [
            "the trading day",
            "the first price of the day",
            "the highest price of the day",
            "the lowest price of the day",
            "the last price of the day: the only column the forecasts use",
            "how much was traded (0 for the VIX and USD/TRY)",
        ],
    },
    index=["date", "open", "high", "low", "close", "volume"],
)
VOLATILITY_CONCEPTS = [
    (
        ":material/ssid_chart: **Volatility**",
        "How much a price swings, up or down: the standard deviation σ of its returns. A calm "
        "market has a low σ, a nervous one a high σ. Nobody sees σ directly: it can only be "
        "measured afterwards, expected by traders, or forecast.",
        r"\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (r_i - \bar{r})^2}",
    ),
    (
        ":material/history: **Realized volatility**",
        "The volatility that actually happened, measured from the daily returns of the last "
        "n trading days. The average return is left out, because it's close to 0 for a "
        "day, and the result is annualized. rv_21d, for example, is RV over the last 21 days:",
        r"\text{RV}_n = \sqrt{\frac{252}{n} \sum_{i=1}^{n} r_i^2}",
    ),
    (
        ":material/percent: **Annualized, in %**",
        "Volatility grows with the square root of time, so it's scaled to a year to compare "
        "periods of any length. At 25%, the index typically moves about ±25% in a year and "
        "±7% in 21 trading days:",
        r"\sigma_{\text{year}} = \sqrt{252}\,\sigma_{\text{day}}",
        r"\sigma_{h \text{ days}} = \sqrt{h / 252}\,\sigma_{\text{year}}",
    ),
    (
        ":material/calendar_month: **Target and horizon**",
        "The target is the realized volatility of the h trading days after the forecast day, "
        "so it's only known h days later. h is 21 trading days, about a month: the window "
        "the VIX covers, and where recent volatility starts to track the target well "
        "(correlation 0.65 over 1997–2026, against 0.25 one day ahead).",
        r"\text{target}_t = \text{RV of days } t+1, \dots, t+h",
    ),
]
DATA_FINDINGS = [
    (
        ":material/warning: **Two eras**",
        "In 1997–2002, volatility was often 40–80% and peaked near 120%. From 2003 it mostly "
        "stays at 15–40%, with crisis spikes to 50–75%. The forecasters learn from 2003 on, "
        "because the crisis years make them all forecast too high.",
    ),
    (
        ":material/bolt: **Extreme days are common**",
        "Daily returns have a standard deviation of 2.2% (about 35% a year) and an excess "
        "kurtosis of 7.3: very large days happen far more often than a bell curve predicts.",
    ),
    (
        ":material/waves: **Storms come in streaks**",
        "The size of today's move is linked to the size of moves weeks ago (correlation 0.25 "
        "a day apart, still 0.14 after 60 days), while their direction isn't. So recent "
        "volatility helps forecast, but storms fade back toward the usual level.",
    ),
]
# The colors match each forecaster's lines and bars on the Run page
FORECASTERS = [
    (
        ":gray[**Historical average**] :gray-badge[baseline]",
        "Always forecasts the average volatility of the training years. It ignores what's "
        "happening now.",
        r"f = \overline{\text{target}}_{\text{training years}}",
    ),
    (
        ":orange[**Naive**] :gray-badge[baseline]",
        "Next month will be like the last one: the forecast is the realized volatility of "
        "the last 21 trading days.",
        r"f_t = \text{rv\_21d}_t",
    ),
    (
        ":violet[**VIX rescaled**] :gray-badge[baseline]",
        "The VIX is Wall Street's “fear index”: the volatility traders expect for the US "
        "market over the next 30 days, read from option prices. When global investors get "
        "nervous it jumps, and Turkey often follows. It measures US volatility, so a and b, "
        "fitted on the training years, rescale it to BIST levels.",
        r"f_t = a + b \times \text{vix\_close}_t",
    ),
    (
        ":blue[**GARCH**] :gray-badge[baseline]",
        "The classic volatility model. Tomorrow's variance σ² (volatility squared) is a "
        "constant plus a share α of today's squared return and a share β of today's "
        "variance, so after a shock it forecasts high volatility that fades back to normal. "
        "Its errors follow a Student-t distribution, which allows for BIST's extreme days.",
        r"\sigma^2_{t+1} = \omega + \alpha\, r_t^2 + \beta\, \sigma^2_t",
    ),
    (
        ":red[**Linear regression**] :gray-badge[model]",
        "A weighted sum of four inputs x: rv_1d, rv_5d, rv_21d and vix_close. Short windows "
        "react fast and the long one pulls toward normal levels, the classic HAR model of "
        "volatility. The weights b are the ones that fit the training years best.",
        r"f = b_0 + \sum_{j=1}^{4} b_j \, x_j",
    ),
    (
        ":green[**XGBoost**] :gray-badge[model]",
        "Machine learning: K small decision trees, each a few if-then questions about the same "
        "four inputs. Each tree corrects the ones before it, scaled by the learning rate η. "
        "The trees are kept small and slow (300 trees of depth 3, η = 0.05), because there "
        "are only about 3,700 training days.",
        r"f = f_0 + \eta \sum_{k=1}^{K} \text{tree}_k(x)",
    ),
]
SCORES = [
    (
        "**MAE**",
        "The average miss, in volatility points. A forecast of 25% when 30% happens misses "
        "by 5 points.",
        r"\text{MAE} = \frac{1}{N} \sum_t |f_t - y_t|",
    ),
    (
        "**RMSE**",
        "Like MAE, but big misses count extra, so a forecaster that is sometimes far off "
        "scores worse.",
        r"\text{RMSE} = \sqrt{\frac{1}{N} \sum_t (f_t - y_t)^2}",
    ),
    (
        "**QLIKE**",
        "The standard score for volatility forecasts, on variances. It punishes forecasting "
        "too low, being caught off guard, more than forecasting too high.",
        r"\text{QLIKE} = \tfrac{1}{N} \sum_t (q_t - \ln q_t - 1)",
        r"q_t = y_t^2 / f_t^2",
    ),
    (
        "**Bias**",
        "The average forecast minus the average outcome. Positive means forecasting too "
        "high. Closer to 0 is better.",
        r"\text{bias} = \frac{1}{N} \sum_t (f_t - y_t)",
    ),
    (
        "**Diebold–Mariano test**",
        "Is one forecaster really better, or was it just lucky in these years? It compares "
        "two forecasters' daily losses, allowing for neighbouring days sharing most of their "
        "21-day targets. A p-value below 0.05 means the gap is unlikely to be luck.",
        r"\text{DM} = \frac{\bar{d}}{\text{se}(\bar{d})}",
        r"d_t = \text{loss}^A_t - \text{loss}^B_t",
    ),
]
# Every input the models can use: the four defaults, and three more tested in notebook 04
INPUTS = pd.DataFrame(
    {
        "what it measures": [
            "realized volatility of the last trading day",
            "realized volatility of the last 5 trading days, about a week",
            "realized volatility of the last 21 trading days, about a month",
            "realized volatility of the last 63 trading days, about a quarter",
            "realized volatility of the last 21 trading days, from the down days only",
            "the last VIX close before BIST's close",
            "USD/TRY realized volatility of the last 21 trading days (from 2005)",
        ],
        "used": ["default", "default", "default", "tested", "tested", "default", "tested"],
    },
    index=["rv_1d", "rv_5d", "rv_21d", "rv_63d", "rv_down_21d", "vix_close", "rv_usdtry_21d"],
)
FORECASTER_NAMES = [
    "historical average",
    "naive",
    "VIX rescaled",
    "GARCH",
    "linear regression",
    "XGBoost",
]
# MAE on the test years, 2018–2026, from the saved runs with these settings
TRAINING_MAE = pd.DataFrame(
    {
        "from 2003, refit yearly": [7.263, 8.191, 7.571, 7.103, 6.828, 7.214],
        "from 1997, refit yearly": [10.124, 8.191, 10.075, 7.817, 7.607, 7.392],
        "from 2003, fitted once": [7.270, 8.191, 7.636, 7.167, 6.846, 7.182],
    },
    index=FORECASTER_NAMES,
)
# MAE of each input set, chosen on 2013–2017 and checked once on 2018–2026 (notebook 04)
INPUT_MAE = pd.DataFrame(
    {
        "linear regression, 2013–2017": [5.665, 5.392, 5.642, 5.617, 5.459],
        "linear regression, 2018–2026": [6.828, 6.729, 6.744, 6.761, 6.674],
        "XGBoost, 2013–2017": [5.966, 7.017, 6.016, 6.005, 6.527],
        "XGBoost, 2018–2026": [7.214, 7.149, 7.332, 8.388, 8.065],
    },
    # st.table reads cells as Markdown, so a label can't start with "+" (a list item)
    index=[
        "default four",
        "default + rv_63d",
        "default + rv_down_21d",
        "default + rv_usdtry_21d",
        "all seven",
    ],
)
INPUT_FINDINGS = (
    "- **The validation years pick rv_63d for the linear regression** (MAE 5.39 against "
    "5.67), but on the test years it helps only a little (6.73 against 6.83) and makes QLIKE "
    "worse.\n"
    "- **On the validation years, no extra input helps XGBoost:** its MAE and RMSE are best "
    "with the default four.\n"
    "- **No leverage effect.** In most markets, falls are followed by higher volatility. In "
    "BIST, months dominated by down days were followed by a month 3.7 points calmer. Part of "
    "that is volatility falling back after a turbulent month, but the regression agrees, so "
    "rv_down_21d adds little.\n"
    "- **So the default four stay:** rv_1d, rv_5d, rv_21d and vix_close. No extra input helps "
    "both models reliably. This last choice did look at the test years."
)
# The base run: default settings, data up to 2026-10-16, test years 2018–2026
TEST_SCORES = pd.DataFrame(
    {
        "MAE": [7.263, 8.191, 7.571, 7.103, 6.828, 7.214],
        "RMSE": [9.557, 11.176, 9.800, 9.436, 9.124, 9.632],
        "QLIKE": [0.275, 0.402, 0.274, 0.244, 0.241, 0.268],
        "bias": [0.190, -0.129, 0.533, 1.084, 0.105, 0.174],
    },
    index=FORECASTER_NAMES,
)
# Diebold–Mariano tests of each forecaster against the linear regression, in the base run
TEST_VERDICTS = pd.DataFrame(
    {
        "MAE": [
            "no clear difference (p = 0.20)",
            "worse (p = 0.001)",
            "worse (p = 0.020)",
            "no clear difference (p = 0.070)",
            "worse (p = 0.001)",
        ],
        "RMSE": [
            "no clear difference (p = 0.30)",
            "worse (p < 0.001)",
            "no clear difference (p = 0.092)",
            "no clear difference (p = 0.10)",
            "worse (p = 0.003)",
        ],
        "QLIKE": [
            "no clear difference (p = 0.18)",
            "worse (p < 0.001)",
            "no clear difference (p = 0.21)",
            "no clear difference (p = 0.76)",
            "worse (p < 0.001)",
        ],
    },
    index=["historical average", "naive", "VIX rescaled", "GARCH", "XGBoost"],
)
RESULTS = (
    "- **Simple wins.** The linear regression beats naive and XGBoost on every score, and the "
    "VIX method on the average miss, and the tests say that's not luck.\n"
    "- **But not clearly against GARCH or the historical average.** Its lower scores against "
    "them could be luck (p from 0.07 to 0.76): beyond its usual level, next month's "
    "volatility is only a little predictable.\n"
    "- **GARCH runs a little high.** Its bias is +1.1 points, partly by construction: it "
    "averages the forecast variances and takes the square root, which comes out above the "
    "average volatility.\n"
    "- **More complexity didn't help.** XGBoost doesn't beat the straight line, and extra "
    "inputs didn't help reliably.\n"
    "- **Reacting late is costly.** Naive scores worst: it only catches up with a storm "
    "after it has started."
)
BEST_STYLE = "background-color: rgba(33, 195, 84, 0.2)"


@st.cache_data
def raw_tables() -> pd.DataFrame:
    """The size and dates of each downloaded price table in data/raw."""
    rows = {}
    for ticker, name in TABLE_NAMES.items():
        prices = load_prices(ticker)
        rows[name] = {
            "Yahoo symbol": TICKERS[ticker],
            "rows": f"{len(prices):,}",
            "from": f"{prices.index[0]:%Y-%m-%d}",
            "to": f"{prices.index[-1]:%Y-%m-%d}",
        }
    return pd.DataFrame(rows).T


@st.cache_data
def bist_volatility() -> pd.Series:
    """BIST 100 realized volatility over the last month, for every date."""
    returns = log_returns(load_prices("xu100")["close"]).dropna()
    return realized_volatility(returns, HORIZON).dropna()


def mark_closest_to_zero(values: pd.Series) -> list[str]:
    """Highlights the value closest to 0, the best bias."""
    distance = values.abs()
    return [BEST_STYLE if value == distance.min() else "" for value in distance]


def cards(items: Sequence[tuple[str, ...]], per_row: int = 3) -> None:
    """Bordered cards, `per_row` to a row: a title, a short text, then any formulas."""
    for first in range(0, len(items), per_row):
        row = items[first : first + per_row]
        for column, (title, text, *formulas) in zip(st.columns(per_row), row, strict=False):
            # $$ on their own lines make a centered formula, not one inside the text
            blocks = [title, text, *(f"$$\n{formula}\n$$" for formula in formulas)]
            column.container(border=True, height="stretch").markdown("\n\n".join(blocks))


defaults = Config()

st.title("How much will the Turkish stock market swing?")
st.markdown(
    "Six forecasters, from simple rules to machine learning, predict the volatility of the "
    "BIST 100, Turkey's main stock index. Each one is tested on years it never saw while "
    "learning, to see which would really have worked."
)
cards(SUMMARY)

st.header("Approach", divider="gray")
st.markdown(
    "Four steps, worked out in the notebooks and coded in src/volforecast: look at the data, "
    "turn prices into inputs and a target, build the forecasters, and score them on years "
    "they didn't learn from."
)

st.subheader("1. Data")
st.markdown(
    ":gray-badge[01_eda] Three daily price tables from Yahoo Finance, one row per trading day: "
    "the BIST 100, the VIX and the dollar–lira rate USD/TRY. The data is clean: no break at "
    "the 2005 currency change or the 2020 index rescaling, and only 4 days on which the BIST "
    "100 close didn't change at all."
)
tables_column, columns_column = st.columns([3, 2])
tables_column.table(raw_tables())
columns_column.table(RAW_COLUMNS)
cards(DATA_CONCEPTS)
volatility = bist_volatility().rename("volatility").to_frame()
title = f"BIST 100 realized volatility over the last {HORIZON} trading days (annualized %)"
fig = px.line(volatility, title=title, render_mode="svg")
line_color = "white" if st.context.theme.type == "dark" else "black"
fig.update_traces(line_color=line_color, line_width=1)
train_start, test_start = pd.Timestamp(defaults.train_start), pd.Timestamp(defaults.test_start)
periods = [
    (volatility.index[0], train_start, "left out: crisis years", "gray"),
    (train_start, test_start, "learning", "blue"),
    (test_start, volatility.index[-1], "testing", "red"),
]
for start, end, text, color in periods:
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor=color,
        opacity=0.1,
        line_width=0,
        annotation_text=text,
        annotation_position="top left",
    )
fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None, hovermode="x unified")
fig.update_yaxes(hoverformat=".1f")
st.plotly_chart(fig)
cards(DATA_FINDINGS)

st.subheader("2. Inputs and target")
st.markdown(
    ":gray-badge[transform.py] Everything is computed from daily returns. The inputs use only "
    "what's known at BIST's close on the forecast day, and the target is what the forecasters "
    "try to predict."
)
cards(VOLATILITY_CONCEPTS, per_row=2)
st.table(INPUTS)
st.markdown(
    "Two rules keep the forecasts honest. The VIX and USD/TRY close hours after BIST and have "
    "different holidays, so each BIST day uses their last value before it. And each target "
    "covers the next 21 days, so the last 21 training days before a forecast starts are left "
    "out: their targets would reach into the days being forecast."
)

st.subheader("3. Forecasters")
st.markdown(
    ":gray-badge[02_baselines] :gray-badge[03_models] Four baselines are simple rules, and two "
    "models learn from the inputs. A model is only worth it if it beats the baselines."
)
cards(FORECASTERS)

st.subheader("4. Evaluation")
st.markdown(
    ":gray-badge[evaluation.py] The forecasters learn from 2003–2017 and are scored on "
    "2018–2026, every trading day. At the start of each test year they learn again from all "
    "the years before it, as they would in real life. The inputs were first compared on "
    "2013–2017, a validation period before the test years. Two choices did look at the test "
    "years, the 2003 start and keeping the default inputs, so the test scores may be a little "
    "optimistic. Below, $f_t$ is the forecast made on day $t$, $y_t$ the volatility that "
    "followed, and $N$ the number of test days. For MAE, RMSE and QLIKE, lower is better."
)
cards(SCORES)

st.header("Results", divider="gray")

st.subheader("Training years and refits")
st.markdown(
    "Learning from 1997, crisis years included, makes every fitted forecaster forecast too "
    "high (the historical average by 6.5 points on average) and miss more. Refitting every "
    "year instead of once changes little, but it's how forecasts are made in practice. MAE on "
    "the test years, best in green:"
)
st.table(TRAINING_MAE.style.format(precision=3).highlight_min(axis=1, props=BEST_STYLE))

st.subheader("Which inputs")
st.markdown(
    ":gray-badge[04_inputs] Each set of inputs was scored on the validation years 2013–2017, "
    "and checked once on the test years. MAE, with the best on the validation years in green:"
)
validation_columns = pd.Index(["linear regression, 2013–2017", "XGBoost, 2013–2017"])
inputs_table = INPUT_MAE.style.format(precision=3)
st.table(inputs_table.highlight_min(subset=validation_columns, props=BEST_STYLE))
st.markdown(INPUT_FINDINGS)

st.subheader("Which forecaster is best")
# The first card is wider, so "Linear regression" fits in it on smaller screens
best, miss, level, edge = st.columns([3, 2, 2, 2])
best.metric("Best forecaster", "Linear regression", border=True)
miss.metric("Its average miss (MAE)", "6.8 points", border=True)
level.metric("Average volatility, 2018–2026", "24.9%", border=True)
edge.metric("Smaller miss than GARCH", "4%", border=True)
st.markdown("Scores on the test years, best in green:")
scores = TEST_SCORES.style.format(precision=3)
scores = scores.highlight_min(subset=["MAE", "RMSE", "QLIKE"], props=BEST_STYLE)
st.table(scores.apply(mark_closest_to_zero, subset=["bias"]))
st.markdown("Each forecaster against the linear regression, by Diebold–Mariano tests:")
st.table(TEST_VERDICTS)
st.markdown(
    "What the linear regression learned, fitted on all the data. The volatility weights add up "
    "to 0.41, so the forecast moves less than half as much as recent volatility and leans on "
    "the long-run level, and the VIX adds what BIST's own past doesn't have:"
)
st.latex(
    r"f = 11.7 + 0.02\,\text{rv\_1d} + 0.12\,\text{rv\_5d} + 0.27\,\text{rv\_21d} "
    r"+ 0.18\,\text{vix\_close}"
)
st.markdown(RESULTS)
st.caption(
    "Numbers from the notebooks and the base run: default settings, data up to 2026-10-16. "
    "The Run and Compare pages show the saved runs."
)
