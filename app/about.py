"""The About page: what the project does and what it found, in plain words."""

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

from volforecast.config import HORIZON, Config
from volforecast.extract import load_prices
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
        "volatility is around 25%. But no forecaster is far ahead, because next month's "
        "swings are only a little predictable.",
    ),
]
CONCEPTS = [
    (
        ":material/account_balance: **BIST 100**",
        "The index of the 100 largest companies on Borsa Istanbul, Turkey's stock exchange. "
        "When the news says the Turkish market rose 2% today, it usually means this index. "
        "Bigger companies weigh more, counting only the shares that trade freely:",
        r"\text{BIST 100} \propto \sum_{i=1}^{100} \text{price}_i \times \text{shares}_i",
    ),
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
        "day, and the result is annualized:",
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
        ":material/public: **VIX**",
        "Wall Street's “fear index”: the volatility traders expect for the US stock market "
        "over the next 30 days, read from option prices and already annualized. When global "
        "investors get nervous, it jumps, and markets like Turkey's often follow.",
        r"\text{expected 30-day move} \approx \pm\,\text{VIX} \times \sqrt{30 / 365}",
    ),
    (
        ":material/calendar_month: **Trading days and horizon**",
        "Trading days are the days the exchange is open, about 252 a year and 21 a month. "
        "The horizon h is how far ahead the forecast looks, 21 trading days by default. The "
        "target is only known h days later, so the newest h days don't have one yet:",
        r"\text{target}_t = \text{RV of days } t+1, \dots, t+h",
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
        "1997 to over 10,000, so on a regular scale the early years look flat. A doubling "
        "looks the same size everywhere:",
        r"\log 200 - \log 100 = \log 10{,}000 - \log 5{,}000",
    ),
    (
        ":material/tune: **Weights and importance**",
        "What the models learned, on the Run page's Model tab. A linear regression weight b "
        "is how many points the forecast moves per point of its input x. XGBoost's "
        "importance is each input's share of what its trees relied on, adding up to 1.",
        r"\Delta \text{forecast} = b \times \Delta x",
    ),
]
DATA_FINDINGS = [
    (
        ":material/warning: **Two eras**",
        "1997–2002 were years of crisis, with volatility often at 40–80%. Since 2003, it has "
        "mostly stayed between 15% and 40%. The forecasters learn from 2003 on, because the "
        "crisis years make them all forecast too high.",
    ),
    (
        ":material/waves: **Storms come in streaks**",
        "A turbulent month is usually followed by another, so recent volatility helps "
        "forecast the next month. But storms fade, and volatility drifts back toward its "
        "usual level.",
    ),
    (
        ":material/bolt: **Extreme days are common**",
        "Very large daily moves happen far more often than a bell curve predicts, so GARCH "
        "is given an error distribution that allows for them.",
    ),
    (
        ":material/trending_down: **Falls don't stir up more swings**",
        "In most stock markets, falling prices are followed by higher volatility. In BIST, "
        "months with mostly down days were followed by lower volatility, not higher.",
    ),
]
# The colors match each forecaster's lines and bars on the Run page
FORECASTERS = [
    (
        ":gray[**Historical average**] :gray-badge[baseline]",
        "Always forecasts the average volatility of the training years. It ignores what's "
        "happening now.",
        r"f = \text{average target of the training years}",
    ),
    (
        ":orange[**Naive**] :gray-badge[baseline]",
        "Next month will be like the last one: the forecast is the realized volatility of "
        "the last 21 trading days.",
        r"f_t = \text{rv\_21d}_t",
    ),
    (
        ":violet[**VIX rescaled**] :gray-badge[baseline]",
        "Takes the latest VIX and converts it to BIST levels, with a and b fitted on the "
        "training years.",
        r"f_t = a + b \times \text{vix\_close}_t",
    ),
    (
        ":blue[**GARCH**] :gray-badge[baseline]",
        "A classic finance model. Tomorrow's variance σ² (volatility squared) is a constant "
        "plus a share α of today's squared return and a share β of today's variance. After a "
        "shock, it forecasts high volatility that fades back to normal.",
        r"\sigma^2_{t+1} = \omega + \alpha\, r_t^2 + \beta\, \sigma^2_t",
    ),
    (
        ":red[**Linear regression**] :gray-badge[model]",
        "A weighted sum of four inputs x: the realized volatility of the last day, week and "
        "month, and the VIX. The weights b are the ones that fit the training years best.",
        r"f = b_0 + b_1 x_1 + b_2 x_2 + b_3 x_3 + b_4 x_4",
    ),
    (
        ":green[**XGBoost**] :gray-badge[model]",
        "Machine learning: K small decision trees (300 by default), each a few if-then "
        "questions about the same four inputs. Each tree corrects the ones before it, scaled "
        "by the learning rate η. It can find patterns a straight line can't, if there are any.",
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
        r"\text{QLIKE} = \frac{1}{N} \sum_t \left( \frac{y_t^2}{f_t^2} "
        r"- \ln \frac{y_t^2}{f_t^2} - 1 \right)",
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
        "two forecasters' daily losses; a p-value below 0.05 means the gap is unlikely to be "
        "luck.",
        r"\text{DM} = \frac{\bar{d}}{\text{se}(\bar{d})}, \quad "
        r"d_t = \text{loss}^A_t - \text{loss}^B_t",
    ),
]
SETTINGS = [
    (
        ":material/date_range: **Train from, compare with, test from**",
        "The forecasters learn from the Train from year on and are tested from the Test from "
        "year on. A second training start is scored next to the first, to show what the "
        "starting year changes: by default 2003 against 1997, crisis years included.",
    ),
    (
        ":material/autorenew: **Refit models**",
        "Yearly: at the start of each test year, every forecaster learns again from all the "
        "years before it, as you would in real life. Once: it learns only from the years "
        "before the test and is never updated.",
    ),
    (
        ":material/schedule: **Horizon**",
        "How many trading days ahead to forecast: 5 (a week), 10, 21 (a month) or 63 (a "
        "quarter). The target, what actually happened, changes with it.",
    ),
    (
        ":material/input: **Model inputs**",
        "What the linear regression and XGBoost see. rv_ means realized volatility over a "
        "past window: rv_1d the last day, rv_5d the week, rv_21d the month, rv_63d the "
        "quarter, rv_down_21d the month's down days only. vix_close is the last VIX close, "
        "and rv_usdtry_21d the dollar–lira rate's volatility over the last month.",
    ),
    (
        ":material/bar_chart: **GARCH errors**",
        "How GARCH expects daily moves to be spread. t allows for the extreme days BIST has; "
        "normal assumes a bell curve.",
    ),
    (
        ":material/park: **XGBoost settings**",
        "Trees: how many small trees are added up. Depth: how many if-then questions each "
        "tree may ask. Learning rate: how much each new tree corrects the ones before it. "
        "More and deeper trees fit the past better, but can forecast worse.",
    ),
]
RESULTS = (
    "- **Simple wins.** The linear regression beats naive, XGBoost and the VIX method on the "
    "average miss, and the tests say that's not luck. Against GARCH, the best baseline, the "
    "gap is borderline.\n"
    "- **But not by much.** It can't be told apart from the historical average: beyond its "
    "usual level, next month's volatility is only a little predictable.\n"
    "- **More complexity didn't help.** XGBoost doesn't beat the straight line, and extra "
    "inputs (3-month or down-day volatility, the dollar–lira rate) didn't help reliably.\n"
    "- **Reacting late is costly.** Naive scores worst: it only catches up with a storm "
    "after it has started."
)


@st.cache_data
def bist_volatility() -> pd.Series:
    """BIST 100 realized volatility over the last month, for every date."""
    returns = log_returns(load_prices("xu100")["close"]).dropna()
    return realized_volatility(returns, HORIZON).dropna()


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

st.header("Concepts", divider="gray")
cards(CONCEPTS)

st.header("The data", divider="gray")
st.markdown(
    "Daily closes of the BIST 100 since 1997, the VIX, and the dollar–lira rate (USD/TRY, "
    "since 2005), from Yahoo Finance. The Data page shows them, with every model input."
)
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
cards(DATA_FINDINGS, per_row=2)

st.header("Six forecasters", divider="gray")
st.markdown(
    "The four baselines are simple rules. The two models learn from the data, and they're "
    "only worth it if they beat the baselines."
)
cards(FORECASTERS)

st.header("How they're judged", divider="gray")
st.markdown(
    "Each forecast is checked against the volatility that actually followed, over the test "
    "years 2018–2026. The forecasters only use what was known on the day, and every January "
    "they learn again with the year that just ended, as they would in real life. Below, "
    "$f_t$ is the forecast made on day $t$, $y_t$ the volatility that followed, and $N$ the "
    "number of test days. For MAE, RMSE and QLIKE, lower is better."
)
cards(SCORES)

st.header("Settings you can change", divider="gray")
st.markdown(
    "They're in the sidebar of the Run page. Press Run to see their effect: every run is "
    "saved, and the Compare page shows runs side by side."
)
cards(SETTINGS)

st.header("What we found", divider="gray")
best, miss, level, edge = st.columns(4)
best.metric("Best forecaster", "Linear regression", border=True)
miss.metric("Its average miss (MAE)", "6.8 points", border=True)
level.metric("Average volatility, 2018–2026", "24.9%", border=True)
edge.metric("Smaller miss than GARCH", "4%", border=True)
st.markdown(RESULTS)
st.caption(
    "Numbers for the default settings, with data up to October 2026. The Run page has the "
    "current numbers and lets you try other settings."
)
