"""The About page: the project step by step, from the data to the results.

Each step says what was done and why, with the definitions and formulas it needs. The steps
follow the notebooks (01_eda, 02_baselines, 03_models, 04_inputs) and src/volforecast. The
numbers are fixed, from the notebooks and the base run (2026-10-18_15-32-43, data up to
2026-10-16); the raw-data table and the charts are read from data/raw.
"""

from collections.abc import Sequence
from typing import cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from volforecast.config import HORIZON, Config
from volforecast.extract import TICKERS, load_prices
from volforecast.transform import log_returns, realized_volatility

# Each card is a title (with an icon or a color), a short text and any LaTeX formulas
SUMMARY = [
    (
        ":material/help: **The question**",
        "- How much will the BIST 100, Turkey's main stock index, swing over the next 21 "
        "trading days, about a month?\n"
        "- Only the size of the moves, not their direction.",
    ),
    (
        ":material/model_training: **The approach**",
        "- Six forecasters, from simple rules to machine learning.\n"
        "- Those that learn from data learn from 2003–2017.\n"
        "- All six then forecast every trading day from 2018 to September 2026, and each "
        "forecast is checked against what actually happened.",
    ),
    (
        ":material/emoji_events: **The answer**",
        "- A simple linear regression does best. Its forecast of the next month's volatility "
        "is off by 6.8 percentage points on average, when that volatility averages 24.9% a "
        "year.\n"
        "- It's reliably better than three of the five others.\n"
        "- But the tests can't tell it apart from GARCH, a classic volatility model, or from "
        "the historical average: next month's swings are only a little predictable.",
    ),
]
TABLE_NAMES = {"xu100": "BIST 100", "vix": "VIX", "usdtry": "USD/TRY"}
TABLE_UNITS = {
    "xu100": "index points",
    "vix": "points, read as % a year",
    "usdtry": "lira per US dollar",
}
# The columns every downloaded price table has, each in its table's unit
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
# Step 2: from prices to daily returns
RETURN_CONCEPTS = [
    (
        ":material/stacked_line_chart: **Why a log scale**",
        "- The BIST 100 rose from 18.5 index points in July 1997 to about 11,700 points, "
        "some 630-fold, mostly through inflation.\n"
        "- On a regular axis everything before 2020 hugs the floor.\n"
        "- A log axis spaces values by ratio, so the same % move looks the same size in any "
        "year, and % moves are what volatility is about.\n"
        "- It only changes how the chart is drawn: the returns come from the prices "
        "themselves.\n"
        "- Its ticks are powers of 10 because they read easily; any log base draws the same "
        "picture.",
        r"\log \frac{200}{100} = \log \frac{10{,}000}{5{,}000}",
    ),
    (
        ":material/swap_vert: **Daily return**",
        "- How much the index moved from one close to the next, in %.\n"
        "- It uses the natural log (ln), the one log that reads as a % change: a 2% rise "
        "gives 1.98, where log10 would give 0.86.\n"
        "- Log returns also add up: two days' returns sum to the two-day return.\n"
        "- With $P_t$ the close on day $t$:",
        r"r_t = 100 \times \ln \frac{P_t}{P_{t-1}}",
    ),
    (
        ":material/bolt: **Extreme days are common**",
        "- Kurtosis measures how often extreme days happen. A bell curve (normal "
        "distribution) has 3.\n"
        "- Excess kurtosis is kurtosis minus 3, so a bell curve has 0. It's what pandas and "
        "most software report.\n"
        "- BIST's daily returns have an excess kurtosis of 7.3: very large days are far more "
        "common than a bell curve predicts (the table's last row).\n"
        "- Skewness measures lopsidedness: BIST's −0.16 means the big falls are slightly "
        "bigger than the big rises.",
    ),
]
# Step 3: measuring volatility
VOLATILITY_CONCEPTS = [
    (
        ":material/ssid_chart: **Volatility**",
        "- How much a price swings, up or down: the standard deviation σ of its returns.\n"
        "- Always measured on returns, not prices: a 100-point move is 1% at 10,000 points "
        "but 100% at 100, and price levels also climb with inflation.\n"
        "- Nobody sees σ directly: it can only be measured afterwards, expected by traders, "
        "or forecast.",
        r"\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (r_i - \bar{r})^2}",
    ),
    (
        ":material/history: **Realized volatility**",
        "- The volatility that actually happened over the last n trading days, from their "
        "daily returns.\n"
        "- The average return is left out: for a day it's close to 0 (0.09% a day).\n"
        "- Annualized, in % a year. rv_21d, for example, is RV over the last 21 days.",
        r"\text{RV}_n = \sqrt{\frac{252}{n} \sum_{i=1}^{n} r_i^2}",
    ),
    (
        ":material/percent: **Annualized, in % a year**",
        "- Volatility grows with the square root of time, so it's scaled to a year to compare "
        "periods of any length.\n"
        "- At 25% a year, about two times in three the index ends a year within ±25% of where "
        "it started, and 21 trading days within ±7%.",
        r"\sigma_{\text{year}} = \sqrt{252}\,\sigma_{\text{day}}",
        r"\sigma_{h \text{ days}} = \sqrt{h / 252}\,\sigma_{\text{year}}",
    ),
]
# The biggest moves on the long-run volatility chart: where to look, a spike or a calm, and
# why. The dates and sizes were checked against data/raw/xu100.csv.
EVENTS = [
    (
        "1998-08-01",
        "1998-10-31",
        "peak",
        "**Aug 1998:** Russia's default spreads to emerging markets: −13% on 27 August.",
    ),
    (
        "2000-11-01",
        "2001-04-30",
        "peak",
        "**Nov 2000 – Feb 2001:** Turkey's banking crisis. A run on banks and an IMF rescue, "
        "then a political row sinks the lira, which is floated. The chart's highest peaks, "
        "near 120%.",
    ),
    (
        "2003-02-15",
        "2003-05-31",
        "peak",
        "**Mar 2003:** parliament refuses US troops, then the Iraq war begins: about 82%, the "
        "highest after 2002.",
    ),
    (
        "2008-09-01",
        "2008-12-31",
        "peak",
        "**Sep–Oct 2008:** Lehman Brothers collapses and the global financial crisis hits: "
        "about 77%.",
    ),
    (
        "2013-05-15",
        "2013-07-31",
        "peak",
        "**Jun 2013:** the Gezi Park protests, then the US Federal Reserve signals it will "
        "slow its bond buying: about 61%.",
    ),
    (
        "2016-07-10",
        "2016-08-31",
        "peak",
        "**Jul 2016:** the coup attempt: −7% in one day, but a short shock, about 36%.",
    ),
    (
        "2017-01-01",
        "2017-12-31",
        "calm",
        "**2017:** a calm year, with a steady rally and the VIX near record lows: about 13% in "
        "July.",
    ),
    (
        "2018-08-01",
        "2018-09-30",
        "peak",
        "**Aug 2018:** US sanctions and tariffs crash the lira. The index falls 10% in a "
        "week, but most of the shock hits the currency: about 32%.",
    ),
    (
        "2020-02-15",
        "2020-05-31",
        "peak",
        "**Mar 2020:** the COVID-19 crash: −8% on 16 March, about 57%.",
    ),
    (
        "2021-11-15",
        "2022-01-31",
        "peak",
        "**Dec 2021:** rate cuts sink the lira. Stocks first rise as a hedge, then swing hard "
        "both ways: about 58%.",
    ),
    (
        "2023-02-01",
        "2023-03-31",
        "peak",
        "**Feb 2023:** the earthquakes. −8.6% on 7 February, a week without trading, then "
        "+9.9% on reopening: about 60%, the highest of the test years.",
    ),
    (
        "2025-03-10",
        "2025-05-31",
        "peak",
        "**Mar 2025:** Istanbul's mayor, Ekrem İmamoğlu, is detained: −8.7% on 19 March, "
        "about 50%.",
    ),
]
VOLATILITY_FINDINGS = [
    (
        ":material/warning: **Two eras**",
        "- 1997–2002: volatility often 40–80% a year, peaking near 120%.\n"
        "- From 2003: mostly 15–40% a year, with crisis spikes up to about 80%.\n"
        "- So the forecasters learn from 2003 on: the crisis years make them all forecast too "
        "high (see the results).",
    ),
    (
        ":material/waves: **Storms come in streaks**",
        "- Big moves, up or down, tend to follow big moves, and calm days follow calm days. "
        "This is called volatility clustering.\n"
        "- Measured with correlation, which runs from −1 to 1 (0: no link): the size of a "
        "day's move, |r| (the return without its sign), has 0.25 with the next day's size, "
        "and still 0.14 with the size 60 trading days, about three months, later.\n"
        "- The direction doesn't carry over: the return r itself has a correlation of about 0 "
        "with earlier days' returns. So the size of the moves can be forecast, their "
        "direction can't.\n"
        "- Streaks end: volatility drifts back to its usual level. So the last month says "
        "something about the next one, but not everything.",
    ),
]
# Step 4: how predictable each horizon is over 1997–2026 (notebook 01): the target's average
# level, and how closely the naive forecast, the last h days' volatility, follows it
HORIZONS = pd.DataFrame(
    {
        "average volatility (% a year)": [23.7, 28.7, 29.6, 30.4, 31.2],
        "naive forecast: correlation": [0.25, 0.55, 0.61, 0.65, 0.70],
        "naive forecast: average miss (pp)": [20.8, 12.3, 10.6, 9.4, 8.1],
    },
    index=["1 day", "5 days, a week", "10 days", "21 days, a month", "63 days, a quarter"],
)
HORIZON_CARD = (
    ":material/straighten: **Why 21 trading days**",
    "- **1 day:** almost unpredictable. The naive forecast misses by 20.8 pp, about as much "
    "as the average level, 23.7% a year.\n"
    "- **5 and 10 days:** better (correlation 0.55 and 0.61), but still noisy.\n"
    "- **21 days, a month:** correlation 0.65 and a miss of 9.4 pp. It's the window the VIX "
    "covers, so the VIX baseline compares like with like.\n"
    "- **63 days, a quarter:** only a little more predictable (0.70). The VIX doesn't cover "
    "it, and 2018–2026 holds only about 34 separate quarters to test on, against about 100 "
    "separate months.\n"
    "- These correlations are over 1997–2026. From 2005, without the crisis years, the 21-day "
    "one is 0.45.",
)
TARGET_CARD = (
    ":material/calendar_month: **The target**",
    "- What every forecaster predicts: the realized volatility of the 21 trading days after "
    "the forecast day, in % a year.\n"
    "- It's only known 21 days later, so the newest 21 days have no target yet.",
    r"\text{target}_t = \text{RV of days } t+1, \dots, t+21",
)
# Step 5: the VIX and the lira, the two inputs from outside BIST
INPUT_CONCEPTS = [
    (
        ":material/public: **VIX** (Cboe Volatility Index)",
        "- Wall Street's “fear index”: the volatility traders expect for the US stock market "
        "over the next 30 calendar days, about 21 trading days.\n"
        "- Read from option prices, in index points that read as % a year.\n"
        "- When global investors get nervous it jumps, and Turkey often follows.",
        r"\text{expected 30-day move} \approx \pm\,\text{VIX} \times \sqrt{30 / 365}",
    ),
    (
        ":material/currency_exchange: **USD/TRY**",
        "- Lira per US dollar. Lira stress is the Turkey-specific risk the VIX misses.\n"
        "- But its volatility follows the next month's BIST volatility only weakly "
        "(correlation 0.17 over 2005–2026), and it has been almost flat since late 2023.\n"
        "- So it's tested, not used by default.",
    ),
]
# Step 5: the two rules that keep the inputs to what was known on the forecast day
RULES = [
    (
        ":material/schedule: **Only closes before BIST's**",
        "- The VIX and USD/TRY close hours after BIST, and on other holidays.\n"
        "- So each BIST day uses their last close before BIST's.\n"
        "- The latest forecast, at BIST's close on 16 October 2026, uses the VIX close of 15 "
        "October (16.46 points), not that evening's 19.99.",
    ),
    (
        ":material/content_cut: **A gap before each forecast**",
        "- Each target covers the next 21 trading days.\n"
        "- So the last 21 training days before a forecast starts are left out: their targets "
        "would reach into the days being forecast.",
    ),
]
# Steps 6 and 7. The colors match each forecaster's lines and bars on the Run page.
BASELINES = [
    (
        ":gray[**Historical average**] :gray-badge[baseline]",
        "- Always forecasts the average target of the training years.\n"
        "- Ignores what's happening now.",
        r"f = \overline{\text{target}}_{\text{training years}}",
    ),
    (
        ":orange[**Naive**] :gray-badge[baseline]",
        "- Next month will be like the last one: the forecast is rv_21d, the realized "
        "volatility of the last 21 trading days.\n"
        "- Learns nothing.",
        r"f_t = \text{rv\_21d}_t",
    ),
    (
        ":violet[**VIX rescaled**] :gray-badge[baseline]",
        "- **Why rescale:** the VIX measures the US market, which is usually calmer. Over "
        "2018–2026 it averaged 19.6 points, while BIST's volatility averaged 24.9% a year, so "
        "the raw VIX would forecast too low.\n"
        "- **How:** a straight line fitted by least squares on the training years, and again "
        "every January, turns the VIX into a BIST forecast, with intercept a and slope b.",
        r"f_t = a + b \times \text{vix\_close}_t",
    ),
    (
        ":blue[**GARCH**] (Generalized AutoRegressive Conditional Heteroskedasticity) "
        ":gray-badge[baseline]",
        "- The classic volatility model: a variance that changes over time and depends on its "
        "own past.\n"
        "- Tomorrow's variance σ² (volatility squared) is a constant plus a share α of today's "
        "squared return and a share β of today's variance.\n"
        "- So after a shock it forecasts high volatility that fades back to normal.\n"
        "- Its errors follow a Student-t distribution, a bell curve with fatter tails.\n"
        "- For the next 21 days it averages its 21 daily variance forecasts, annualizes them "
        "and takes the square root.",
        r"\sigma^2_{t+1} = \omega + \alpha\, r_t^2 + \beta\, \sigma^2_t",
    ),
]
MODELS = [
    (
        ":red[**Linear regression**] :gray-badge[model]",
        "- A weighted sum of four inputs x: rv_1d, rv_5d and rv_21d, the classic HAR model "
        "(Heterogeneous AutoRegressive: short and long windows side by side), plus "
        "vix_close.\n"
        "- The weights b are the ones that fit the training years best (least squares).",
        r"f = b_0 + \sum_{j=1}^{4} b_j \, x_j",
    ),
    (
        ":green[**XGBoost**] (eXtreme Gradient Boosting) :gray-badge[model]",
        "- K small decision trees, each a few if-then questions about the same four inputs.\n"
        "- Each tree corrects the errors of the ones before it, scaled by the learning rate "
        "η, and sees a random 80% of the days.\n"
        "- Kept small and slow (300 trees of depth 3, η = 0.05): the first fit has about "
        "3,700 training days, but with targets overlapping by 21 days that's only about 180 "
        "independent months.",
        r"f = f_0 + \eta \sum_{k=1}^{K} \text{tree}_k(x)",
    ),
]
# Step 8
SCORES = [
    (
        "**MAE** (mean absolute error)",
        "- The average miss, in pp.\n- A forecast of 25% a year when 30% happens misses by 5 pp.",
        r"\text{MAE} = \frac{1}{N} \sum_t |f_t - y_t|",
    ),
    (
        "**RMSE** (root mean squared error)",
        "- Like MAE, in pp, but big misses count extra.\n"
        "- So a forecaster that is sometimes far off scores worse.",
        r"\text{RMSE} = \sqrt{\frac{1}{N} \sum_t (f_t - y_t)^2}",
    ),
    (
        "**QLIKE** (quasi-likelihood loss)",
        "- The standard score for volatility forecasts, computed on variances, with no unit.\n"
        "- It punishes forecasting too low, being caught off guard, more than forecasting too "
        "high.",
        r"\text{QLIKE} = \tfrac{1}{N} \sum_t (q_t - \ln q_t - 1)",
        r"q_t = y_t^2 / f_t^2",
    ),
    (
        "**Bias**",
        "- The average forecast minus the average outcome, in pp.\n"
        "- Positive means forecasting too high; closer to 0 is better.",
        r"\text{bias} = \frac{1}{N} \sum_t (f_t - y_t)",
    ),
    (
        "**Diebold–Mariano test**",
        "- Is one forecaster really better, or was it just lucky in these years?\n"
        "- It averages the daily differences d in loss (for RMSE, in squared error) and "
        "divides by their standard error.\n"
        "- Neighbouring days share 20 of their 21 target days, so their differences move "
        "together: the standard error counts that correlation, the autocovariances γ, over "
        "20 days in full.\n"
        "- With p below 0.05, the gap is unlikely to be luck.",
        r"\text{DM} = \frac{\bar{d}}{\text{se}(\bar{d})}",
        r"d_t = \text{loss}^A_t - \text{loss}^B_t",
        r"\text{se}(\bar{d})^2 = \frac{1}{N} \Big( \gamma_0 + 2 \sum_{k=1}^{20} \gamma_k \Big)",
    ),
]
# Step 5: every input the models can use, the four defaults and three more tested in step 9
INPUTS = pd.DataFrame(
    {
        "what it measures, in % a year": [
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
# MAE on the test years 2018–2026: the base run's two training starts, and the fit-once
# columns of notebooks 02 and 03
TRAINING_MAE = pd.DataFrame(
    {
        "from 2003, refit yearly (pp)": [7.263, 8.191, 7.571, 7.103, 6.828, 7.214],
        "from 1997, refit yearly (pp)": [10.124, 8.191, 10.075, 7.817, 7.607, 7.392],
        "from 2003, fitted once (pp)": [7.270, 8.191, 7.636, 7.167, 6.846, 7.182],
    },
    index=FORECASTER_NAMES,
)
# MAE of each input set, chosen on 2013–2017 and checked once on 2018–2026 (notebook 04)
INPUT_MAE = pd.DataFrame(
    {
        "linear regression, 2013–2017 (pp)": [5.665, 5.392, 5.642, 5.617, 5.459],
        "linear regression, 2018–2026 (pp)": [6.828, 6.729, 6.744, 6.761, 6.674],
        "XGBoost, 2013–2017 (pp)": [5.966, 7.017, 6.016, 6.005, 6.527],
        "XGBoost, 2018–2026 (pp)": [7.214, 7.149, 7.332, 8.388, 8.065],
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
    "- **The validation years pick rv_63d for the linear regression:** MAE 5.39 pp against "
    "5.67. But on the test years it helps only a little (6.73 pp against 6.83) and makes "
    "QLIKE worse.\n"
    "- **No extra input improves XGBoost's MAE or RMSE on the validation years.**\n"
    "- **No sign of a leverage effect month to month.** In most markets, falls are followed "
    "by higher volatility. In BIST, months dominated by down days were followed by a month "
    "3.7 pp calmer than the one before, and months dominated by up days by one 2.4 pp "
    "livelier.\n"
    "- Part of that is volatility falling back after a turbulent month, but the linear "
    "regression also gives down-day volatility a negative weight, so rv_down_21d adds "
    "little.\n"
    "- **So the default four stay:** rv_1d, rv_5d, rv_21d and vix_close. No extra input helps "
    "both models reliably. This last choice did look at the test years."
)
# The base run: default settings, data up to 2026-10-16, test years 2018–2026
TEST_SCORES = pd.DataFrame(
    {
        "MAE (pp)": [7.263, 8.191, 7.571, 7.103, 6.828, 7.214],
        "RMSE (pp)": [9.557, 11.176, 9.800, 9.436, 9.124, 9.632],
        "QLIKE": [0.275, 0.402, 0.274, 0.244, 0.241, 0.268],
        "bias (pp)": [0.190, -0.129, 0.533, 1.084, 0.105, 0.174],
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
    "- **Simple wins.** The linear regression beats naive and XGBoost on every score, and VIX "
    "rescaled on MAE, and the tests say that's not luck.\n"
    "- **But not clearly against GARCH and the historical average**, or against VIX rescaled "
    "on RMSE and QLIKE: those gaps could be luck (p from 0.07 to 0.76). Beyond its usual "
    "level, next month's volatility is only a little predictable.\n"
    "- **GARCH runs a little high, partly by construction.** It forecasts variance, and even "
    "when that's right on average, its square root comes out above the average volatility "
    "that follows. Its bias is +1.1 pp.\n"
    "- **More complexity didn't help.** XGBoost doesn't beat the linear regression, and extra "
    "inputs didn't help reliably.\n"
    "- **Reacting late is costly.** Naive scores worst: it only catches up with a storm "
    "after it has started."
)
# The base run's forecasts for the 21 trading days after 2026-10-16, annualized %
LATEST_FORECASTS = pd.DataFrame(
    [[25.2, 26.1, 24.2, 27.0, 25.6, 23.8]],
    columns=FORECASTER_NAMES,
    index=["forecast (% a year)"],
)
BEST_STYLE = "background-color: rgba(33, 195, 84, 0.2)"


@st.cache_data
def raw_tables() -> pd.DataFrame:
    """The size, unit and dates of each downloaded price table in data/raw, one per column.

    One column per table leaves room for the dates, which would otherwise wrap.
    """
    rows = {}
    for ticker, name in TABLE_NAMES.items():
        prices = load_prices(ticker)
        rows[name] = {
            "Yahoo symbol": TICKERS[ticker],
            "unit": TABLE_UNITS[ticker],
            "trading days": f"{len(prices):,}",
            "from": f"{prices.index[0]:%Y-%m-%d}",
            "to": f"{prices.index[-1]:%Y-%m-%d}",
        }
    return pd.DataFrame(rows)


@st.cache_data
def bist_close() -> pd.Series:
    """The BIST 100's daily close, from data/raw."""
    return load_prices("xu100")["close"]


def signed(value: float, decimals: int) -> str:
    """A number with its sign, written with a real minus sign as in the text."""
    return f"{value:+.{decimals}f}".replace("-", "−")


@st.cache_data
def return_stats() -> pd.DataFrame:
    """Basic statistics of the BIST 100's daily log returns, in %, over all the data."""
    returns = 100 * log_returns(bist_close()).dropna()
    dates = f"{returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d}"
    std = float(returns.std())
    excess_kurtosis = cast(float, returns.kurt())
    # A bell curve has 0.27% of its days beyond 3 standard deviations
    extreme_share = 100 * float((returns.abs() > 3 * std).mean())
    rows = {
        "returns": f"{len(returns):,}, {dates}",
        "mean": f"{signed(returns.mean(), 2)}% a day",
        "median": f"{signed(returns.median(), 2)}% a day",
        "standard deviation": f"{std:.2f}% a day, {std * 252**0.5:.1f}% a year",
        "lowest": f"{signed(returns.min(), 1)}% on {returns.idxmin():%Y-%m-%d}",
        "highest": f"{signed(returns.max(), 1)}% on {returns.idxmax():%Y-%m-%d}",
        "skewness": f"{signed(cast(float, returns.skew()), 2)} (a bell curve: 0)",
        "kurtosis": f"{excess_kurtosis + 3:.1f} (a bell curve: 3)",
        "excess kurtosis": f"{excess_kurtosis:.1f} (a bell curve: 0)",
        "days beyond ±3 standard deviations": f"{extreme_share:.1f}% (a bell curve: 0.3%)",
    }
    return pd.Series(rows, name="BIST 100 daily return").to_frame()


@st.cache_data
def bist_volatility() -> pd.Series:
    """BIST 100 realized volatility over the last month, for every date."""
    returns = log_returns(bist_close()).dropna()
    return realized_volatility(returns, HORIZON).dropna().rename("volatility")


def line_color() -> str:
    """The color of a chart's only line: black, or white on a dark theme."""
    return "white" if st.context.theme.type == "dark" else "black"


def price_chart() -> go.Figure:
    """The BIST 100 close since 1997, on a log scale."""
    close = bist_close().rename("close").to_frame()
    fig = px.line(close, log_y=True, title="BIST 100 close, on a log scale", render_mode="svg")
    fig.update_traces(line_color=line_color(), line_width=1)
    fig.update_layout(
        showlegend=False, xaxis_title=None, yaxis_title="index points", hovermode="x unified"
    )
    fig.update_yaxes(hoverformat=",.0f")
    return fig


def volatility_chart() -> go.Figure:
    """BIST 100 realized volatility since 1997, with each of EVENTS numbered on it."""
    volatility = bist_volatility()
    title = f"BIST 100 realized volatility over the last {HORIZON} trading days"
    fig = px.line(volatility.to_frame(), title=title, render_mode="svg")
    fig.update_traces(line_color=line_color(), line_width=1)
    for number, (start, end, kind, _) in enumerate(EVENTS, start=1):
        window = volatility.loc[start:end]
        day, value = (
            (window.idxmax(), window.max()) if kind == "peak" else (window.idxmin(), window.min())
        )
        fig.add_annotation(
            x=day,
            y=value,
            text=str(number),
            showarrow=False,
            # Above a spike, below a calm
            yshift=12 if kind == "peak" else -12,
            bgcolor="#ff4b4b",
            font={"color": "white", "size": 11},
            borderpad=2,
        )
    fig.update_layout(
        showlegend=False, xaxis_title=None, yaxis_title="% a year", hovermode="x unified"
    )
    fig.update_yaxes(hoverformat=".1f")
    return fig


def split_chart() -> go.Figure:
    """The years left out, learned from and tested on, as one bar from 1997 to today."""
    defaults = Config()
    volatility = bist_volatility()
    train_start, test_start = pd.Timestamp(defaults.train_start), pd.Timestamp(defaults.test_start)
    columns = {
        "years": ["left out: crisis years", "learning: grows every January", "testing"],
        "start": [volatility.index[0], train_start, test_start],
        "end": [train_start, test_start, volatility.index[-1]],
    }
    fig = px.timeline(
        pd.DataFrame(columns),
        x_start="start",
        x_end="end",
        y=["split"] * 3,
        color="years",
        text="years",
        color_discrete_sequence=["#a3a8b8", "#1c83e1", "#ff4b4b"],
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle")
    fig.update_layout(
        height=110, showlegend=False, xaxis_title=None, yaxis_title=None, margin={"t": 10, "b": 10}
    )
    fig.update_yaxes(visible=False)
    return fig


def mark_closest_to_zero(values: pd.Series) -> list[str]:
    """Highlights the value closest to 0, the best bias."""
    distance = values.abs()
    return [BEST_STYLE if value == distance.min() else "" for value in distance]


def card(title: str, text: str, *formulas: str) -> None:
    """One bordered card: a bold title, a short text, then any formulas, centered."""
    # $$ on their own lines make a centered formula, not one inside the text
    blocks = [title, text, *(f"$$\n{formula}\n$$" for formula in formulas)]
    st.container(border=True, height="stretch").markdown("\n\n".join(blocks))


def cards(items: Sequence[tuple[str, ...]], per_row: int = 3) -> None:
    """Bordered cards, `per_row` to a row."""
    for first in range(0, len(items), per_row):
        row = items[first : first + per_row]
        for column, item in zip(st.columns(per_row), row, strict=False):
            with column:
                card(*item)


st.title("How much will the Turkish stock market swing?")
st.markdown(
    "- Six forecasters, from simple rules to machine learning, predict the volatility of the "
    "BIST 100, Turkey's main stock index: how much it will swing, up or down.\n"
    "- Each one is tested on years it didn't learn from, to see which would really have "
    "worked."
)
cards(SUMMARY)

st.header("The project, step by step", divider="gray")

st.subheader("1. Get the data")
st.markdown(
    ":gray-badge[extract.py] :gray-badge[01_eda]\n\n"
    "- Daily prices from Yahoo Finance, one row per trading day, for three series: the BIST "
    "100, which is forecast; the VIX, Wall Street's “fear index” (step 5); and USD/TRY, the "
    "dollar–lira exchange rate.\n"
    "- The closes are clean: no break at the 2005 currency change, when the lira lost six "
    "zeros, or at the 2020 index rescaling, when the index lost two.\n"
    "- Only 4 days have an unchanged BIST close, one of them the session halted after the "
    "2023 earthquakes.\n"
    "- The volume column has gaps and changes units in July 2020, but it isn't used."
)
tables_column, columns_column = st.columns([3, 2])
tables_column.table(raw_tables())
columns_column.table(RAW_COLUMNS)

st.subheader("2. From prices to daily returns")
st.markdown(
    ":gray-badge[transform.py] :gray-badge[01_eda]\n\n"
    "- Volatility is about how much prices move, so each day's close is turned into a daily "
    "return first.\n"
    "- The chart draws the prices on a log scale; the returns are computed from the prices "
    "themselves."
)
st.plotly_chart(price_chart())
log_scale, daily_return, extremes = RETURN_CONCEPTS
cards([log_scale, daily_return], per_row=2)
extremes_column, stats_column = st.columns(2)
with extremes_column:
    card(*extremes)
stats_column.table(return_stats())

st.subheader("3. Measure volatility")
st.markdown(
    ":gray-badge[transform.py] :gray-badge[01_eda]\n\n"
    "- Volatility can't be seen directly.\n"
    "- It's measured from the returns that already happened, as realized volatility over a "
    "window of days."
)
cards(VOLATILITY_CONCEPTS)
st.plotly_chart(volatility_chart())
events_intro = (
    "The numbered moves on the chart, and what drove them. A rise or fall is one day's "
    "return; a level is the chart's volatility, in % a year:"
)
event_lines = [f"{number}. {text}" for number, (*_, text) in enumerate(EVENTS, start=1)]
st.markdown(events_intro + "\n\n" + "\n".join(event_lines))
cards(VOLATILITY_FINDINGS, per_row=2)

st.subheader("4. Choose the horizon and build the target")
st.markdown(
    ":gray-badge[01_eda]\n\n"
    "- The horizon is how many trading days ahead to forecast.\n"
    "- To choose it, for each horizon h the naive forecast, the volatility of the last h "
    "days, was compared with the volatility of the next h days, over 1997–2026.\n"
    "- Its miss is in pp, percentage points: the gap between two percentages, so a forecast "
    "of 25% a year when 30% happens misses by 5 pp."
)
horizons = HORIZONS.style.format(precision=1)
st.table(horizons.format(precision=2, subset=["naive forecast: correlation"]))
cards([HORIZON_CARD, TARGET_CARD], per_row=2)

st.subheader("5. Build the inputs")
st.markdown(
    ":gray-badge[transform.py]\n\n"
    "- Each forecast may only use what's known at BIST's close on the forecast day.\n"
    "- The inputs: BIST's realized volatility over several past windows, the VIX and the "
    "lira's volatility.\n"
    "- Four are used by default; three more are tested in step 9."
)
st.table(INPUTS)
cards(INPUT_CONCEPTS, per_row=2)
st.markdown("Two rules keep the inputs honest:")
cards(RULES, per_row=2)

st.subheader("6. Baselines: simple rules to beat")
st.markdown(
    ":gray-badge[baselines.py] :gray-badge[02_baselines]\n\n"
    "- Four simple rules set the bar: a model is only worth it if it beats them.\n"
    "- Three learn a little from the training years; naive learns nothing."
)
cards(BASELINES, per_row=2)

st.subheader("7. Models that learn from the inputs")
st.markdown(
    ":gray-badge[models.py] :gray-badge[03_models]\n\n"
    "- Two models learn how the four default inputs relate to the target over the training "
    "years.\n"
    "- A linear regression, a weighted sum of the inputs, and XGBoost, many small decision "
    "trees."
)
cards(MODELS, per_row=2)

st.subheader("8. Test them on years they didn't learn from")
st.markdown(
    ":gray-badge[pipeline.py] :gray-badge[evaluation.py]\n\n"
    "- **Learn:** the forecasters learn from 2003–2017, the training years.\n"
    "- **Test:** on every trading day from 2 January 2018 to 17 September 2026, 2,173 days "
    "(the last with 21 known days after it), each forecasts the next 21 trading days from "
    "what's known at that day's close.\n"
    "- **Refit:** each January, the forecasters that learn are fitted again on all the years "
    "before, as they would be in real life.\n"
    "- **A second start:** every run also scores training from 1997, to show what the crisis "
    "years do."
)
st.plotly_chart(split_chart())
st.markdown(
    "- Each forecast $f_t$ is checked against the volatility $y_t$ that actually followed, "
    "over the $N$ test days.\n"
    "- MAE, RMSE and bias are in pp. For MAE, RMSE and QLIKE, lower is better.\n"
    "- Two choices did look at the test years, the 2003 start and keeping the default inputs "
    "(step 9), so the test scores may be a little optimistic."
)
cards(SCORES)

st.subheader("9. Choose the inputs")
st.markdown(
    ":gray-badge[04_inputs]\n\n"
    "- Each set of inputs was scored on 2013–2017, the validation years, refitting every "
    "January as in the test.\n"
    "- The validation years come before the test years, so a choice made on them doesn't "
    "peek at the test.\n"
    "- Each set was then checked once on the test years.\n"
    "- MAE in pp, with the best on the validation years in green:"
)
validation_columns = pd.Index(["linear regression, 2013–2017 (pp)", "XGBoost, 2013–2017 (pp)"])
inputs_table = INPUT_MAE.style.format(precision=3)
st.table(inputs_table.highlight_min(subset=validation_columns, props=BEST_STYLE))
st.markdown(INPUT_FINDINGS)

st.header("Results", divider="gray")

st.subheader("Which forecaster is best")
# Widths to fit each card's text, so none is cut off on a 1080px screen
best, miss, level, edge = st.columns([5, 3, 4, 3])
best.metric("Best forecaster", "Linear regression", border=True)
miss.metric("Its average miss (MAE)", "6.8 pp", border=True)
level.metric("Average volatility, 2018–2026", "24.9% a year", border=True)
edge.metric(
    "Miss vs GARCH",
    "−4%",
    help="MAE 6.83 pp against GARCH's 7.10 pp: 0.27 pp, or 4%, smaller.",
    border=True,
)
st.markdown("Scores on the test years, best in green:")
scores = TEST_SCORES.style.format(precision=3)
scores = scores.highlight_min(subset=["MAE (pp)", "RMSE (pp)", "QLIKE"], props=BEST_STYLE)
st.table(scores.apply(mark_closest_to_zero, subset=["bias (pp)"]))
st.markdown("Each forecaster against the linear regression, by Diebold–Mariano tests:")
st.table(TEST_VERDICTS)
st.markdown(RESULTS)

st.subheader("Training years and refits")
st.markdown(
    "- Learning from 1997, crisis years included, makes every fitted forecaster forecast too "
    "high, the historical average by 6.5 pp on average, and miss more.\n"
    "- Refitting every year instead of once changes little, but it's how forecasts are made "
    "in practice.\n"
    "- MAE in pp on the test years, best in green:"
)
st.table(TRAINING_MAE.style.format(precision=3).highlight_min(axis=1, props=BEST_STYLE))

st.subheader("What the models learned")
st.markdown(
    "- The linear regression's latest fit, on 2003 to September 2026, is the one behind the "
    "latest forecast:"
)
st.latex(
    r"f = 11.7 + 0.02\,\text{rv\_1d} + 0.12\,\text{rv\_5d} + 0.27\,\text{rv\_21d} "
    r"+ 0.18\,\text{vix\_close}"
)
st.markdown(
    "- The forecast and its inputs are all in % a year, so each weight reads directly: if "
    "rv_21d is 10 pp higher, the forecast is 2.7 pp higher.\n"
    "- The weights on BIST's own volatility add up to 0.41, so the forecast moves less than "
    "half as much as recent volatility. The intercept, 11.7% a year, pulls it toward the "
    "usual level.\n"
    "- The VIX's weight, 0.18, adds what BIST's own past doesn't have: global stress.\n"
    "- XGBoost has no weights. Its importances say how much each input's splits improved its "
    "trees, as shares of the total with no direction: rv_21d 42%, rv_5d 29%, vix_close 20% "
    "and rv_1d 8%."
)

st.subheader("The latest forecast")
st.markdown(
    "- At BIST's close on 16 October 2026, every forecaster, refitted on 2003 up to that "
    "close, forecast the next 21 trading days.\n"
    "- The linear regression says 25.6% a year.\n"
    "- So about two times in three, the index ends those 21 days within ±7.4% of where it "
    "started."
)
st.table(LATEST_FORECASTS.style.format(precision=1))
st.caption(
    "Numbers from the notebooks and the base run: default settings, data up to 2026-10-16. "
    "The Run page shows any saved run, and Compare puts runs side by side with the base run."
)
