### How it works

1. **Data:** daily BIST 100, VIX and USD/TRY closes from Yahoo Finance.
2. **Target:** BIST 100 realized volatility over the next 21 trading days, annualized in %. That's about one month, the same window VIX covers.
3. **Forecasts,** each made at BIST's close with only what's known at that moment:
   - **Baselines:** the historical average, naive (the last 21 days' volatility), VIX rescaled to BIST levels, and GARCH(1,1).
   - **Models:** linear regression and XGBoost on the last day's, week's and month's volatility plus VIX.
4. **Testing:** trained from 2003, refitted at the start of each year, and scored on 2018–2026. Every run is saved in History.

### What the data shows

- **Two eras.** In 1997–2002, volatility was often 40–80%; since 2003 it mostly stays at 15–40%. Training starts in 2003, because including the crisis years makes every fitted model over-forecast.
- **Volatility clusters.** Calm and turbulent periods persist, so recent volatility predicts the next month, but it soon falls back toward its long-run level.
- **Fat tails.** Extreme days are far more common than a normal distribution predicts.
- **No leverage effect.** Unlike in most markets, months dominated by down days were followed by lower volatility, not higher.

### Results, with the default settings

- **The linear regression scores best:** it misses by 6.8 volatility points on average, against an average level of 24.9%. That's about 4% better than GARCH, the best baseline.
- **The edge is real against naive, XGBoost and VIX,** borderline against GARCH, and not significant against the historical average. Monthly volatility is only a little predictable beyond its long-run level.
- **More complexity didn't help.** XGBoost doesn't beat the straight line, and extra inputs (quarterly or down-day volatility, USD/TRY) didn't help robustly when chosen on a 2013–2017 validation period.
