"""Download daily price history from Yahoo Finance and cache it as CSV."""

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# File name in data/raw -> Yahoo Finance symbol
TICKERS = {
    "xu100": "XU100.IS",  # BIST 100 index
    "vix": "^VIX",  # Cboe Volatility Index (S&P 500 implied volatility)
    "usdtry": "TRY=X",  # US dollar / Turkish lira (lira per dollar)
}


def download(symbol: str) -> pd.DataFrame:
    """Fetch the daily OHLCV history for one Yahoo Finance symbol, up to yesterday."""
    df = yf.download(symbol, period="max", progress=False, multi_level_index=False)
    # yfinance logs failed downloads instead of raising, and returns an empty frame
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {symbol}")
    # Today's row can be a live, unfinished bar (or a weekend placeholder for FX)
    df = df.loc[df.index < pd.Timestamp.today().normalize()]
    return df[["Open", "High", "Low", "Close", "Volume"]].rename_axis("Date")


def save(name: str) -> None:
    """Download one entry of TICKERS to data/raw/<name>.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download(TICKERS[name]).to_csv(DATA_DIR / f"{name}.csv")


def load(name: str) -> pd.DataFrame:
    """Read data/raw/<name>.csv with a parsed date index, downloading it first if missing."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        save(name)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def main() -> None:
    for name, symbol in TICKERS.items():
        save(name)
        df = load(name)
        start, end = df.index[0], df.index[-1]
        print(f"{name} ({symbol}): {len(df):,} rows, {start:%Y-%m-%d} to {end:%Y-%m-%d}")
        print(f"  missing values: {df.isna().sum().sum()}")

        # Largest close-to-close moves, to spot bad data like an unadjusted price rescaling
        returns = df["Close"].pct_change()
        top = returns.loc[returns.abs().nlargest(5).index]
        moves = ", ".join(f"{d:%Y-%m-%d} {r:+.1%}" for d, r in top.items())
        print(f"  largest daily moves: {moves}")
    print(f"Saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
