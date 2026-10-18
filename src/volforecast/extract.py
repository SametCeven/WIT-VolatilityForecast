"""Extract daily price history from Yahoo Finance into data/raw/ as CSV, and read it back."""

from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# CSV file name in data/raw -> Yahoo Finance symbol
TICKERS = {
    "xu100": "XU100.IS",  # BIST 100 index
    "vix": "^VIX",  # Cboe Volatility Index (S&P 500 implied volatility)
    "usdtry": "TRY=X",  # US dollar / Turkish lira (lira per dollar)
}

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def download(symbol: str) -> pd.DataFrame:
    """Daily OHLCV history for one Yahoo Finance symbol, up to yesterday."""
    prices = yf.download(symbol, period="max", progress=False, multi_level_index=False)
    # yfinance logs failed downloads instead of raising, and returns an empty frame
    if prices is None or prices.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {symbol}")
    # Today's row can be a live, unfinished bar (or a weekend placeholder for FX)
    prices = prices.loc[prices.index < pd.Timestamp.today().normalize()]
    prices = prices.rename(columns=str.lower).rename_axis("date")
    return prices[PRICE_COLUMNS]


def download_to_csv(ticker: str) -> Path:
    """Download one entry of TICKERS to data/raw/<ticker>.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ticker}.csv"
    download(TICKERS[ticker]).to_csv(path)
    return path


def download_all() -> None:
    """Download every entry of TICKERS to data/raw, writing the files once all have arrived.

    If one download fails, every file stays as it was, so the tables never mix old and new.
    """
    tables = {ticker: download(symbol) for ticker, symbol in TICKERS.items()}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, prices in tables.items():
        prices.to_csv(RAW_DIR / f"{ticker}.csv")


def load_prices(ticker: str) -> pd.DataFrame:
    """Read data/raw/<ticker>.csv with a parsed date index, downloading it first if missing."""
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        download_to_csv(ticker)
    return pd.read_csv(path, index_col="date", parse_dates=True)


def data_version() -> float:
    """When data/raw last changed: the newest modification time of its CSVs, or 0 without any.

    The app checks it to notice new data, for example after `make data`.
    """
    paths = [RAW_DIR / f"{ticker}.csv" for ticker in TICKERS]
    return max((path.stat().st_mtime for path in paths if path.exists()), default=0.0)


def main() -> None:
    download_all()
    for ticker, symbol in TICKERS.items():
        prices = load_prices(ticker)
        first, last = prices.index[0], prices.index[-1]
        print(f"{ticker} ({symbol}): {len(prices):,} rows, {first:%Y-%m-%d} to {last:%Y-%m-%d}")
        print(f"  missing values: {prices.isna().sum().sum()}")

        # Largest close-to-close moves, to spot bad data like an unadjusted price rescaling
        moves = prices["close"].pct_change()
        largest = moves.loc[moves.abs().nlargest(5).index]
        summary = ", ".join(f"{day:%Y-%m-%d} {move:+.1%}" for day, move in largest.items())
        print(f"  largest daily moves: {summary}")
    print(f"Saved to {RAW_DIR}")


if __name__ == "__main__":
    main()
