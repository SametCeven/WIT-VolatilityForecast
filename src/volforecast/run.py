"""Entry point for `make pipeline`: run with the default settings, save, and print a summary."""

from volforecast.config import Config
from volforecast.pipeline import run, save_run


def main() -> None:
    config = Config()
    results = run(config)
    folder = save_run(results, config)
    print("Scores on the test period:")
    print(results["scores"].round(3).to_string())
    print("\nDiebold–Mariano tests against the linear regression (statistic, p-value):")
    print(results["significance_tests"].round(4).to_string())
    print(f"\nVolatility forecast for the next {config.horizon} trading days (annualized %):")
    print(results["latest_forecast"].round(1).to_string())
    print(f"\nSaved to {folder}")


if __name__ == "__main__":
    main()
