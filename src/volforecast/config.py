"""Every setting of a run, with the defaults the pipeline and the app start from."""

from dataclasses import dataclass
from typing import Literal

HORIZON = 21  # forecast horizon in trading days, about one month
TEST_START = "2018-01-01"  # first day of the test period

# The saved run the app compares every other run with, by its folder in data/output/runs.
# If it isn't saved (or this is None), the base is the newest run with the default settings.
BASE_RUN: str | None = "2026-10-18_15-32-43"

# Default inputs: realized volatility over the last day, week and month, and the last VIX close
DEFAULT_FEATURES = ("rv_1d", "rv_5d", "rv_21d", "vix_close")

# Every input transform.build_dataset makes, in display order. USD/TRY data starts in 2005.
ALL_FEATURES = ("rv_1d", "rv_5d", "rv_21d", "rv_63d", "rv_down_21d", "vix_close", "rv_usdtry_21d")


def scenario_name(train_start: str) -> str:
    """How a training start is named in the results, for example "from_2003"."""
    return f"from_{train_start[:4]}"


@dataclass(frozen=True)
class Config:
    """Every setting of a run.

    The notebooks use the same defaults; 02 and 03 fit each model once, 04 refits yearly.
    """

    horizon: int = HORIZON
    test_start: str = TEST_START
    train_start: str = "2003-01-01"  # main scenario: tests, model weights, latest forecast
    comparison_train_start: str = "1997-01-01"  # second training start, scored for comparison
    refit: Literal["once", "yearly"] = "yearly"  # when the models are fitted again
    features: tuple[str, ...] = DEFAULT_FEATURES
    garch_distribution: Literal["normal", "t"] = "t"
    xgboost_trees: int = 300
    xgboost_max_depth: int = 3
    xgboost_learning_rate: float = 0.05

    @property
    def scenarios(self) -> dict[str, str]:
        """Scenario name -> training start, for the main and the comparison scenario."""
        starts = (self.train_start, self.comparison_train_start)
        return {scenario_name(start): start for start in starts}

    @property
    def main_scenario(self) -> str:
        return scenario_name(self.train_start)
