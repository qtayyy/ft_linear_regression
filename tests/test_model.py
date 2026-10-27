import pandas as pd
import pytest

from precision.main import metrics
from training.model import LinearRegression

# price = 1000 - 0.2 * km exactly, so training must find theta0 = 1000 and theta1 = -0.2
PERFECT_LINE = pd.DataFrame({"km": [0, 1000, 2000], "price": [1000, 800, 600]})


def test_fit_finds_perfect_line() -> None:
    model = LinearRegression()
    model.fit(PERFECT_LINE, learning_rate=0.1, epochs=500)
    assert model.theta0 == pytest.approx(1000)
    assert model.theta1 == pytest.approx(-0.2)


def test_fit_loss_goes_down() -> None:
    losses = LinearRegression().fit(PERFECT_LINE, learning_rate=0.1, epochs=50)
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("km", [[], [1000], [1000, 1000]])
def test_fit_rejects_too_few_mileages(km: list[int]) -> None:
    df = pd.DataFrame({"km": km, "price": [5000] * len(km)})
    with pytest.raises(ValueError):
        LinearRegression().fit(df, learning_rate=0.1, epochs=10)


def test_metrics_perfect_model() -> None:
    model = LinearRegression()
    model.theta0, model.theta1 = 1000, -0.2
    mse, rmse, r2 = metrics(PERFECT_LINE, model)
    assert mse == pytest.approx(0)
    assert rmse == pytest.approx(0)
    assert r2 == pytest.approx(1)


def test_metrics_rejects_same_prices() -> None:
    df = pd.DataFrame({"km": [1000, 2000], "price": [5000, 5000]})
    with pytest.raises(ValueError):
        metrics(df, LinearRegression())
