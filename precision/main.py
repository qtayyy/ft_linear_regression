import math
from pathlib import Path

import pandas as pd

from training.model import LinearRegression

DATA_PATH = Path(__file__).resolve().parent.parent / "training" / "data.csv"

# R² = Σ(predicted - mean_price)² / Σ(price - mean_price)²: Explained Variation / Total Variation
# Alternatively, R² = 1 - (Σ(predicted - price)² / Σ(price - mean_price)²)
def metrics(df: pd.DataFrame, model: LinearRegression) -> tuple[float, float, float]:
    predicted_prices = [model.predict(km) for km in df['km']]
    mean_price = df['price'].mean()
    ss_res = sum((predicted_price - price) ** 2 for predicted_price, price in zip(predicted_prices, df['price']))
    ss_tot = sum((price - mean_price) ** 2 for price in df['price'])
    if ss_tot == 0: # also true for 0 or 1 rows
        raise ValueError("data needs at least 2 cars with different prices")
    mse = ss_res / len(df)
    return mse, math.sqrt(mse), 1 - ss_res / ss_tot


def main() -> None:
    model = LinearRegression.load()
    if model is None: # untrained: same zero thetas as the predict program
        print("Model not trained yet. Using theta0 = 0 and theta1 = 0.")
        model = LinearRegression()
    mse, rmse, r2 = metrics(pd.read_csv(DATA_PATH), model)
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f} USD (average prediction error)")
    print(f"R²: {r2:.4f} (1 = perfect, 0 = no better than guessing the average price)")

if __name__ == "__main__":
    try:
        main()
    except ValueError as e: # bad data file: short message instead of traceback
        print(f"Error: {e}")
