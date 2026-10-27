from pathlib import Path
from typing import Self
import pandas as pd

THETA_PATH = Path(__file__).resolve().parent / "theta_values.txt"

class LinearRegression:
    def __init__(self) -> None:
        self.theta0: float = 0.0
        self.theta1: float = 0.0

    def fit(self, df: pd.DataFrame, learning_rate: float, epochs: int) -> list[float]:
        theta0 = 0.0
        theta1 = 0.0
        mileage_mean = df['km'].mean()
        mileage_std = df['km'].std()
        # whole-column math: each line below works on every row at once
        mileage_norm = (df['km'] - mileage_mean) / mileage_std
        losses: list[float] = []
        for _ in range(epochs):
            errors = theta0 + theta1 * mileage_norm - df['price']
            # Note that the PDF wants us to absorb the 2 into the learning rate for the slopes
            theta0_slope = errors.mean()
            theta1_slope = (errors * mileage_norm).mean()
            theta0 -= learning_rate * theta0_slope
            theta1 -= learning_rate * theta1_slope
            losses.append(float((errors ** 2).mean()))
        self.theta0 = theta0 - theta1 * mileage_mean / mileage_std
        self.theta1 = theta1 / mileage_std
        return losses

    def predict(self, km: float) -> float:
        return self.theta0 + self.theta1 * km

    def save(self) -> None:
        try:
            with open(THETA_PATH, 'w', encoding='utf-8') as f:
                f.write(f"Theta0: {self.theta0}\n")
                f.write(f"Theta1: {self.theta1}\n")
            print(f"Values saved to {THETA_PATH.name}")
        except OSError as e: # disk full, no permission, etc.
            print(f"Error saving theta values: {e}")

    @classmethod
    def load(cls, path: Path = THETA_PATH) -> Self | None:
        model = cls() # Create a new instance of the class
        try:
            with open(path, 'r', encoding='utf-8') as f:
                model.theta0 = float(f.readline().split(':')[1].strip())
                model.theta1 = float(f.readline().split(':')[1].strip())
            print(f"Theta values loaded from {path.name}")
            return model
        except FileNotFoundError: # not trained yet: normal, callers print their own message
            return None
        except (OSError, ValueError, IndexError) as e: # unreadable, not a number, or missing ":"
            print(f"Error loading theta values: {e}")
            return None
