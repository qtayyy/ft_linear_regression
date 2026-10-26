import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from model import LinearRegression
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data.csv"

# batch_size = 10

def graph(df, formula, x_range):
    x = np.array(x_range)
    y = formula(x)
    plt.plot(x, y, color='red')
    plt.scatter(df['km'], df['price'])
    plt.xlabel('km')
    plt.ylabel('price').set_color('black')
    plt.title('Car Prices vs. Mileage').set_color('black')
    plt.show()

def load_data(path = DATA_PATH):
    df = pd.read_csv(path)
    return df

# R² = Σ(predicted - mean_price)² / Σ(price - mean_price)²: Explained Variation / Total Variation
# Alternatively, R² = 1 - (Σ(predicted - price)² / Σ(price - mean_price)²)
def metrics(df, model):
    predicted_prices = [model.predict(km) for km in df['km'] if model is not None]
    mean_price = df['price'].mean()
    mse = sum((predicted_price - price) ** 2 for predicted_price, price in zip(predicted_prices, df['price'])) / len(df)
    rmse = math.sqrt(mse)
    ss_tot = sum((price - mean_price) ** 2 for price in df['price'])
    r2 = 1 - sum((predicted_price - price) ** 2 for predicted_price, price in zip(predicted_prices, df['price'])) / ss_tot
    return mse, rmse, r2


def main():
    learning_rate = 0.1
    epochs = 150
    df = load_data(DATA_PATH)
    model = LinearRegression()
    model.fit(df, learning_rate, epochs)
    mse, rmse, r2 = metrics(df, model)
    print(f"MSE: {mse}, RMSE: {rmse}, R²: {r2}, Theta0: {model.theta0}, Theta1: {model.theta1}")
    model.save()
    graph(df, model.predict, [0, df["km"].max()])

if __name__ == "__main__":
    main()
# plt.plot(losses)
# plt.show()



