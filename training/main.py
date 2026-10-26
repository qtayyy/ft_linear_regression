import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

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

def save_theta_values(theta0, theta1):
    try:
        with open('theta_values.txt', 'w', encoding='utf-8') as f:
            f.write(f"Theta0: {theta0}\n")
            f.write(f"Theta1: {theta1}\n")
    except Exception as e:
        print(f"Error saving theta values: {e}")

def load_data(path = 'data.csv'):
    df = pd.read_csv(path)
    return df

def fit(df, learning_rate, epochs):
    theta0 = 0
    theta1 = 0
    row_count = len(df)
    mileage_mean = df['km'].mean()
    mileage_std = df['km'].std()
    for _ in range(epochs):
        errors = []
        for _, row in df.iterrows():
            mileage = row['km']
            actual_price = row['price']
            mileage_norm = (mileage - mileage_mean) / mileage_std
            predicted_price = theta0 + theta1 * mileage_norm
            error = predicted_price - actual_price
            errors.append((error, mileage_norm))

        # Note that the PDF wants us to absorb the 2 into the learning rate for the slopes
        theta0_slope = sum(error for error, _ in errors) / row_count
        theta1_slope = sum(error * mileage_norm for error, mileage_norm in errors) / row_count

        theta0 -= learning_rate * theta0_slope
        theta1 -= learning_rate * theta1_slope

    theta0_raw = theta0 - theta1 * mileage_mean / mileage_std
    theta1_raw = theta1 / mileage_std
    return theta0_raw, theta1_raw

# R² = Σ(predicted - mean_price)² / Σ(price - mean_price)²: Explained Variation / Total Variation
# Alternatively, R² = 1 - (Σ(predicted - price)² / Σ(price - mean_price)²)
def metrics(df, theta0, theta1):
    predicted_prices = [theta0 + theta1 * km for km in df['km']]
    mean_price = df['price'].mean()
    mse = sum((predicted_price - price) ** 2 for predicted_price, price in zip(predicted_prices, df['price'])) / len(df)
    rmse = math.sqrt(mse)
    ss_tot = sum((price - mean_price) ** 2 for price in df['price'])
    r2 = 1 - sum((predicted_price - price) ** 2 for predicted_price, price in zip(predicted_prices, df['price'])) / ss_tot
    return mse, rmse, r2


def main():
    learning_rate = 0.1
    epochs = 150
    df = load_data("data.csv")
    theta0, theta1 = fit(df, learning_rate, epochs)
    mse, rmse, r2 = metrics(df, theta0, theta1)
    print(f"MSE: {mse}, RMSE: {rmse}, R²: {r2}, Theta0: {theta0}, Theta1: {theta1}")
    save_theta_values(theta0, theta1)
    print("Values saved to theta_values.txt")
    graph(df, lambda x: theta0 + theta1 * x, [0, df["km"].max()])

if __name__ == "__main__":
    main()
# plt.plot(losses)
# plt.show()



