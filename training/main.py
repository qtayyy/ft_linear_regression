import csv
import pandas as pd

theta0 = 0
theta1 = 0
learning_rate = 1.5
epochs = 3000

# batch_size = 10

df = pd.read_csv('data.csv')
row_count = len(df)

for epoch in range(epochs):
    # Errors must be reset for each epoch
    errors = []
    mileage_mean = df['km'].mean()
    mileage_std = df['km'].std()
    for _, row in df.iterrows():
        mileage = row['km']
        actual_price = row['price']
        mileage_norm = (mileage - mileage_mean) / mileage_std
        predicted_price = theta0 + theta1 * mileage_norm
        error = predicted_price - actual_price
        errors.append((error, mileage_norm))

    mse = sum(error**2 for error, _ in errors) / row_count
    # Note that the PDF wants us to absorb the 2 into the learning rate for the slopes
    theta0_slope = sum(error for error, _ in errors) / row_count
    theta1_slope = sum(error * mileage_norm for error, mileage_norm in errors) / row_count

    theta0 -= learning_rate * theta0_slope
    theta1 -= learning_rate * theta1_slope

print(f"MSE: {mse}")
print(f"Theta0: {theta0}")
print(f"Theta1: {theta1}")