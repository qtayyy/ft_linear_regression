import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import LinearRegression
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data.csv"

def plot_model(df, formula, x_range):
    x = np.array(x_range)
    y = formula(x)
    plt.plot(x, y, color='red')
    plt.scatter(df['km'], df['price'])
    plt.xlabel('km')
    plt.ylabel('price').set_color('black')
    plt.title('Car Prices vs. Mileage').set_color('black')

def plot_loss(losses):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.title('Loss')

def load_data(path = DATA_PATH):
    df = pd.read_csv(path)
    return df

def main():
    learning_rate = 0.1
    epochs = 150
    df = load_data(DATA_PATH)
    model = LinearRegression()
    losses = model.fit(df, learning_rate, epochs)
    print(f"Theta0: {model.theta0}, Theta1: {model.theta1}")
    model.save()
    plot_model(df, model.predict, [0, df["km"].max()])
    plot_loss(losses)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt: # Ctrl-C while training or with plots open: exit cleanly instead of traceback
        print()



