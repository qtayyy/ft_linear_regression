from collections.abc import Callable
import pandas as pd
import matplotlib.pyplot as plt
from training.model import LinearRegression
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data.csv"

def plot_model(df: pd.DataFrame, formula: Callable[[float], float], x_range: list[float]) -> None:
    y = [formula(x) for x in x_range] # straight line: the two end points are enough
    plt.plot(x_range, y, color='red')
    plt.scatter(df['km'], df['price'])
    plt.xlabel('km')
    plt.ylabel('price').set_color('black')
    plt.title('Car Prices vs. Mileage').set_color('black')

def plot_loss(losses: list[float]) -> None:
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.title('Loss')

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def main() -> None:
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



