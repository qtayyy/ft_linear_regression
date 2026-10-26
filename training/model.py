from pathlib import Path

THETA_PATH = Path(__file__).resolve().parent / "theta_values.txt"

class LinearRegression:
    def __init__(self):
        self.theta0 = 0
        self.theta1 = 0

    def fit(self, df, learning_rate, epochs):
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
        self.theta0 = theta0 - theta1 * mileage_mean / mileage_std
        self.theta1 = theta1 / mileage_std

    def predict(self, km):
        return self.theta0 + self.theta1 * km

    def save(self):
        try:
            with open(THETA_PATH, 'w', encoding='utf-8') as f:
                f.write(f"Theta0: {self.theta0}\n")
                f.write(f"Theta1: {self.theta1}\n")
            print("Values saved to theta_values.txt")
        except Exception as e:
            print(f"Error saving theta values: {e}")

    @classmethod
    def load(cls, path = THETA_PATH):
        model = cls() # Create a new instance of the class
        try:
            with open(path, 'r', encoding='utf-8') as f:
                model.theta0 = float(f.readline().split(':')[1].strip())
                model.theta1 = float(f.readline().split(':')[1].strip())
            print("Theta values loaded from theta_values.txt")
            return model
        except Exception as e:
            print(f"Error loading theta values: {e}")
            return None