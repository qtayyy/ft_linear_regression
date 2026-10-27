import math
from training.model import LinearRegression

model = LinearRegression.load()
if model is None: # untrained: subject requires theta0 = theta1 = 0
    print("Model not trained yet. Using theta0 = 0 and theta1 = 0.")
    model = LinearRegression()

while True: # loop until the user enters a valid mileage
    try:
        mileage = input("Enter the mileage of the car (km): ")
    except (EOFError, KeyboardInterrupt): # Ctrl-D / Ctrl-C: exit cleanly instead of traceback
        print()
        break
    try:
        mileage = float(mileage)
    except ValueError:
        mileage = -1.0 # not a number: fall through to the invalid message
    if not math.isfinite(mileage) or mileage < 0: # isfinite rejects "nan" and "inf", which float() accepts
        print("Invalid mileage value. Mileage must be a positive number.")
        continue

    predicted_price = model.predict(mileage)
    if predicted_price < 0:
        print("Mileage is too high for this model to predict a positive price.")
        continue
    print(f"The predicted price of the car is {predicted_price:.2f} USD")
