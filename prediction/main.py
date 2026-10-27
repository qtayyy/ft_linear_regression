from training.model import LinearRegression

model = LinearRegression.load()
if model is None: # untrained: subject requires theta0 = theta1 = 0
    print("Model not trained yet. Using theta0 = 0 and theta1 = 0.")
    model = LinearRegression()

while True: # loop until the user enters a valid mileage
    mileage = input("Enter the mileage of the car (km): ")
    if not mileage.isdigit() or float(mileage) < 0: # check if the mileage is a positive number
            print("Invalid mileage value. Mileage must be a positive number.")
            continue
    mileage = float(mileage)

    predicted_price = model.predict(mileage)
    if predicted_price < 0:
        print("Mileage is too high for this model to predict a positive price.")
        continue
    print(f"The predicted price of the car is {predicted_price:.2f} USD")
