from training.model import LinearRegression

model = LinearRegression.load()
if model is None:
    print("Model not found. Please train the model first.")
    exit(1)

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
