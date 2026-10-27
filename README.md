# ft_linear_regression

Predict a car's price from its mileage with linear regression.

`estimatePrice(mileage) = θ0 + θ1 × mileage`

Training starts from θ0 = θ1 = 0, reads `training/data.csv` (`km`, `price`), and writes `training/theta_values.txt`. That file is gitignored, so a fresh clone predicts 0 until you train.

## Requirements

- [uv](https://docs.astral.sh/uv/)

The first `make` run installs Python 3.13, matplotlib, numpy, and pandas from the lockfile. It needs network access.

## Usage

```sh
make train      # fit the model, save thetas, show the data and the loss
make predict    # ask for a mileage and print the estimated price
make precision  # MSE, RMSE, and R² on data.csv
```

`make train` blocks on the plots. Close both windows to exit. Ctrl-C exits cleanly.

`make predict` keeps asking until the mileage is a finite number ≥ 0. Ctrl-D or Ctrl-C exits. A mileage that would produce a negative price is rejected.

## Why mileage is normalized during training

Normalization puts a feature on a similar scale to everything else in the model. Here the feature is mileage, which goes up to 240,000 km while prices are only a few thousand.

In the update formulas, θ1's step is multiplied by `mileage[i]`, so it comes out ~100,000× bigger than θ0's step. Both share one learning rate, which leaves two bad options:

- A learning rate that suits θ0 makes θ1 overshoot further every epoch. The values grow past what a float can hold and become `nan`. This is the **NaN trap**.
- A learning rate that suits θ1 moves θ0 by cents per epoch, so the model takes millions of epochs to **converge**.

Normalizing fixes both: the model converges quickly and stays out of the NaN trap.

### Which normalization method

Three popular methods, and how each fits mileage in `data.csv`:

| Method          | Formula                           | Good choice when                                               | Mileage in `data.csv`                                      |
| --------------- | --------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------- |
| Linear scaling  | `x' = (x - xmin) / (xmax - xmin)` | Few or no extreme outliers, roughly uniform across its range   | Not uniform: half the cars sit between 59k and 95k km      |
| Z-score scaling | `x' = (x - μ) / σ`                | Roughly normal distribution (bunched in the middle)            | Bunched around 65k–140k km, with a tail of high mileages   |
| Log scaling     | `x' = ln(x)`                      | Power law distribution (most values small, a few huge)         | Not a power law                                            |

Mileage is not perfectly normal either: it leans right (skew 1.03), and the 240,000 km car sits 2.64 standard deviations above the mean. Still, it is much closer to "bunched in the middle" than to "spread evenly", so **z-score scaling** is the better fit of the three.

Log scaling is out for a second reason too. The subject requires `estimatePrice = θ0 + θ1 × mileage`. Linear scaling and z-score scaling only shift and stretch mileage, so the thetas can be converted back to raw km afterwards (see below). A log would make the line curve, and no θ0 and θ1 on raw km could reproduce it.

### Why z-score beats linear scaling for gradient descent

Linear scaling also solves the size problem. The difference is **where 0 sits**.

**With linear scaling, every mileage is between 0 and 1, so every value is positive.** Tilting the line (changing θ1) then pushes *every* prediction in the same direction, so it also raises or lowers the line as a whole. That is θ0's job. The two thetas get in each other's way: θ0 moves the line up, θ1's change moves it down again, θ0 corrects, and so on. Training zig-zags instead of heading straight for the answer.

**With z-score scaling, the average mileage becomes 0.** The values below 0 and above 0 balance out. Tilting the line now pivots it around the average car, like a seesaw balanced at its centre: one side goes up as much as the other goes down, so the average prediction does not move. θ0 handles up/down and θ1 handles tilt, and neither undoes the other's work.

Measured on `data.csv` with the same learning rate (0.1):

| Scaling         | θ1 after 150 epochs | Epochs to get within 0.1% of the exact answer |
| --------------- | ------------------- | --------------------------------------------- |
| Z-score scaling | -0.021449 (correct) | 69                                            |
| Linear scaling  | -0.006758 (wrong)   | 1462                                          |

Linear scaling converges eventually, but needs about 20× more epochs. A bigger learning rate narrows the gap but does not close it, because the zig-zag comes from the two thetas overlapping, not from the step size.

Linear scaling also depends on only two cars, the lowest-mileage and the highest-mileage one. Those two act as the reference points for the whole scale, so one extreme car at either end changes the scaling for everyone. Z-score scaling uses the mean and standard deviation of all cars, so one extreme car moves it less.

### Converting back to raw km

`fit()` trains on `(mileage - mean) / std`. After z-score scaling, a car's value is the number of standard deviations it sits from the mean. On `data.csv` the mean is 101,066 km and the standard deviation is 52,674 km, so the scaled values range from -1.48 to +2.64.

At the end of training, the thetas are converted back to raw km:

```python
self.theta0 = theta0 - theta1 * mileage_mean / mileage_std
self.theta1 = theta1 / mileage_std
```

The saved thetas therefore work with the subject's formula on raw km, and `predict` never needs to know about the normalization. The final line is the same one gradient descent on raw mileage would reach eventually.
