# Linear Regression

## Introduction

Linear regression is a statistical technique used to find the relationship between variables. In ML, linear regression finds the relationship between **features** and a **label**.

A simple linear regression equation:

$$
y' = b + w_1 x_1
$$

- $y'$ = predicted label (output, dependent variable)
- $b$ = bias (aka $w_0$)
- $w_1$ = weight of the feature
- $x_1$ = feature (input, independent variable)

During training, the model updates the bias and weights to produce the best model for the given dataset.

## Loss

A metric that describes how wrong a model's predictions are. The goal of training a model is to minimize the loss.

In statistics and machine learning, loss measures the difference between the predicted and actual values. Two methods to remove the signs:

- Absolute value
- Square of the difference

### Types of losses

| Loss                      | Formula                                            |
| ------------------------- | -------------------------------------------------- |
| L1 loss                   | $\sum_{i} \lvert y_i - y'_i \rvert$                |
| Mean absolute error (MAE) | $\frac{1}{N} \sum_{i} \lvert y_i - y'_i \rvert$   |
| L2 loss                   | $\sum_{i} (y_i - y'_i)^2$                          |
| Mean squared error (MSE)  | $\frac{1}{N} \sum_{i} (y_i - y'_i)^2$             |
| Root MSE (RMSE)           | $\sqrt{\frac{1}{N} \sum_{i} (y_i - y'_i)^2}$      |

Where $y_i$ is the actual label and $y'_i$ is the predicted label for the i-th example.

**Effect of squaring in L2 loss:** when the difference between the prediction and the label is large, squaring makes the loss even larger. When the difference is small (less than 1), squaring makes the loss even smaller.

Loss metrics like MAE and RMSE may be preferable to L2 loss or MSE in some use cases because they tend to be more human-interpretable, as they measure error on the same scale as the model's predicted value.

**Note:** MAE and RMSE can differ quite widely. MAE represents the average prediction error, whereas RMSE represents the "spread" of the errors and is more skewed by larger errors.

When processing multiple examples at once, we recommend averaging the losses across all the examples, whether using MAE, MSE, or RMSE.

Think about how you want to treat outliers when choosing the best loss function:

- MSE moves the model more towards the outliers, while MAE doesn't.
- L2 loss incurs a much higher penalty for an outlier than L1 loss.

## Gradient descent

A mathematical technique that iteratively finds the weights and bias that produce the model with the lowest loss.

The model begins training with randomized weights and biases near zero, then repeats the following steps:

1. Calculate the loss with the current weight and bias.
2. Determine the direction to move the weights and bias that reduces loss.
3. Move the weight and bias values a small amount (learning rate $\alpha$) in the direction that reduces loss:

    $$
    w \leftarrow w - \alpha \frac{\partial L}{\partial w}, \qquad b \leftarrow b - \alpha \frac{\partial L}{\partial b}
    $$

4. Return to step 1 and repeat the process until the model can't reduce the loss any further (i.e., it converges).

### How is step 2 carried out?

1. Calculate the derivatives of your chosen loss function with respect to the weight and bias.
2. The signs of the derivatives tell you what happens to the loss if you increase the weight/bias.
    - $\frac{\partial L}{\partial w} < 0$ means it's better to increase the value, since moving right will decrease the loss.
    - $\frac{\partial L}{\partial w} > 0$ means it's better to decrease the value, since moving right will increase the loss.
    - Note that $L \geq 0$ only means the loss is bounded below; it doesn't guarantee a minimum is ever reached (e.g., logistic regression loss can approach 0 forever without reaching it). For linear regression with MSE, a minimum does exist because the loss is a convex bowl (quadratic) in the weights and bias.
3. Pick the direction based on the sign of the derivative.

### Model convergence and loss curve

A loss curve can help you see if the model has converged. It has loss on the y-axis and iterations on the x-axis.

## Hyperparameters

Hyperparameters are variables that control different aspects of training. Three common hyperparameters are:

- Learning rate
- Batch size
- Epochs

### Learning rate

Learning rate is a number that influences how quickly the model converges. If the learning rate is too low, the model can take a long time to converge. However, if the learning rate is too high, the model never converges.

### Batch size

Batch size is a hyperparameter that refers to the number of examples the model processes before updating its weights and bias.

- **Example:** the values of one row of features and possibly a label.

Sometimes it might not be practical to look at the full batch of examples in the dataset. Two common techniques to get the right gradient on average, without needing to look at every example in the dataset before updating the weights and bias, are:

1. Stochastic gradient descent (SGD)
2. Mini-batch stochastic gradient descent

#### Stochastic gradient descent (SGD)

- Uses only a single example (a batch size of one) per iteration.
- The usual GD (full-batch GD) uses all examples per iteration to calculate the loss.
- Given enough iterations, SGD works but is very noisy. "Noise" refers to variations during training that cause the loss to increase rather than decrease during an iteration. The term "stochastic" indicates that the one example comprising each batch is chosen at random.

#### Mini-batch stochastic gradient descent (mini-batch SGD)

A compromise between full-batch GD and SGD. For N data points, the batch size can be any number greater than 1 and less than N. The model chooses the examples included in each batch at random, averages their gradients, and then updates the weights and bias once per iteration. The number of examples for each batch depends on the dataset and the available compute resources. In general, small batch sizes behave like SGD, and larger batch sizes behave like full-batch gradient descent.

### Epochs

An epoch means that the model has processed every example in the training set once.

For example, given a training set with 1,000 examples and a mini-batch size of 100 examples, it will take the model 10 iterations to complete one epoch.

Training typically requires many epochs; that is, the system needs to process every example in the training set multiple times.

## Exercises

Click each question to reveal a sample answer.

<details>
<summary>1. Why do we <b>subtract</b> the product of a <b>small amount</b> and the slope from our old weight/bias to get the new weight/bias?</summary>

- The slope (gradient) points in the direction where the loss **increases** fastest. Subtracting it moves us the opposite way, i.e., downhill towards lower loss.
- We multiply by a **small amount** (the learning rate $\alpha$) because the slope only describes the loss near the current point. A big step can overshoot the minimum and land somewhere with a higher loss.

$$
w_{\text{new}} = w_{\text{old}} - \alpha \frac{\partial L}{\partial w}
$$

</details>

<details>
<summary>2. Does a minimum always exist in a loss function? (Concept of a convex surface)</summary>

- Not always. $L \geq 0$ means the loss can't go below 0, but it may only get closer and closer to some value without reaching it (e.g., logistic regression on perfectly separable data, where loss keeps approaching 0 as the weights grow).
- If the loss is **convex** (a bowl shape), like MSE for linear regression, a minimum exists and any local minimum is also the global minimum.
- If the bowl is **strictly** convex (e.g., no feature is an exact copy or combination of others), that minimum is unique.

</details>

<details>
<summary>3. If the derivative = 0, does this mean you're guaranteed to have found the best fit? (Global vs. local minima)</summary>

- No. A derivative of 0 only means the surface is flat at that point (a stationary point). It could be:
    - a **global minimum** (the best fit),
    - a **local minimum** (best in the neighbourhood, but not overall),
    - a **local maximum**, or
    - a **saddle point** (minimum in one direction, maximum in another).
- For a convex loss like MSE in linear regression, the only stationary point is the global minimum, so there it **is** guaranteed.
- In practice, gradient descent rarely hits exactly 0; it stops when the loss stops improving meaningfully.

</details>

<details>
<summary>4. When doing partial differentiation on a loss function, what does it tell you when the derivative contains other variables that you haven't differentiated with respect to?</summary>

- It means the parameters are **coupled**: the slope in one direction depends on the current value of the others. For MSE:

$$
\frac{\partial L}{\partial w_1} = -\frac{2}{N} \sum_{i} x_i \left( y_i - (b + w_1 x_i) \right), \qquad
\frac{\partial L}{\partial b} = -\frac{2}{N} \sum_{i} \left( y_i - (b + w_1 x_i) \right)
$$

- Both contain $b$ and $w_1$, so the best $w_1$ depends on $b$ and vice versa. You can't optimize each one in isolation.
- This is why all gradients are computed using the **old** values first, then all parameters are updated at the same time (simultaneous update).

</details>

<details>
<summary>5. Is it advisable to have a loss of 0?</summary>

- Usually not, at least on the training data. Real data contains noise, so a loss of 0 usually means the model has **memorized** the training set (overfitting) and will generalize poorly to new data.
- What matters is a low loss on **unseen** (validation/test) data.
- In logistic regression, a loss of 0 would require infinitely large weights, which is why regularization is used.

</details>

<details>
<summary>6. Why is it important that the difference between the old model parameters and the new model parameters is proportional to the slope of the loss function?</summary>

- Far from the minimum, the slope is steep, so steps are big and progress is fast.
- Near the minimum, the slope flattens, so steps shrink automatically and the model settles in instead of jumping over the minimum.
- At the minimum, the slope is 0, so the step is 0 and the parameters stop changing. That is what convergence means.
- With a fixed step size instead, the model would keep bouncing around the minimum and never settle.

</details>
