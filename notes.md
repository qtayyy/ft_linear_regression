# Introduction
Linear regression is a statistical technique used to find the relationship between variables.
In ML, linear regression finds relationship between **features** and a **label**
A simple linear regression eqn: y' = b + w1x1
    y' = predicted label (output - dependent var)
    b = bias (aka w0)
    w1 = weight of feature
    x1 = feature (input - independent var)
During training, the model updates the bias and weights that produce the best model based on the given dataset.

# Loss
Metric that describes how wrong a model's predictions are.
The goal of training a model is to minimize the loss.
In statistics and machine learning, loss measures the difference between the predicted and actual values. Two methods to remove the signs:
    - Absolute val
    - Square of difference

## Types of losses
L1 loss: Σ|actual - predicted|
Mean absolute error (MAE): 1/N * Σ|actual - predicted|
L2 loss: Σ(actual - predicted)^2
Mean squared error (MSE): 1/N * Σ(actual - predicted)^2
Root MSE (RMSE): sqrt(1/N * Σ(actual - predicted)^2)

Effect of squaring in L2 loss: When the difference between the prediction and label is large, squaring makes the loss even larger. When the difference is small (less than 1), squaring makes the loss even smaller.

Loss metrics like MAE and RMSE may be preferable to L2 loss or MSE in some use cases because they tend to be more human-interpretable, as they measure error using the same scale as the model's predicted value.

Note: MAE and RMSE can differ quite widely. MAE represents the average prediction error, whereas RMSE represents the "spread" of the errors, and is more skewed by larger errors.
When processing multiple examples at once, we recommend averaging the losses across all the examples, whether using MAE, MSE, or RMSE.

Think about how you want to treat outliers when choosing the best loss function:
    -  MSE moves the model more towards the outliers while MAE doesn't.
    - L2 loss incurs a much higher penalty for an outlier than L1 loss.

# Gradient descent
A mathematical technique that iteratively finds the weights and bias that produce the model with the lowest loss.
The model begins training with randomized weights and biases near zero, and then repeats the following steps:
1. Calculate the loss with the current weight and bias.
2. Determine the direction to move the weights and bias that reduce loss.
3. Move the weight and bias values a small amount (learning rate) in the direction that reduces loss.
4. Return to step one and repeat the process until the model can't reduce the loss any further (i.e., it converges).

## How is step 2 carried out?
1. Calculate the weight and bias derivatives of the loss function of your choice.
2. The signs of the derivatives tell you what happens to the loss if you increase the weight/bias.
    - derivative <0 means its better to increase the value since moving right will decrease the slope.
    - derivative >0 means its better to decrease the value since moving right will incresae the slope.
    - Note that a minimum is guaranteed because a loss function will only produce results that are >= 0.
3. Pick the direction based on value of derivative.

## Model convergence and loss curve
A loss curve can help you see if the model has converged. It has loss on the y-axis and iterations on the x-axis.

# Hyperparameters
Hyperparameters are variables that control different aspects of training. Three common hyperparameters are:
Learning rate
Batch size
Epochs

## Learning rate
Learning rate is a number that influences how quickly the model converges. If the learning rate is too low, the model can take a long time to converge. However, if the learning rate is too high, the model never converges

## Batch size
Batch size is a hyperparameter that refers to the number of examples the model processes before updating its weights and bias.
    - examples: The values of one row of features and possibly a label.

Sometimes, it might not be practical to look at the full batch of the examples in the dataset. Two common techniques to get the right gradient on average without needing to look at every example in the dataset before updating the weights and bias are:
1. Stochastic gradient descent (SGD)
2. Mini-batch stochastic gradient descent

### Stochastic gradient descent (SGD)
Uses only a single example (a batch size of one) per iteration
The usual GD (full batch GD) will use all examples per iteration to calculate the loss.
Given enough iterations, SGD works but is very noisy. "Noise" refers to variations during training that cause the loss to increase rather than decrease during an iteration. The term "stochastic" indicates that the one example comprising each batch is chosen at random.

### Mini-batch stochastic gradient descent (mini-batch SGD)
A compromise between full-batch and SGD. For N number of data points, the batch size can be any number greater than 1 and less than N. The model chooses the examples included in each batch at random, averages their gradients, and then updates the weights and bias once per iteration. Determining the number of examples for each batch depends on the dataset and the available compute resources. In general, small batch sizes behaves like SGD, and larger batch sizes behaves like full-batch gradient descent.

## Epoch
An epoch means that the model has processed every example in the training set once.
For example, given a training set with 1,000 examples and a mini-batch size of 100 examples, it will take the model 10 iterations to complete one epoch.
Training typically requires many epochs. That is, the system needs to process every example in the training set multiple times.

# Exercises
1. Why do we **subtract** the product of a **small amount** and the slope from our old weight/bias to get the new weight/bias?
2. Does a minimum always exist in a loss function? (Concept of convex surface)
3. If the derivative = 0, does this mean you're guaranteed to have found the best fit? (global vs local minima)
4. When doing partial differentiation on a loss function, what does it tell you when the derivative contains other variables that you've not differentiated wrt?
5. Is it advisable to have a loss of 0?
6. Why is it important that the difference between the old model parameters and the new model parameters is proportional to the slope of the loss function?