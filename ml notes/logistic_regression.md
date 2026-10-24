# Logistic Regression

## Introduction

- Used when you want to predict the probability of a given outcome.
- Usually used for cases where the outcome is binary (spam or not spam).
- It is built on top of linear regression, so it's only used when we believe the outcome can be modelled using a linear model.

## Calculating a probability

- The idea of logistic regression is to still model the outcome using a linear model, e.g., $z = b + w_1 x_1$, BUT to then pass it through a logistic function with range (0, 1).
- This is done using the sigmoid function: $f(z) = \frac{1}{1 + e^{-z}}$
    - Note that $z$ is also known as the log odds, $\ln\left(\frac{p}{1 - p}\right)$, where $p$ is the probability of "success".
    - The log odds is the inverse of the sigmoid function, which you can prove algebraically from $f(z) = p = \frac{1}{1 + e^{-z}}$.

## Loss and regularization

### Log loss

- In logistic regression, we use log loss as the score to judge how good our model's predictions are.
- How log loss works:
    1. For each data point, we note down how likely the outcome is according to our current model.
        - **Example:**
            - Email A is spam and email B is not spam.
            - Our model gave $p = 0.9$ for email A and $p = 0.2$ for email B.
            - But the number we should care about is the probability the model gave to the TRUE outcome. This means we should use $1 - p = 0.8$ for email B.
    2. We then multiply those $p$ and $(1 - p)$ values together, which gives us the probability that the model would have produced exactly the answers we observed.
        - A better model gives a bigger number, and training means picking the model that maximizes this number (theoretical max is 1).
    3. We then take the log of the result to make it practical.
        - Multiplying probabilities can give us numbers that are very small, which can be a problem in several ways:
            - **Memory/precision:** numbers with too many decimal places can underflow, giving us zeros. This makes it difficult to tell whether a model is improving or just plain bad.
            - It's also easier to work with sums than products when finding the downhill direction for training.
        - This gives us: $\sum_{i} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$
            - $y_i$ is the label of the i-th example. It is either 0 or 1 and acts as a "switch".
    4. We then flip the sign, as the convention is to minimize the loss instead of maximizing it.
    5. Finally, we take the average of the sum of logs. This helps with a few things:
        - Allows us to compare loss results across different batch sizes.
            - This also allows us to use similar learning rates across batches.
        - Allows us to be consistent with our regularization (see below).
        - Final formula: $-\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$

### Regularization

- A mechanism for penalizing model complexity during training.
    - How much flexibility does the model have to fit the training data?
        - Size of weights
        - Number of active features
- Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in cases where the model has a large number of features.
- Two strategies to decrease model complexity:
    - L2 regularization
    - Early stopping: limiting the number of training steps to halt training while loss is still decreasing.
- We will revisit regularization later.
