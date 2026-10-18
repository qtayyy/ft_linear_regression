# Introduction
- Used when what you want to predict the probability of a given outcome.
- Usually used for cases where the outcome is binary (Spam or not spam).
- It is built on top of linear regression so it's only used when we believe that the outcome can be modelled using a linear model.

# Calculating a probability
- The idea of logistic regression is to still model the outcome using a linear model e.g., z = b + w1x1 BUT to then convert it into a logistic function with range [0, 1].
- This is done using a sigmoid function: f(z) = 1/(1 + e^-z).
    - Note that z is also known as the log odds, ln(p/1-p) where p is the probability of 'success'.
    - It is the inverse of the sigmoid function and you can prove it algebraically since f(z) = p = 1/(1 + e^-z)

# Loss and regularization
## Log loss
- In logistic regression, we use log loss as the score to judge how good our model's predictions are.
- How log loss works:
1. For each data point we have, we note down how likely an outcome is according to our current model.
    Example 1:
    - Email A is spam and email B is not spam.
    - Our model gave p = 0.9 for email A and p = 0.2 for email B.
    - But the number we should care about is the probability the model gave to the TRUE outcome. This means we should use (1 - p) = 0.8 for email B.
2. We then multiply those p's and (1 - p)'s together, which gives us the probability that the model would have produced exactly the answers we observed.
    - A better model gives a bigger number and training means picking the model that maximizes this number (Theoretical max is 1).
3. We then log the result to make it practical.
    - Multiplying probabilities can give us number that are very small which can be a problem in many aspects:
        - Memory/precision: Numbers with too many decimal points can result in underflow, giving us zeros. Difficult to tell if a model is improving or just plain bad.
        - It's also easier to work with sums than products when finding the downhill direction for training.
    - This gives us: Σᵢ [ yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ) ]
        - yᵢ is the label of the i-th example. It is either 0 or 1 and acts as a 'switch'
4. When then flip the sign as it's a convention to minimize the loss, instead of maximizing it.
5. And finally, we take the average of the sum of logs. This helps with a few things:
    - Allows us to compare loss results across different batch sizes.
        - This also allows us to use similar learning rates across the batches.
    - Allows us to be consistent with our regularization*.
    - −(1/N) Σᵢ [ yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ) ]

## Regularization
- A mechanism for penalizing model complexity during training.
    - How much flexibility model has to fit training data?
        - Size of weights
        - Num of active features
- Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in cases where the model has a large number of features.
- Two strategies to decrease model complexity:
    - L2 regularization
    - Early stopping: Limiting the number of training steps to halt training while loss is still decreasing.
- We will revisit regularization later.