# Classification

## Introduction

- Classification is the task of predicting which class/category an example belongs to.
- Here, we'll focus on binary classification, but multi-class classification is possible too.

## Threshold and the confusion matrix

### Classification threshold

- This is the threshold probability used to assign examples to either the positive class or the negative class.
- If the predicted score equals the classification threshold, how it's handled depends on the particular implementation chosen.
- A threshold of 0.5 isn't a good idea when the cost of one type of misclassification is greater than the other, or when the classes are imbalanced (e.g., only a few examples exist for the positive class).

### Confusion matrix

The probability score is not the ground truth. There are four possible outcomes for each output from a binary classifier:

1. True positive (TP)
2. False positive (FP)
3. True negative (TN)
4. False negative (FN)

The confusion matrix helps us identify the best classification threshold based on the purpose or nature of the classification.

**Note:**

| Quantity            | Formula    |
| ------------------- | ---------- |
| Predicted positives | $TP + FP$  |
| Predicted negatives | $TN + FN$  |
| Real positives      | $TP + FN$  |
| Real negatives      | $FP + TN$  |

## Metrics in classification

### Accuracy

- Proportion of all classifications that were correct: $\frac{TP + TN}{TP + TN + FP + FN}$
- Given a balanced dataset, with similar numbers of examples in both classes, accuracy can serve as a coarse-grained measure of model quality.
- However, when the dataset is imbalanced, or when one kind of mistake (FN or FP) is more costly than the other, it's better to optimize for one of the other metrics instead.
- For heavily imbalanced datasets, where one class appears very rarely (say, 1% of the time), a model that predicts negative 100% of the time would score 99% on accuracy despite being useless.

### Recall / true positive rate

- Proportion of all actual positives that were correctly classified as positives: $\frac{TP}{TP + FN}$
- Also known as the probability of detection, or **sensitivity**.
- In an imbalanced dataset where the number of actual positives is very low, recall is a more meaningful metric than accuracy.

### False positive rate

- Proportion of all actual negatives that were *incorrectly* classified as positives: $\frac{FP}{FP + TN}$
- Also known as the probability of false alarm.
- Related to **specificity** (true negative rate), the proportion of actual negatives correctly classified as negatives: $\frac{TN}{TN + FP} = 1 - FPR$
- Generally also more informative than accuracy in an imbalanced dataset.
- However, it may not be suitable if the number of actual negatives is low (see precision).

### Precision

- Proportion of all the model's positive classifications that are actually positive: $\frac{TP}{TP + FP}$
- In an imbalanced dataset where the number of actual positives is very low, precision is less meaningful as a metric.
- Precision improves as false positives decrease, while recall improves as false negatives decrease.

**Note:** Increasing the classification threshold tends to decrease the number of false positives and increase the number of false negatives, while decreasing the threshold has the opposite effect. As a result, precision and recall often show an inverse relationship, where improving one worsens the other.

## Choice of metric and tradeoffs

| Metric                        | Guidance                                                                                                                                                                                                    |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Accuracy                      | Use as a rough indicator of model training progress/convergence for balanced datasets. For model performance, use only in combination with other metrics. Avoid for imbalanced datasets; consider another metric. |
| Recall (true positive rate)   | Use when false negatives are more expensive than false positives.                                                                                                                                           |
| False positive rate           | Use when false positives are more expensive than false negatives.                                                                                                                                           |
| Precision                     | Use when it's very important for positive predictions to be accurate.                                                                                                                                       |

### F1 score

The F1 score is the harmonic mean (a kind of average) of precision and recall:

$$
F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} = \frac{2TP}{2TP + FP + FN}
$$

This metric balances the importance of precision and recall, and is preferable to accuracy for class-imbalanced datasets. When precision and recall both have perfect scores of 1.0, F1 will also have a perfect score of 1.0. More broadly, when precision and recall are close in value, F1 will be close to their value. When precision and recall are far apart, F1 will be similar to whichever metric is worse.

## ROC and AUC

The metrics in the previous sections are calculated at a single threshold value. The tools below allow us to evaluate a model's quality across all possible thresholds.

### Receiver operating characteristic curve (ROC)

- A visual representation of model performance across all thresholds.
- The name is a holdover from WWII radar detection.
- It is drawn by calculating the true positive rate (TPR) and false positive rate (FPR) at every possible threshold (in practice, at selected intervals), then graphing TPR over FPR.
- A perfect model, which at some threshold has a TPR of 1.0 and an FPR of 0.0, can be represented either by a point at (0, 1) if all other thresholds are ignored, or by a full curve.
    - The curve looks like a square for a perfect model.

### Area under the curve (AUC)

- Represents the probability that the model, if given a randomly chosen positive and negative example, will rank the positive higher than the negative.
- AUC is a useful measure for comparing the performance of two different models, as long as the dataset is roughly balanced.
- The model with the greater area under the curve is generally the better one.

### Precision-recall curves

When the dataset is imbalanced, precision-recall curves (PRCs) and the area under those curves may offer a better comparative visualization of model performance. Precision-recall curves are created by plotting precision on the y-axis and recall on the x-axis across all thresholds.

## Prediction bias

- The difference between the mean of a model's predictions and the mean of the ground-truth labels in the data.
- Can be caused by:
    - Biases or noise in the data, including biased sampling for the training set
    - Too-strong regularization, meaning that the model was oversimplified and lost some necessary complexity
    - Bugs in the model training pipeline
    - The set of features provided to the model being insufficient for the task

## Multi-class classification

- An extension of binary classification to more than two classes.
- Can be done iteratively, e.g., by turning the problem into N separate binary classification problems.
