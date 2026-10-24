# Introduction
- Models ingest an array of floating-point values called a feature vector.
- These vectors seldom use the dataset's raw values. Instead, they're usually altered througha process called feature engineering.
- Two of the most common technoques are:
    - Normalization: converting numerical values into a standard range.
    - Binning/bucketing: converting numerical values into buckets of ranges.

# Visualizing and evaluating numerical data
- It's good practice to visualize your data using scatter plots/histogram before starting analysis.
- This helps us find patterns or anomalies or outliers hiding in the data.
    - Outliers could be due to a mistake or they could be legeitimate data points.
        - If it's a genuine mistake, deleting them using techniques like clipping could be a good choice.
- In fact, it's often a good idea to visualize your data throughout data transformations.
- Pandas is a good choice.
- It's also a good idea to gather basic statistics (mean, median, std dev) etc.

# Normalization
- The goal of normalization is to transform features to be on a similar scale.
- Three popular normalization methods:
1. Linear scaling
2. Z-score scaling
3. Log scaling

## Benefits
- Helps models converge more quickly during training.
    - Nowadays, advanced optimizers like Adagrad and Adam protect against this problem by changing the effective learning rate over time.
- Helps models infer better predictions. When different features have different ranges, the resulting model might make somewhat less useful predictions.
- Helps avoid the "NaN trap" when feature values are very high. i.e., when a value in a model exceeds the floating-point precision limit.
- Helps the model learn appropriate weights for each feature. Without feature scaling, the model pays too much attention to features with wide ranges and not enough attention to features with narrow ranges.

## Linear scaling
- This method converts numbers into a standard range, usually 0 to 1 or -1 to +1.
    - x' = (x-xmin)/(xmax-xmin)
- It's importany to note that linear scaling preserves the relative distances between the values. It merely changes the numerical scale, not the underlying distribution or relative distances.
- Good choice when:
    - Lower/upper bounds of your data don't change much over time.
        - Important cuz they act like reference points in the compressed number line.
    - Feature contains few/no outliers that are extreme.
    - Feature is approximately uniformly distributed across its range.

## Z-score scaling
- A Z-score is the number of standard deviations a value is from the mean.
    - x' = (x - mu)/sigma
- Z-score is a good choice when the data follows a normal distribution or a distribution somewhat like a normal distribution.

## Log scaling
- Computes the log of the raw values.
    - x' = ln(x)
- Log scaling is helpful when the data conforms to a power law distribution.
    - i.e., when some examples varies according to the power of other examples. 
- It helps the model to not overcompensate for much larger feature values:
    - E.g., book sales conform to a power law distribution because:
        1. most books sell a small number of copies
        2. some sell a moderate number
        3. only a few sell >1 million copies
    - Let's say we want to find the relationship between book covers to book sales. Without log scaling, it doesn't make sense to find sth about book covers on bestselling books that is 10,000 more powerful than the other book covers.

## Clipping
- A technique used to minimize the influence of extreme outliers.
- It caps the value of outliers to a specific max/min value.
    - This means those extreme values now count towards the capped values.
    - Note that outliers that are genuine mistakes should be DELETED not clipped.
- Note that clipping can be done pre- or post-normalization.
- Do NOT clip outliers that are actually important.
