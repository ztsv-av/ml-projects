# Anomaly Detection

Anomaly detection is identification of rare items, events or observations which deviate significantly from the majority of the data and do not conform to a well defined notion of normal behaviour. It is primarly unsupervised learning algorithm.

The most common way to carry out anomaly detection is to use **density estimation**. With density estimation, you build a model for the **probability of x**. In other words, the learning algorithm will try to figure out what are the values of the data features that have high probability and what are the values that are less likely or have a lower chance or lower probability of being seen in the data set. Given test example, you would compute the probability of this example. If the probability of test example is less than some small threshold, we will raise a flag to say that this could be an anomaly.

## Gaussian Distribution

Besides density estimation, we will use **Gaussian/normal/bell-shaped** distribution. Probability of number $x$ is determined by a Gaussian with mean $\mu$ and variance $\sigma^2$. If we had an infinite number of examples, we would end up essentially with a bell-shaped curve centered at $\mu$ and determined by parameter $\sigma$.

Probability of a number $x$ with **Univariate Gaussian** distribution:

$$p(x) = \frac{1}{\sqrt{2\pi}\sigma} exp^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

, where

- $\mu = \frac{1}{M}\sum^m_{i=1}x^{(i)}$,
- $\sigma^2=\frac{1}{M}\sum^m_{i=1}(x^{(i)} - \mu)^2$

These are *maximum likelihood estimations for $\mu$ and $\sigma$*. Sometimes people would use $\frac{1}{M - 1}$ instead of $\frac{1}{M}$.

Probability of a number $x$ with **Multivariate Gaussian** distribution:

$$p(x) = (\frac{1}{2\pi})^{\frac{p}{2}} |\sum_{\sigma}|^{-\frac{1}{2}} exp^{(-\frac{1}{2}(x-\mu)^T\sum_{\sigma}^{-1}(x-\mu))}$$

, where

- $p$ - number of features,
- $|\sum_{\sigma}|$ - determinant of the variance-covariance matrix $\sum_{\sigma}$,
- $\sum_{\sigma}^{-1}$ - inverse of the variance-covariance matrix $\sum_{\sigma}$

The lower the variance, the higher and thinner the curve. The higher the variance, the wider the distribution.

Even if one of the features of $x$ is very large or very low compared to feature values of other examples, then still $p(x)$ would be very small, indicating an anomaly (e.g. features probabilities multipled together $0.9 * 0.8 * 0.9 * 0.1 = 0.0648$)

## Algorithm

1. Choose $n$ features $x_i$ that might be indicative of anomalous examples.
2. Fit parameters $\mu_1, \sigma_1^2, ..., \mu_n, \sigma_n^2$:
    - $\mu = \frac{1}{M}\sum^m_{i=1}x^{(i)}_j$,
   - $\sigma^2=\frac{1}{M}\sum^m_{i=1}(x^{(i)}_j - \mu_j)^2$
3. Given new example $x$, compute $p(x)$:
   - $\prod^n_{j=1}p(x_j;\mu_j,\sigma_j^2) = \prod^n_{j=1} \frac{1}{\sqrt{2\pi}\sigma_j} exp^{-\frac{(x_j-\mu_j)^2}{2\sigma_j^2}}$
4. Flag as anomaly if $p(x) < \epsilon$.

## Imports

```
import numpy as np
import matplotlib.pyplot as plt
```

## Data

```
X_train, X_val, y_val = load_data()

print("The first 5 elements of X_train are:\n", X_train[:5])  
print("The first 5 elements of X_val are\n", X_val[:5]) 
print("The first 5 elements of y_val are\n", y_val[:5]) 

print ('The shape of X_train is:', X_train.shape)
print ('The shape of X_val is:', X_val.shape)
print ('The shape of y_val is: ', y_val.shape)

plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b') 

# set the title
plt.title("The first dataset")
# set the y-axis label
plt.ylabel('Throughput (mb/s)')
# set the x-axis label
plt.xlabel('Latency (ms)')
# set axis range
plt.axis([0, 30, 0, 30])

plt.show()
```

## Gaussian/Mean and Variance Estimation

```
def estimate_gaussian(X): 

    """
    calculates mean and variance
        of all features in the dataset
    
    parameters
    ----------
        X: ndarray (m, n)
            data matrix
    
    returns
    -------
        mu: ndarray (n,)
            mean of all features
        var: ndarray (n,)
            variance of all features
    """

    m, n = X.shape
    
    mu = np.sum(X, axis=0)/m
    var = np.sum((X - mu)**2, axis=0)/m
        
    return mu, var

mu, var = estimate_gaussian(X_train)   

# returns the density of the multivariate normal
# at each data point (row) of X_train
p = multivariate_gaussian(X_train, mu, var)

visualize_fit(X_train, mu, var)
```

Now that we have estimated the Gaussian parameters, we can investigate which examples have a very high probability given this distribution and which examples have a very low probability.  

- The low probability examples are more likely to be the anomalies in our dataset. 
- One way to determine which examples are anomalies is to select a threshold based on a cross validation set. 

We will select the threshold $\epsilon$ using the $F_1$ score on a cross validation set.

- For this, we will use a cross validation set
$\{(x_{\rm cv}^{(1)}, y_{\rm cv}^{(1)}),\ldots, (x_{\rm cv}^{(m_{\rm cv})}, y_{\rm cv}^{(m_{\rm cv})})\}$, where the label $y=1$ corresponds to an anomalous example, and $y=0$ corresponds to a normal example. 
- For each cross validation example, we will compute $p(x_{\rm cv}^{(i)})$. The vector of all of these probabilities $p(x_{\rm cv}^{(1)}), \ldots, p(x_{\rm cv}^{(m_{\rm cv})})$ is passed to `select_threshold` in the vector `p_val`. 
- The corresponding labels $y_{\rm cv}^{(1)}, \ldots, y_{\rm cv}^{(m_{\rm cv})}$ are passed to the same function in the vector `y_val`.

## Selecting Anomaly Threshold

```
def select_threshold(y_val, p_val): 

    """
    finds the best threshold to use for selecting outliers 
        based on the results from a validation set (p_val) 
        and the ground truth (y_val)
    
    parameters
    ----------
        y_val: ndarray
            ground truth on validation set
        p_val: ndarray
            results on validation set
        
    returns
    -----: 
        epsilon: float
            threshold chosen 
        F1: float
            F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):

        predictions = (p_val < epsilon)
        
        tp = sum((predictions == 1) & (y_val == 1))
        fp = sum((predictions == 1) & (y_val == 0))
        fn = sum((predictions == 0) & (y_val == 1))
            
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        F1 = (2 * precision * recall)/(precision + recall)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1

p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

# find the outliers in the training set 
outliers = p < epsilon

# visualize the fit
visualize_fit(X_train, mu, var)

# draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)
```

## Anomaly Detection on High Dimensional Dataset

The algorithm is as follows:

1. Estimate the Gaussian parameters ($\mu_i$ and $\sigma_i^2$)
2. Evaluate the probabilities for both the training data `X_train_high` from which you estimated the Gaussian parameters, as well as for the the cross-validation set `X_val_high`. 
3. Finally, it will use `select_threshold` to find the best threshold $\varepsilon$. 

```
X_train_high, X_val_high, y_val_high = load_data_multi()

print ('the shape of X_train_high is:', X_train_high.shape)
print ('the shape of X_val_high is:', X_val_high.shape)
print ('the shape of y_val_high is: ', y_val_high.shape)

# estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)

# evaluate the probabilites for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)

# evaluate the probabilites for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

# find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('best epsilon found using cross-validation: %e'% epsilon_high)
print('best F1 on Cross Validation Set:  %f'% F1_high)
print('# anomalies found: %d'% sum(p_high < epsilon_high))
```
