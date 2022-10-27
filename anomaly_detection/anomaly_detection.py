import numpy as np
import matplotlib.pyplot as plt

X_train = np.load("X_part1.npy")
X_val = np.load("X_val_part1.npy")
y_val = np.load("y_val_part1.npy")

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


def multivariate_gaussian(X, mu, var):

    """
    computes the probability 
        density function of the examples X under the multivariate gaussian 
        distribution with parameters mu and var
    if var is a matrix, it is
        treated as the covariance matrix
    if var is a vector, it is treated
        as the var values of the variances in each dimension 
        (a diagonal covariance matrix)
    """
    
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p


def visualize_fit(X, mu, var):

    """
    this visualization shows you the 
        probability density function of the Gaussian distribution
    each example
        has a location (x1, x2) that depends on its feature values
    """
    
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), linewidths=1)
        
    # Set the title
    plt.title("The Gaussian contours of the distribution fit to the dataset")
    # Set the y-axis label
    plt.ylabel('Throughput (mb/s)')
    # Set the x-axis label
    plt.xlabel('Latency (ms)')


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
    
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
        
    return mu, var

mu, var = estimate_gaussian(X_train)   

# returns the density of the multivariate normal
# at each data point (row) of X_train
p = multivariate_gaussian(X_train, mu, var)

visualize_fit(X_train, mu, var)


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

# ------------------------------------------------------------------------------------------
#                                       Bigger Data
# ------------------------------------------------------------------------------------------

X_train_high = np.load("X_part2.npy")
X_val_high = np.load("X_val_part2.npy")
y_val_high = np.load("y_val_part2.npy")

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
