import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def draw_line(p1, p2, style="-k", linewidth=1):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)


def plot_data_points(X, idx):
    # plots data points in X, coloring them so that those with the same
    # index assignments in idx have the same color
    plt.scatter(X[:, 0], X[:, 1], c=idx)
    

def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    # Plot the examples
    plot_data_points(X, idx)
    
    # Plot the centroids as black 'x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    
    # Plot history of the centroids with lines
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    
    plt.title("Iteration number %d" %i)


def kMeans_init_centroids(X, K):

    """
    this function initializes K centroids that are to be 
        used in K-Means on the dataset X
    
    parameters
    ----------
        X : ndarray
            data points 
        K : int
            number of centroids/clusters
    
    returns
    -------
        centroids : ndarray
            initialized centroids
    """
    
    # randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids


def find_closest_centroids(X, centroids):

    """
    computes the centroid memberships for every example
    
    parameters
    ----------
        X : ndarray (m, n) 
            input values      
        centroids : ndarray
            k centroids
    
    returns
    -------
        idx : array_like (m,)
            closest centroids
    
    """

    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    
    for i in range(X.shape[0]):
        min_distance = np.inf
        
        for j in range(centroids.shape[0]):
            curr_distance = np.linalg.norm(X[i] - centroids[j])
                                     
            if curr_distance < min_distance:
                min_distance = curr_distance
                idx[i] = j
    
    return idx


def compute_centroids(X, idx, K):

    """
    returns the new centroids by computing the means of the 
        data points assigned to each centroid.
    
    parameters
    ----------
        X : ndarray   (m, n)
            data points
        idx : ndarray (m,)
            array containing index of closest centroid for each 
                example in X
            concretely, idx[i] contains the index of 
                the centroid closest to example i
        K : int
            number of centroids
    
    returns
    -------
        centroids : ndarray (K, n)
            new centroids computed
    """
    
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for k in range(K):
        idx_k = [i for i, x in enumerate(idx) if x == k]
        X_k = X[idx_k]
        mu_k = np.mean(X_k, axis=0)
        centroids[k] = mu_k
    
    return centroids


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):

    """
    runs the K-Means algorithm on data matrix X
    where each row of X is a single example
    """

    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    
    for i in range(max_iters):
        
        # output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # for each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # hiven the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)

    plt.show() 

    return centroids, idx


# load an example dataset
X = np.load("k_means_image_compression_data.npy")

# set initial centroids
initial_centroids = np.array([[3,3], [6,2], [8,5]])
K = 3

max_iters = 10
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)


# ------------------------------------------------------------------------------------------
#                               Image Compression
# ------------------------------------------------------------------------------------------

K = 8          # number of colors to keep                   
max_iters = 10  # try different values   
original_img = plt.imread('test2.jpg')

plt.imshow(original_img)
print("shape of original_img is:", original_img.shape)

original_img = original_img / 255

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

print(X_img)          

# choose random pixels as centroids
initial_centroids = kMeans_init_centroids(X_img, K) 
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters) 

print("shape of idx:", idx.shape)
print("closest centroid for the first five elements:", idx[:5])
print(centroids[idx, :])

# represent image in terms of indices
X_recovered = centroids[idx, :] 
# reshape recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape)

print(X_recovered)

# display original image
fig, ax = plt.subplots(1,2, figsize=(8,8))
plt.axis('off')

ax[0].imshow(original_img * 255)
ax[0].set_title('original')
ax[0].set_axis_off()

# display compressed image
ax[1].imshow(X_recovered * 255)
ax[1].set_title('compressed with %d colours'%K)
ax[1].set_axis_off()

X_save = Image.fromarray((X_recovered * 255).astype(np.uint8))
X_save.save("compressed.jpg")
