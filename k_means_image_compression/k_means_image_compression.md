# Image Compression using K-means Algorithm

K-means is a clustering algorithm. The goal of K-means ('K' stands for number of clusters, or centroids) algorithm is to group similar data points into clusters and detect underlying patterns. K-means is a center-based clustering algorithm. This algorithm uses distance between points (that is values of their features) as a measure of similarity. The objective is to minimize the sum of distances of the points to their respective centroid.

K-means will repeatedly do two different things:
1. assign points to cluster centroids;
2. move cluster centroids.

Detailed explanation of steps in K-means algorithm:
1. Randomly initialize $K$ cluster centroids $\mu_1, \mu_2, ..., \mu_K$, where the dimension of $\mu_K$ equal to the dimension of a training example.
2. Repeat:
   1. Assign points to cluster centroids:
      - for $i=1$ to $m$:
        - $c^{(i)}:=$ index (from 1 to $K$) to cluster centroid closest to $x^{(i)}$ by computing distance between a point and every cluster: 
          - $dist_{x^{(i)}, \mu_k}=min_k||x^{(i)} - \mu_k||^2$
    2. Move cluster centroids:
         - for $k=1$ to $K$:
           - $\mu_k:=$ average (mean) of points assigned to cluster $k$:
             - $\mu_k=\frac{\sum_{i=1}^jx^{(i)}_k}{N_k}$ , where 
               - $x^{(i)}_k=x^{(i)}$'s example assigned to $k$'s centoid
               - $N_k=$ number of examples assigned to $k$'s centoid.

**Distortion - cost function**:

$$J(c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_k)=\frac{1}{m}\sum^m_{i=1}||x^{(i)} - \mu_{c^{(i)}}||^2$$

- $c^{(i)} =$ index of cluster $(1,2,...,K)$ to which example $x^{(i)}$ is currently assigned;
- $\mu_k=$ cluster centroid $k$;
- $\mu_{c^{(i)}}=$ cluster cenriod of cluster to which example $x^{(i)}$ has been assigned.



## Imports

```
import numpy as np
import matplotlib.pyplot as plt
```

## Plotting functions

```
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
```

## K-means Functions

```
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
```

## Testing K-means 

```
# load an example dataset
X = np.load("k_means_image_compression_data.npy")

# set initial centroids
initial_centroids = np.array([[3,3], [6,2], [8,5]])
K = 3

# number of iterations
max_iters = 10

centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
```

## Image Compression

- In a straightforward 24-bit color representation of an image, each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green and blue intensity values. This encoding is often refered to as the RGB encoding.
- Image usually contains thousands of colors, and here we will reduce the number of colors to $K$.
- By making this reduction, it is possible to represent (compress) the photo in an efficient way. 
- Specifically, you only need to store the RGB values of the $K$ selected colors, and for each pixel in the image you now need to only store the index of the color at that location (where only 4 bits are necessary to represent $K$ possibilities).
- We will treat every pixel in the original image as a data example and use the K-means algorithm to find the $K$ colors that best group (cluster) the pixels in the 3- dimensional RGB space. 
- Once we have computed the cluster centroids on the image, we will then use the $K$ colors to replace the pixels in the original image.

### Preprocess Image

```
original_img = plt.imread('bird.png')

plt.imshow(original_img)
print("shape of original_img is:", original_img.shape)
```

- Divide by 255 so that all values are in the range 0 - 1.
- Reshape the image into an $m x 3$ matrix, where $m =$ total number of pixels in a channel. Each row will contain the red, green and blue pixel values. This gives us our dataset matrix X_img that we will use K-Means on.

```
original_img = original_img / 255

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

print(X_img)
```

### K-means on Image

```
K = 16          # number of colors to keep                   
max_iters = 10  # try different values             

# choose random pixels as centroids
initial_centroids = kMeans_init_centroids(X_img, K) 

centroids, idx = run_kMeans(X_img, initial_centroids, max_iters) 

print("shape of idx:", idx.shape)
print("closest centroid for the first five elements:", idx[:5])
print(centroids[idx, :])
```

### Compressing Image

After finding the top $K$ colors to represent the image, we can now
assign each pixel position to its closest centroid using the
*find_closest_centroids* function. 
- This allows us to represent the original image using the centroid assignments of each pixel. 
- We now have significantly reduced the number of bits that are required to describe the image: 
    - the original image required 24 bits for each one of the original pixel locations, resulting in total size of $n \times m \times 24$ bits; 
    - the new representation requires some overhead storage in form of a dictionary of $K$ colors, each of which require 24 bits, but the image itself then only requires 4 bits per pixel location. 
    - the final number of bits used is therefore $K \times 24 + n \times m \times 4$ bits, which corresponds to compressing the original image by about a factor of 6.

```
# represent image in terms of indices
X_recovered = centroids[idx, :] 

# reshape recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape)

print(X_recovered)
```

### Plotting Compressed Image

```
# display original image
fig, ax = plt.subplots(1,2, figsize=(8,8))
plt.axis('off')

ax[0].imshow(original_img*255)
ax[0].set_title('original')
ax[0].set_axis_off()


# display compressed image
ax[1].imshow(X_recovered*255)
ax[1].set_title('compressed with %d colours'%K)
ax[1].set_axis_off()
```
