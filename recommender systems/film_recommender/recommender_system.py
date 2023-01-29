## IMPORTS

import numpy as np
import tensorflow as tf
import pandas as pd
from numpy import loadtxt
from tensorflow import keras

## UTILITY

def normalizeRatings(Y, R):

    """
    preprocess data by subtracting mean rating for every movie (every row)
    only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
        has a rating of 0 on average. Unrated moves then have a mean rating (0)
    returns the mean rating in Ymean.
    """
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 

    return(Ynorm, Ymean)


def load_precalc_params_small():

    file = open('./data/small_movies_X.csv', 'rb')
    X = loadtxt(file, delimiter = ",")

    file = open('./data/small_movies_W.csv', 'rb')
    W = loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_b.csv', 'rb')
    b = loadtxt(file,delimiter = ",")
    b = b.reshape(1,-1)
    
    num_movies, num_features = X.shape
    num_users,_ = W.shape

    return(X, W, b, num_movies, num_features, num_users)
    

def load_ratings_small():

    file = open('./data/small_movies_Y.csv', 'rb')
    Y = loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_R.csv', 'rb')
    R = loadtxt(file,delimiter = ",")

    return(Y,R)


def load_Movie_List_pd():

    """ 
    returns df with and index of movies in the order they are in in the Y matrix
    """

    df = pd.read_csv('./data/small_movie_list.csv', header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()

    return(mlist, df)


## DATASET

X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

print("Y", Y.shape, "R", R.shape)
print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("num_features", num_features)
print("num_movies",   num_movies)
print("num_users",    num_users)

tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
print(f"Average rating for movie 1 : {tsmean:0.3f} / 5" )


## COST FUNCTION

def cofi_cost_func_v(X, W, b, Y, R, lambda_):

    """
    returns the cost for the content-based filtering
    vectorized for speed
    uses tensorflow operations to be compatible with custom training loop

    parameters
    ----------
        X : ndarray (num_movies,num_features))
            matrix of item features
        W : ndarray (num_users,num_features)) 
            matrix of user parameters
        b : ndarray (1, num_users)            
            vector of user parameters
        Y : ndarray (num_movies,num_users)    
            matrix of user ratings of movies
        R : ndarray (num_movies,num_users)    
            matrix, where R(i, j) = 1 
            if the i-th movies was rated by the j-th user
        lambda_ : float
            regularization parameter

    returns
    -------
        J : float
            cost
    """

    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))

    return J

## ADDING OWN RATINGS

movieList, movieList_df = load_Movie_List_pd()

my_ratings = np.zeros(num_movies)   # initialize my ratings

# check the file small_movie_list.csv for id of each movie in the dataset
# and set whatever rating you want for a particular movie
# for example, Toy Story 3 (2010) has ID 2700, 
#   so to rate it "5", you can set 
my_ratings[2700] = 5 

my_ratings[929]  = 5   # Lord of the Rings: The Return of the King, The
my_ratings[246]  = 5   # Shrek (2001)
my_ratings[2716] = 3   # Inception
my_ratings[1150] = 5   # Incredibles, The (2004)
my_ratings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
my_ratings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
my_ratings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
my_ratings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
my_ratings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
my_ratings[2937] = 1   # Nothing to Declare (Rien à déclarer)
my_ratings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

print('\nnew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0 :
        print(f'rated {my_ratings[i]} for {movieList_df.loc[i,"title"]}')

# add new user ratings to Y
Y = np.c_[my_ratings, Y]

# add new user indicator matrix to R
R = np.c_[(my_ratings != 0).astype(int), R]

# normalize the dataset
Ynorm, Ymean = normalizeRatings(Y, R)

## MODEL

num_movies, num_users = Y.shape
num_features = 100

# set initial parameters (W, X)
# use tf.Variable to track these variables
tf.random.set_seed(1337) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64), name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64), name='X')
b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')

# instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

## TRAINING

iterations = 200
lambda_ = 1
for iter in range(iterations):
    # use TensorFlow’s GradientTape
    #   to record the operations used to compute the cost 
    with tf.GradientTape() as tape:

        # compute the cost (forward pass included in cost)
        cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

    # use the gradient tape to automatically retrieve
    #   the gradients of the trainable variables 
    #   with respect to the loss
    grads = tape.gradient(cost_value, [X,W,b])

    # run one step of gradient descent by updating
    #   the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, [X,W,b]))

    # log periodically.
    if iter % 20 == 0:
        print(f"training loss at iteration {iter}: {cost_value:0.1f}")

## RECOMMENDATIONS

# make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

# restore the mean
pm = p + Ymean

# take user 0
# user 0 is us
my_predictions = pm[:,0]

# sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')

for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')

print('\n\noriginal vs predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'original {my_ratings[i]}, predicted {my_predictions[i]:0.2f} for {movieList[i]}')

filter=(movieList_df["number of ratings"] > 20)
movieList_df["pred"] = my_predictions
movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)

