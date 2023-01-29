# Recommender System

A recommender system, or a recommendation system (sometimes replacing 'system' with a synonym such as platform or engine), is a subclass of information filtering system that provide suggestions for items that are most pertinent to a particular user. In a very general way, recommender systems are algorithms aimed at suggesting relevant items to users (items being movies to watch, text to read, products to buy or anything else depending on industries).

In the typical recommender system, you have some number of users as well as some number of items (e.g. movies). You might as well have features related to those items (e.g. how much a movie related to romance, action, etc. genres)

## Notation

| General Notation | Description | Python (if any) |
| ---------------- |-------------| --------------- |
| $r(i,j)$     | scalar; = 1  if user j rated movie i  = 0  otherwise             ||
| $y(i,j)$     | scalar; = rating given by user j on movie  i    (if r(i,j) = 1 is defined) ||
|$\mathbf{w}^{(j)}$ | vector; parameters for user j ||
|$b^{(j)}$     |  scalar; parameter for user j ||
| $\mathbf{x}^{(i)}$ |   vector; feature ratings for movie i        ||     
| $n_u$        | number of users |num_users|
| $n_m$        | number of movies | num_movies |
| $n$          | number of features | num_features                    |
| $\mathbf{X}$ |  matrix of vectors $\mathbf{x}^{(i)}$         | X |
| $\mathbf{W}$ |  matrix of vectors $\mathbf{w}^{(j)}$         | W |
| $\mathbf{b}$ |  vector of bias parameters $b^{(j)}$ | b |
| $\mathbf{R}$ | matrix of elements $r(i,j)$                    | R |

## Collaborative Filtering Algorithm

The goal of a collaborative filtering recommender system is to generate two vectors: for each user, a 'parameter vector' that embodies the movie tastes of a user. For each movie, a feature vector of the same size which embodies some description of the movie. The dot product of the two vectors plus the bias term should produce an estimate of the rating the user might give to that movie.

For user $j$ and movie $i$, predict rating: $w^{(j)} * x^{(i)} + b^{(j)}$

To learn $w^{j}, b^{j}$ for a single user use **Mean Squared Error**:

$min_{w^{(j)}b^{(j)}}J(w^{(j)}, b^{(j)}) = \frac{1}{2m^{(j)}}\sum_{i:r(i,j)=1}(w^{(j)} * x^{(i)} + b^{(j)}-y^{(i, j)})^2  + \frac{\lambda}{2m^{(j)}}\sum_{k=1}^n(w_k^{(j)})^2$

You might remove $m^{(j)}$ term. It a constant and it does not affect cost function much.

So, overall cost function for all users looks like this (without learning features):

$$J(w^{(1)}, ..., w^{(n_u)}, b^{(1)}, ..., b^{(n_u)}) = \frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}(w^{(j)} * x^{(i)} + b^{(j)}-y^{(i, j)})^2  + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n(w_k^{(j)})^2$$

The difference between simple linear regression model is that now we train different linear regression unit for each user.

Collaboratiove filtering algorithm is used to  **learn** **parameters** as well as **features**.

Given $w^{(1)}, ..., w^{(n_u)}, b^{(1)}, ..., b^{(n_u)}$, **cost function** to learn $x^{(1)}, ..., x^{(n_m)}$:

$J(x^{(1)}, ..., x^{(n_m)}) = \frac{1}{2}\sum_{i=1}^{n_m}\sum_{j:r(i,j)=1}(w^{(j)} * x^{(i)} + b^{(j)}-y^{(i, j)})^2  + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n(x_k^{(i)})^2$

Putting parameters and features cost functions together, we have a new cost function that we want to minimize, based on parameter values $w, b$ and features values $x$ for each user:

$$J(w, b, x) = \frac{1}{2}\sum_{(i, j):r(i,j)=1}(w^{(j)} * x^{(i)} + b^{(j)}-y^{(i, j)})^2  + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n(w_k^{(j)})^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n(x_k^{(i)})^2$$

To minimize cost function, use **Gradient Descent**:

$repeat \space\{$

- $w_i^{(j)}=w_i^{(j)}-\alpha \frac{\partial}{\partial w_i^{(j)}}J(w,b,x)$

- $b^{(j)}=b^{(j)}-\alpha \frac{\partial}{\partial b^{(j)}}J(w,b,x)$
  
- $x_k^{(i)} = x_k^{(i)} - \alpha \frac{\partial}{\partial x_k^{(i)}}J(w,b,x)$

$\}$

### Imports

```
import numpy as np
import tensorflow as tf
import pandas as pd
from numpy import loadtxt
from tensorflow import keras
```

### Utility Functions

```
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

```

### Dataset

```
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
```

### Cost Function

```
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
```

### Add Ratings Yourself

```
movieList, movieList_df = load_Movie_List_pd()

my_ratings = np.zeros(num_movies)          #  Initialize my ratings

# check the file small_movie_list.csv for id of each movie in our dataset
# for example, Toy Story 3 (2010) has ID 2700, 
#   so to rate it "5", you can set 
my_ratings[2700] = 5 

# or suppose you did not enjoy Persuasion (2007), you can set
my_ratings[2609] = 2

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
```

### Prepare Model

```
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
```

### Training

```
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
```

### Recommendations

```
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
```

## Content-based Filtering

Collaborative filtering recommends items based on ratings of users who gave similar ratings as you.

Content-based filtering **recommends** items **based on features of user and item** to find a good match.

### Imports

```
import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
import tabulate
import pickle5 as pickle
import tabulate
import csv
from collections import defaultdict
from numpy import genfromtxt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from recsysNN_utils import *
pd.set_option("display.precision", 1)
```

### Utility Functions

```
def load_data():

    """
    called to load preprepared data
    """

    item_train = genfromtxt('./data/content_item_train.csv', delimiter=',')
    user_train = genfromtxt('./data/content_user_train.csv', delimiter=',')
    y_train    = genfromtxt('./data/content_y_train.csv', delimiter=',')
    with open('./data/content_item_train_header.txt', newline='') as f:    #csv reader handles quoted strings better
        item_features = list(csv.reader(f))[0]
    with open('./data/content_user_train_header.txt', newline='') as f:
        user_features = list(csv.reader(f))[0]
    item_vecs = genfromtxt('./data/content_item_vecs.csv', delimiter=',')

    movie_dict = defaultdict(dict)
    count = 0

    with open('./data/content_movie_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0:
                count += 1  #skip header
                #print(line) print
            else:
                count += 1
                movie_id = int(line[0])
                movie_dict[movie_id]["title"] = line[1]
                movie_dict[movie_id]["genres"] = line[2]

    with open('./data/content_user_to_genre.pickle', 'rb') as f:
        user_to_genre = pickle.load(f)

    return(item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre)


def pprint_train(x_train, features, vs, u_s, maxcount=5, user=True):

    """
    prints user_train or item_train nicely
    """

    if user:
        flist = [".0f", ".0f", ".1f",
                 ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f"]
    else:
        flist = [".0f", ".0f", ".1f", 
                 ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f"]

    head = features[:vs]
    if vs < u_s: print("error, vector start {vs} should be greater then user start {u_s}")
    for i in range(u_s):
        head[i] = "[" + head[i] + "]"
    genres = features[vs:]
    hdr = head + genres
    disp = [split_str(hdr, 5)]
    count = 0
    for i in range(0, x_train.shape[0]):
        if count == maxcount: break
        count += 1
        disp.append([x_train[i, 0].astype(int),
                     x_train[i, 1].astype(int),
                     x_train[i, 2].astype(float),
                     *x_train[i, 3:].astype(float)
                    ])
    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=flist, numalign='center')
    return table


def split_str(ifeatures, smax):

    """
    split the feature name strings to tables fit
    """

    ofeatures = []
    for s in ifeatures:
        if not ' ' in s:  # skip string that already have a space
            if len(s) > smax:
                mid = int(len(s)/2)
                s = s[:mid] + " " + s[mid:]
        ofeatures.append(s)
    return ofeatures
    

def print_pred_movies(y_p, item, movie_dict, maxcount=10):

    """
    print results of prediction of a new user
    inputs are expected to be in sorted order, unscaled
    """

    count = 0
    disp = [["y_p", "movie id", "rating ave", "title", "genres"]]

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        movie_id = item[i, 0].astype(int)
        disp.append([np.around(y_p[i, 0], 1), item[i, 0].astype(int), np.around(item[i, 2].astype(float), 1),
                     movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
    return table


def gen_user_vecs(user_vec, num_items):

    """
    given a user vector return: 
        user predict maxtrix to match the size of item_vecs
    """

    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs


# predict on everything, filter on print/use
def predict_uservec(user_vecs, item_vecs, model, u_s, i_s, scaler):

    """
    given a scaled user vector, does the prediction on all movies in scaled print_item_vecs
    returns 
        an array predictions sorted by predicted rating,
        arrays of user and item, sorted by predicted rating sorting index
    """

    y_p = model.predict([user_vecs[:, u_s:], item_vecs[:, i_s:]])
    y_pu = scaler.inverse_transform(y_p)

    if np.any(y_pu < 0):
        print("error, expected all positive predictions")
    sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()  # negate to get largest rating first
    sorted_ypu   = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]
    sorted_user  = user_vecs[sorted_index]
    return(sorted_index, sorted_ypu, sorted_items, sorted_user)
                
def get_user_vecs(user_id, user_train, item_vecs, user_to_genre):

    """
    given a user_id, return:
        user train/predict matrix to match the size of item_vecs
        y vector with ratings for all rated movies and 0 for others of size item_vecs
    """

    if not user_id in user_to_genre:
        print("error: unknown user id")
        return None
    else:
        user_vec_found = False
        for i in range(len(user_train)):
            if user_train[i, 0] == user_id:
                user_vec = user_train[i]
                user_vec_found = True
                break
        if not user_vec_found:
            print("error in get_user_vecs, did not find uid in user_train")
        num_items = len(item_vecs)
        user_vecs = np.tile(user_vec, (num_items, 1))

        y = np.zeros(num_items)
        for i in range(num_items):  # walk through movies in item_vecs and get the movies, see if user has rated them
            movie_id = item_vecs[i, 0]
            if movie_id in user_to_genre[user_id]['movies']:
                rating = user_to_genre[user_id]['movies'][movie_id]
            else:
                rating = 0
            y[i] = rating
    return(user_vecs, y)

def get_item_genres(item_gvec, genre_features):

    """
    takes in the item's genre vector and list of genre names
    returns the feature names where gvec was 1
    """

    offsets = np.nonzero(item_gvec)[0]
    genres = [genre_features[i] for i in offsets]
    return genres


def print_existing_user(y_p, y, user, items, ivs, uvs, movie_dict, maxcount=10):

    """
    print results of prediction for a user who was in the database
    inputs are expected to be in sorted order, unscaled
    """

    count = 0
    disp = [["y_p", "y", "user", "user genre ave", "movie rating ave", "movie id", "title", "genres"]]
    count = 0
    for i in range(0, y.shape[0]):
        if y[i, 0] != 0:  # zero means not rated
            if count == maxcount:
                break
            count += 1
            movie_id = items[i, 0].astype(int)

            offsets = np.nonzero(items[i, ivs:] == 1)[0]
            genre_ratings = user[i, uvs + offsets]
            disp.append([y_p[i, 0], y[i, 0],
                         user[i, 0].astype(int),      # userid
                         np.array2string(genre_ratings, 
                                         formatter={'float_kind':lambda x: "%.1f" % x},
                                         separator=',', suppress_small=True),
                         items[i, 2].astype(float),    # movie average rating
                         movie_id,
                         movie_dict[movie_id]['title'],
                         movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=[".1f", ".1f", ".0f", ".2f", ".1f"])
    return table

def sq_dist(a,b):

    """
    returns the squared distance between two vectors

    parameters
    ----------
      a : ndarray (n,)
        vector with n features
      b : ndarray (n,)
        vector with n features

    returns
    -------
      d : float
        distance
    """
    
    d = np.linalg.norm(a-b)**2
  
    return d

```

### Data

*Dataset link: https://grouplens.org/datasets/movielens/latest/*

The original dataset has roughly 9000 movies rated by 600 users with ratings on a scale of 0.5 to 5 in 0.5 step increments. The dataset has been reduced in size to focus on movies from the years since 2000 and popular genres. The reduced dataset has  𝑛𝑢=397  users,  𝑛𝑚=847  movies and 25521 ratings. For each movie, the dataset provides a movie title, release date, and one or more genres. For example "Toy Story 3" was released in 2010 and has several genres: "Adventure|Animation|Children|Comedy|Fantasy". This dataset contains little information about users other than their ratings.

```
top10_df = pd.read_csv("./data/content_top10_df.csv")
bygenre_df = pd.read_csv("./data/content_bygenre_df.csv")

print(top10_df)
print(bygenre_df)

# load Data, set configuration variables
item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
num_item_features = item_train.shape[1] - 1  # remove movie id at train time
uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items

print(f"number of training vectors: {len(item_train)}")
```

The training set consists of all the ratings made by the users in the data set. Some ratings are repeated to boost the number of training examples of underrepresented genre's. The training set is split into two arrays with the same number of entries, a user array and a movie/item array.

```
pprint_train(user_train, user_features, uvs,  u_s, maxcount=5)
```

Some of the user and item/movie features are not used in training. In the table above, the features in brackets "[]" such as the "user id", "rating count" and "rating ave" are not included when the model is trained and used. We can see the per genre rating average for user 2. Zero entries are genre's which the user had not rated. The user vector is the same for all the movies rated by a user.

```
pprint_train(item_train, item_features, ivs, i_s, maxcount=5, user=False)
```

Above, the movie array contains the year the film was released, the average rating and an indicator for each potential genre. The indicator is one for each genre that applies to the movie. The movie id is not used in training but is useful when interpreting the data.

```
print(f"y_train[:5]: {y_train[:5]}")
```

The target, y, is the movie rating given by the user.

Above, we can see that movie 6874 is an Action/Crime/Thriller movie released in 2003. User 2 rates action movies as 3.9 on average. MovieLens users gave the movie an average rating of 4. 'y' is 4 indicating user 2 rated movie 6874 as a 4 as well. A single training example consists of a row from both the user and item arrays and a rating from y_train.

#### Prepare Data

We will scale data for better algorithm perfomance.

```
# scale training data
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled    = y_train

scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)

scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train = scalerTarget.transform(y_train.reshape(-1, 1))
#ynorm_test = scalerTarget.transform(y_test.reshape(-1, 1))

# check if scaling is successful
print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))
```

And split data into train/test sets.

```
item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)
print(f"movie/item training data shape: {item_train.shape}")
print(f"movie/item test data shape: {item_test.shape}")
```

The scaled, shuffled data now has a mean of zero. Check results:

```
pprint_train(user_train, user_features, uvs, u_s, maxcount=5)
```

### Cost Function

Cost function for content-based filtering is as follows:

$$J = \sum_{(i, j):r(i,j)=1}(v_u^{(j)} \cdot v_m^{(i)} - y^{(i,j)})^2 + NN \space regularization$$

### Algorithm

To develop content-based filtering, we will use combined prediction of two neural networks: user and movie networks.

![image](https://user-images.githubusercontent.com/73081144/199142303-f58a32da-a0ed-4ad5-b9ba-1100e7a07471.png)

*Fig. 1. Content-based filtering neural networks.*

We can apply sigmoid function to the dot-product to predict the probability that $y^{(i,j)}$ is 1.

![image](https://user-images.githubusercontent.com/73081144/199143453-fa109850-63b4-4e89-9185-d5d45bf458c6.png)

*Fig. 2. Content-based filtering prediction process.*

Notice that both neural networks train at the same time. We're going to judge the two networks according to how well $v_u$ and $v_m$ predict $y^{(i,j)}$, and with this cost function, we're going to use gradient descent or some other optimization algorithm to tune the parameters of the neural network to cause the cost function J to be as small as possible. If you want to regularize this model, we can also add the usual neural network regularization term to encourage the neural networks to keep the values of their parameters small.

### Model

![image](https://user-images.githubusercontent.com/73081144/199153817-1e857eed-11d0-41c5-b857-31bcf5db5bef.png)

*Fig. 3. Tensorflow implementation of content-based filtering model.*

```

num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs, activation='linear')
  
])

item_NN = tf.keras.models.Sequential([
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs, activation='linear') 
    
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

model.summary()
```

### Compiling

```
tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss=cost_fn)
```

### Training

```
tf.random.set_seed(1)
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)
```

### Evaluation

```
model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)
```

### Prediction

#### New User

First, we'll create a new user and have the model suggest movies for that user. After you have tried this on the example user content, feel free to change the user content to match your own preferences and see what the model suggests. Note that ratings are between 0.5 and 5.0, inclusive, in half-step increments.

```
new_user_id = 5000
new_rating_ave = 0.0
new_action = 0.0
new_adventure = 5.0
new_animation = 0.0
new_childrens = 0.0
new_comedy = 0.0
new_crime = 0.0
new_documentary = 0.0
new_drama = 0.0
new_fantasy = 5.0
new_horror = 0.0
new_mystery = 0.0
new_romance = 0.0
new_scifi = 0.0
new_thriller = 0.0
new_rating_count = 3

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])
```

The new user enjoys movies from the adventure, fantasy genres. Let's find the top-rated movies for the new user.
Below, we'll use a set of movie/item vectors, item_vecs that have a vector for each movie in the training/test set. This is matched with the new user vector above and the scaled vectors are used to predict ratings for all the movies.

```
# generate and replicate the user vector to match the number movies in the data set.
user_vecs = gen_user_vecs(user_vec,len(item_vecs))

# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction 
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  # negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  # using unscaled vectors for display

print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount = 10)
```

#### Existing User

Let's look at the predictions for "user 2", one of the users in the data set. We can compare the predicted ratings with the model's ratings.

```
uid = 2 
# form a set of user vectors. This is the same vector, transformed and repeated.
user_vecs, y_vecs = get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre)

# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction 
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  # negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  # using unscaled vectors for display
sorted_user  = user_vecs[sorted_index]
sorted_y     = y_vecs[sorted_index]

# print sorted predictions for movies rated by the user
print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, ivs, uvs, movie_dict, maxcount = 50)
```

The model prediction is generally within 1 of the actual rating though it is not a very accurate predictor of how a user rates specific movies. This is especially true if the user rating is significantly different than the user's genre average.

### Finding Similar Items

The neural network produces two feature vectors, a user feature vector $v_u$, and a movie feature vector, $v_m$. These are 32 entry vectors whose values are difficult to interpret. However, similar items will have similar vectors. This information can be used to make recommendations. For example, if a user has rated "Toy Story 3" highly, one could recommend similar movies by selecting movies with similar movie feature vectors.

A similarity measure is the squared distance between the two vectors $v_m^{(k)}$ and $v_m^{(i)}$:

$$||v_m^{(k)} - v_m^{(i)}||^2 = \sum_{l=1}^{n}(v_{m_l}^{(k)} - v_{m_l}^{(i)})^2$$

A matrix of distances between movies can be computed once when the model is trained and then reused for new recommendations without retraining. The first step, once a model is trained, is to obtain the movie feature vector, $v_m$, for each of the movies. To do this, we will use the trained *item_NN* and build a small model to allow us to run the movie vectors through it to generate $v_m$.

```
input_item_m = tf.keras.layers.Input(shape=(num_item_features))    # input layer
vm_m = item_NN(input_item_m)                                       # use the trained item_NN
vm_m = tf.linalg.l2_normalize(vm_m, axis=1)                        # incorporate normalization as was done in the original model
model_m = tf.keras.Model(input_item_m, vm_m)                                
model_m.summary()
```

Once we have a movie model, we can create a set of movie feature vectors by using the model to predict using a set of item/movie vectors as input. item_vecs is a set of all of the movie vectors. It must be scaled to use with the trained model. The result of the prediction is a 32 entry feature vector for each movie.

```
scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:,i_s:])
print(f"size of all predicted movie feature vectors: {vms.shape}")
```

We can then find the closest movie by finding the minimum along each row. We will make use of numpy masked arrays to avoid selecting the same movie. The masked values along the diagonal won't be included in the computation.

```
count = 50  # number of movies to display
dim = len(vms)
dist = np.zeros((dim,dim))

for i in range(dim):
    for j in range(dim):
        dist[i,j] = sq_dist(vms[i, :], vms[j, :])
        
m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal

disp = [["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i,0])
    movie2_id = int(item_vecs[min_idx,0])
    disp.append( [movie_dict[movie1_id]['title'], movie_dict[movie1_id]['genres'],
                  movie_dict[movie2_id]['title'], movie_dict[movie1_id]['genres']]
               )
```

## Results

This structure is the basis of many commercial recommender systems. The user content can be greatly expanded to incorporate more information about the user if it is available. Items are not limited to movies. This can be used to recommend any item, books, cars or items that are similar to an item in your 'shopping cart'.
