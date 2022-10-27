## Decision Tree, Bagged Forest, Random Forest, XGBoost

Decision Tree (DT) is a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation. As the name goes, it uses a tree-like model of decisions.

When building a decision tree, the way we'll decide what feature to split on at a node will be based on what choice of feature reduces entropy the most. Reduces entropy or reduces impurity, or maximizes purity. In decision tree learning, the reduction of entropy is called **information gain**.

**Entropy** formula:

$$H(p_1) = -p_1log_2(p_1) - (1 - p_1)log_2(1 - p_1)$$

*Note: in entropy, $0log(0) = 0$*

**Information gain** formula:

$$IG = H(p_1^{root}) - (w^{left}H(p_1^{left}) + w^{right}H(p_1^{right}))$$

## Single Decision Tree

Steps for building a decision tree are as follows:
- start with all examples at the root node;
- calculate information gain for splitting on all possible features, and pick the one with the highest information gain;
- split dataset according to the selected feature, and create left and right branches of the tree;
- keep repeating splitting process until stopping criteria is met.
  
  
We will need following functions, which will let us split a node into left and right branches using the feature with the highest information gain
- calculate the entropy at a node; 
- split the dataset at a node into left and right branches based on a given feature;
- calculate the information gain from splitting on a given feature;
- choose the feature that maximizes information gain.

### Imports

```
import numpy as np
import matplotlib.pyplot as plt
```

### Dataset

```
X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))
```

### Entropy

```
def compute_entropy(y):

    """
    computes the entropy at node
    
    parameters
    ----------
        y : ndarray
            numpy array indicating whether each example at a node is
                edible (`1`) or poisonous (`0`)
       
    returns
    -------
        entropy : float
            entropy at that node
        
    """
    
    if len(y) == 0:
        return 0
    
    entropy = 0.
    
    p1 = sum(y)/len(y)
    
    if p1 == 1.0 or p1 ==0 :
        return 0
    
    entropy = -p1*np.log2(p1) - (1 - p1)*np.log2(1 - p1) 

    return entropy
```

### Split Dataset

```
def split_dataset(X, node_indices, feature):

    """
    splits the data at the given node into
        left and right branches
    
    parameters
    ----------
        X : ndarray
            data matrix of shape (n_samples, n_features)
        node_indices : list
            list containing the active indices.
            i.e, the samples being considered at this step.
        feature : int
            index of feature to split on
    
    returns
    -------
        left_indices : list
            indices with feature value == 1
        right_indices : list
            indices with feature value == 0
    """
    
    left_indices = []
    right_indices = []
    
    for i in node_indices:
        
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
        
    return left_indices, right_indices
```

### Information Gain

```
def compute_information_gain(X, y, node_indices, feature):
    
    """
    compute the information of splitting the node on a given feature
    
    parameters
    ----------
        X : ndarray
            data matrix of shape (n_samples, n_features)
        y : array like
            list or ndarray with n_samples containing the target variable
        node_indices : ndarray
            list containing the active indices. 
            i.e, the samples being considered in this step.
        feature : int
            feature id to split on
   
    returns
    -------
        cost : float
            cost computed
    """    
    # split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    information_gain = 0
    
    hp1_node = compute_entropy(y_node)
    hp1_left = compute_entropy(y_left)
    hp1_right = compute_entropy(y_right)
    
    w_left = len(X_left)/len(X_node)
    w_right = len(X_right)/len(X_node)
    
    information_gain = hp1_node - (w_left*hp1_left + w_right*hp1_right)
    
    return information_gain
```

### Best Split

Let's write a function to get the best feature to split on by computing the information gain from each feature as we did above and returning the feature that gives the maximum information gain

```
def get_best_split(X, y, node_indices):   

    """
    returns the optimal feature and threshold value
        to split the node data 
    
    parameters
    ----------
        X : ndarray
            data matrix of shape (n_samples, n_features)
        y : array like
            list or ndarray with n_samples containing the target variable
        node_indices : ndarray
            list containing the active indices. 
            i.e, the samples being considered in this step.

    returns
    -------
        best_feature : int
            the index of the best feature to split
    """    
    
    # some useful variables
    num_features = X.shape[1]

    best_feature = -1
    best_info_gain = 0
    
    for feature in range(num_features):
        
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature  
   
    return best_feature
```

### Building a Decision Tree

```
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):

    """
    build a tree using the recursive algorithm that split the dataset 
        into 2 subgroups at each node.
    this function just prints the tree.
    
    parameters
    ----------
        X : ndarray
            data matrix of shape (n_samples, n_features)
        y : array like
            list or ndarray with n_samples containing the target variable
        node_indices : ndarray
            list containing the active indices. 
            i.e, the samples being considered in this step.
        branch_name : string
            name of the branch. 
            ['Root', 'Left', 'Right']
        max_depth : int
            max depth of the resulting tree. 
        current_depth : int
            current depth. 
            parameter used during recursive call.
   
    """ 

    # maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # otherwise, get best split and split the data
    # get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    # continue splitting the left and the right child. 
    # increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)

build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
```
