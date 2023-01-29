# Lunar Lander

Lunar lander is a reinforcement learning problem, where the job of the algorithm is to land a spacecraft into a specified zone. The landing pad is always at coordinates  (0,0) 

The goal of this reinforcement learning algorithm is to learn a policy $\pi$ that, when given $s$ (chech below), picks action $a=\pi(s)$ so as to maximize the return.

## Actions

- do nothing = 0;
- fire right thruster = 1;
- fire main thruster = 2;
- fire left thruster = 3.

## State

$s$ is going to consist of: $x, y, \dot{x}, \dot{y}, \theta, \dot{\theta}, l, r$, where

- $x =$ position on $x$ axis;
- $y =$ position on $y$ axis;
- $\dot{x} =$ $x$ coordinate velocity;
- $\dot{y} =$ $y$ coordinate velocity;
- $\theta =$ angle / how far lunar lander is tilted to left/right;
- $\dot{\theta} =$ how fast lunar lander is tilting to left/right; angular velocity;
- $l =$ whether left leg of the lunar lander is on the ground (binary value);
- $r =$ whether right leg of the lunar lander is on the ground (binary value).

## Reward Function

- getting to landing pad: 100 - 140;
- additional reward for moving toward/away from pad;
- crash: -100;
- soft landing: +100;
- leg grounded: +10;
- fire main engine: -0.3 each frame;
- fire size thruster: -0.03 each frame.

## Imports

```
import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import logging

from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

logging.getLogger().setLevel(logging.ERROR) # suppress warnings from imageio

# set up a virtual display to render the Lunar Lander environment
Display(visible=0, size=(840, 480)).start();

# set the random seed for TensorFlow
tf.random.set_seed(SEED)
```

## Utility Functions

```
def check_update_conditions(t, num_steps_upd, memory_buffer):

    """
    determines if the conditions are met to perform a learning update

    checks if the current time step t is a multiple of num_steps_upd 
    and if the memory_buffer has enough experience tuples to fill a mini-batch 
        (for example, if the mini-batch size is 64, 
        then the memory buffer should have more than 64 experience
        tuples in order to perform a learning update).
    
    parameters
    ----------
    t (int)
        the current time step
    num_steps_upd (int)
        the number of time steps used to determine how often to perform a learning update 
        a learning update is only performed every num_steps_upd time steps
     memory_buffer (deque)
        a deque containing experiences
        the experiences are stored in the memory buffer as namedtuples: 
        namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    returns
    -------
       a boolean that will be True if conditions are met and False otherwise 
    """

    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False


def get_new_eps(epsilon):

    """
    updates the epsilon value for the ε-greedy policy
    
    gradually decreases the value of epsilon towards a minimum value (E_MIN) using the
    given ε-decay rate (E_DECAY).

    parameters
    ----------
    epsilon (float)
        the current value of epsilon

    returns
    -------
    a float with the updated value of epsilon
    """

    return max(E_MIN, E_DECAY * epsilon)


def get_experiences(memory_buffer):

    """
    returns a random sample of experience tuples drawn from the memory buffer

    retrieves a random sample of experience tuples from the given memory_buffer and
    returns them as TensorFlow Tensors
    the size of the random sample is determined by the mini-batch size (MINIBATCH_SIZE) 
    
    parameters
    ----------
    memory_buffer (deque)
        a deque containing experiences
        the experiences are stored in the memory buffer as namedtuples:
        namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    returns
    -------
    a tuple (states, actions, rewards, next_states, done_vals) where:

        - states are the starting states of the agent.
        - actions are the actions taken by the agent from the starting states.
        - rewards are the rewards received by the agent after taking the actions.
        - next_states are the new states of the agent after taking the actions.
        - done_vals are the boolean values indicating if the episode ended.

        all tuple elements are TensorFlow Tensors whose shape is determined by the
        mini-batch size and the given Gym environment
        for the Lunar Lander environment the states and next_states will have a shape of [MINIBATCH_SIZE, 8] 
        while the actions, rewards, and done_vals will have a shape of [MINIBATCH_SIZE]
        all TensorFlow Tensors have elements with dtype=tf.float32.
    """

    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=tf.float32,
    )
    return (states, actions, rewards, next_states, done_vals)


def get_action(q_values, epsilon=0.0):

    """
    returns an action using an ε-greedy policy

    this function will return an action according to the following rules:
        - with probability epsilon, it will return an action chosen at random.
        - with probability (1 - epsilon), it will return the action that yields the
        maximum Q value in q_values
    
    parameters
    ----------
    q_values (tf.Tensor)
        the Q values returned by the Q-Network
        for the Lunar Lander environment
            this TensorFlow Tensor should have a shape of [1, 4] and its elements should
            have dtype=tf.float32
    epsilon (float)
        the current value of epsilon

    returns
    -------
    an action (numpy.int64)
        for the Lunar Lander environment, actions are
            represented by integers in the closed interval [0,3]
    """

    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(4))


def update_target_network(q_network, target_q_network):

    """
    updates the weights of the target Q-Network using a soft update
    the weights of the target_q_network are updated using the soft update rule:
    
                    w_target = (TAU * w) + (1 - TAU) * w_target
    
    where w_target are the weights of the target_q_network, TAU is the soft update
    parameter, and w are the weights of the q_network
    
    parameters
    ----------
    q_network (tf.keras.Sequential) 
        the Q-Network
    target_q_network (tf.keras.Sequential)
        the Target Q-Network
    """

    for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
            target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)


def plot_history(point_history, **kwargs):

    """
    plots the total number of points received by the agent after each episode together
    with the moving average (rolling mean)

    parameters
    ----------
    point_history (list)
        a list containing the total number of points the agent received after each episode
    **kwargs: optional
        window_size (int)
            size of the window used to calculate the moving average (rolling mean)
            this integer determines the fixed number of data points used for each window
            the default window size is set to 10% of the total number of data points in point_history, 
                i.e. if point_history has 200 data points
                the default window size will be 20
        lower_limit (int):
            the lower limit of the x-axis in data coordinates
            default value is 0
        upper_limit (int):
            the upper limit of the x-axis in data coordinates
            default value is len(point_history)
        plot_rolling_mean_only (bool):
            if True, only plots the moving average (rolling mean) without the point history
            default value is False
        plot_data_only (bool):
            if True, only plots the point history without the moving average
            default value is False
    """

    lower_limit = 0
    upper_limit = len(point_history)

    window_size = (upper_limit * 10) // 100

    plot_rolling_mean_only = False
    plot_data_only = False

    if kwargs:
        if "window_size" in kwargs:
            window_size = kwargs["window_size"]

        if "lower_limit" in kwargs:
            lower_limit = kwargs["lower_limit"]

        if "upper_limit" in kwargs:
            upper_limit = kwargs["upper_limit"]

        if "plot_rolling_mean_only" in kwargs:
            plot_rolling_mean_only = kwargs["plot_rolling_mean_only"]

        if "plot_data_only" in kwargs:
            plot_data_only = kwargs["plot_data_only"]

    points = point_history[lower_limit:upper_limit]

    # generate x-axis for plotting
    episode_num = [x for x in range(lower_limit, upper_limit)]

    # use Pandas to calculate the rolling mean (moving average)
    rolling_mean = pd.DataFrame(points).rolling(window_size).mean()

    plt.figure(figsize=(10, 7), facecolor="white")

    if plot_data_only:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
    elif plot_rolling_mean_only:
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")
    else:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")

    text_color = "black"

    ax = plt.gca()
    ax.set_facecolor("black")
    plt.grid()
    plt.xlabel("episode", color=text_color, fontsize=30)
    plt.ylabel("total points", color=text_color, fontsize=30)
    yNumFmt = mticker.StrMethodFormatter("{x:,}")
    ax.yaxis.set_major_formatter(yNumFmt)
    ax.tick_params(axis="x", colors=text_color)
    ax.tick_params(axis="y", colors=text_color)
    plt.show()


def display_table(initial_state, action, next_state, reward, done):

    """
    displays a table containing the initial state, action, next state, reward, and done
    values from Gym's Lunar Lander environment

    all floating point numbers in the table are displayed rounded to 3 decimal places
    and actions are displayed using their labels instead of their numerical value (i.e
    if action = 0, the action will be printed as "Do nothing" instead of "0").

    parameters
    ----------
    initial_state (numpy.ndarray)
        the initial state vector returned when resetting the Lunar Lander environment, 
            i.e the value returned by the env.reset() method
    action (int)
        the action taken by the agent
        in the Lunar Lander environment, actions are represented by integers 
            in the closed interval [0,3] corresponding to:
                - Do nothing = 0
                - Fire right engine = 1
                - Fire main engine = 2
                - Fire left engine = 3
    next_state (numpy.ndarray)
        the state vector returned by the Lunar Lander environment after the agent
            takes an action, i.e the observation returned after running a single time
            step of the environment's dynamics using env.step(action)
    reward (numpy.float64)
        the reward returned by the Lunar Lander environment after the agent takes an
            action, i.e the reward returned after running a single time step of the
            environment's dynamics using env.step(action)
    done (bool)
        the done value returned by the Lunar Lander environment after the agent
            takes an action, i.e the done value returned after running a single time
            step of the environment's dynamics using env.step(action)
    
    returns
    ----------
    table (statsmodels.iolib.table.SimpleTable)
        a table object containing the initial_state, action, next_state, reward, and done values
        this will result in the table being displayed in the Jupyter Notebook.
    """

    action_labels = [
        "Do nothing",
        "Fire right engine",
        "Fire main engine",
        "Fire left engine",
    ]

    # do not use column headers
    column_headers = None

    # display all floating point numbers rounded to 3 decimal places
    with np.printoptions(formatter={"float": "{:.3f}".format}):
        table_info = [
            ("Initial State:", [f"{initial_state}"]),
            ("Action:", [f"{action_labels[action]}"]),
            ("Next State:", [f"{next_state}"]),
            ("Reward Received:", [f"{reward:.3f}"]),
            ("Episode Terminated:", [f"{done}"]),
        ]

    # generate table
    row_labels, data = zip_longest(*table_info)
    table = SimpleTable(data, column_headers, row_labels)

    return table


def embed_mp4(filename):

    """
    embeds an MP4 video file in a Jupyter notebook
    
    parameters
    ----------
    filename (string)
        the path to the the MP4 video file that will be embedded 
            (i.e. "./videos/lunar_lander.mp4")
    
    returns
    -------
    returns a display object from the given video file
    this will result in the video being displayed in the Jupyter Notebook
    """

    video = open(filename, "rb").read()
    b64 = base64.b64encode(video)
    tag = """
    <video width="840" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>""".format(
        b64.decode()
    )

    return IPython.display.HTML(tag)


def create_video(filename, env, q_network, fps=30):

    """
    creates a video of an agent interacting with a Gym environment

    the agent will interact with the given env environment using the q_network to map
    states to Q values and using a greedy policy to choose its actions (i.e it will
    choose the actions that yield the maximum Q values)
    
    the video will be saved to a file with the given filename. The video format must be
    specified in the filename by providing a file extension (.mp4, .gif, etc..). If you 
    want to embed the video in a Jupyter notebook using the embed_mp4 function, then the
    video must be saved as an MP4 file
    
    parameters
    ----------
    filename (string)
        the path to the file to which the video will be saved
        the video format will be selected based on the filename
        therefore, the video format must be specified in the filename by providing a file extension 
            (i.e. "./videos/lunar_lander.mp4")
        to see a list of supported formats see the imageio documentation: 
            https://imageio.readthedocs.io/en/v2.8.0/formats.html
    env (Gym Environment) 
        the Gym environment the agent will interact with
    q_network (tf.keras.Sequential)
        a TensorFlow Keras Sequential model that maps states to Q values
    fps (int)
        the number of frames per second 
        specifies the frame rate of the output video
        the default frame rate is 30 frames per second  
    """

    with imageio.get_writer(filename, fps=fps) as video:
        done = False
        state = env.reset()
        frame = env.render(mode="rgb_array")
        video.append_data(frame)
        while not done:
            state = np.expand_dims(state, axis=0)
            q_values = q_network(state)
            action = np.argmax(q_values.numpy()[0])
            state, _, done, _ = env.step(action)
            frame = env.render(mode="rgb_array")
            video.append_data(frame)
```

## Hyperparameters

```
MEMORY_SIZE = 100_000     # size of memory buffer.
GAMMA = 0.995             # discount factor.
ALPHA = 1e-3              # learning rate.
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps.
SEED = 0  # Seed for the pseudo-random number generator.
MINIBATCH_SIZE = 64  # mini-batch size
TAU = 1e-3  # soft update parameter
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy
E_MIN = 0.01  # minimum ε value for the ε-greedy policy
```

## Environment

### Load Environment

We start by loading the *LunarLander-v2* environment from the gym library by using the *.make()* method.

```
env = gym.make('LunarLander-v2')
```

Once we load the environment we use the .reset() method to reset the environment to the initial state. The lander starts at the top center of the environment and we can render the first frame of the environment by using the .render() method

```
env.reset()
PIL.Image.fromarray(env.render(mode='rgb_array'))
```

In order to build our neural network later on we need to know the size of the state vector and the number of valid actions. We can get this information from our environment by using the .observation_space.shape and action_space.n methods, respectively.

```
state_size = env.observation_space.shape
num_actions = env.action_space.n

print('State Shape:', state_size)
print('Number of actions:', num_actions)
```

### Interacting with Environment

The Gym library implements the standard “agent-environment loop” formalism. In the standard “agent-environment loop” formalism, an agent interacts with the environment in discrete time steps  $𝑡=0,1,2,...$. At each time step  $𝑡$, the agent uses a policy $𝜋$ to select an action $𝐴_𝑡$ based on its observation of the environment's state $𝑆_𝑡$. The agent receives a numerical reward $𝑅_𝑡$ and on the next time step, moves to a new state $𝑆_{𝑡+1}$.

### Exploring the Environment's Dynamics

In Open AI's Gym environments, we use the .step() method to run a single time step of the environment's dynamics. In the version of gym that we are using the .step() method accepts an action and returns four values:

- observation (object): an environment-specific object representing your observation of the environment. In the Lunar Lander environment this corresponds to a numpy array containing the positions and velocities of the lander.
- reward (float): amount of reward returned as a result of taking the given action. In the Lunar Lander environment this corresponds to a float of type numpy.float64.
- done (boolean): When done is True, it indicates the episode has terminated and it’s time to reset the environment.
- info (dictionary): diagnostic information useful for debugging.

To begin an episode, we need to reset the environment to an initial state. We do this by using the .reset() method.

```
# reset the environment and get the initial state
initial_state = env.reset()
```

Checking how returned values change depending on the action taken:

```
# select an action
action = 0

# run a single time step of the environment's dynamics with the given action
next_state, reward, done, _ = env.step(action)

# display table with values
# all values are displayed to 3 decimal places
display_table(initial_state, action, next_state, reward, done)
```

In practice, when we train the agent we use a loop to allow the agent to take many consecutive actions during an episode.

## Deep Q-Learning

In cases where both the state and action space are discrete we can estimate the action-value function iteratively by using the Bellman equation:

$$
Q_{i+1}(s,a) = R + \gamma \max_{a'}Q_i(s',a')
$$

This iterative method converges to the optimal action-value function $Q^*(s,a)$ as $i\to\infty$. This means that the agent just needs to gradually explore the state-action space and keep updating the estimate of $Q(s,a)$ until it converges to the optimal action-value function $Q^*(s,a)$. However, in cases where the state space is continuous it becomes practically impossible to explore the entire state-action space. Consequently, this also makes it practically impossible to gradually estimate $Q(s,a)$ until it converges to $Q^*(s,a)$.

In the Deep $Q$-Learning, we solve this problem by using a neural network to estimate the action-value function $Q(s,a)\approx Q^*(s,a)$. We call this neural network a $Q$-Network and it can be trained by adjusting its weights at each iteration to minimize the mean-squared error in the Bellman equation.

Unfortunately, using neural networks in reinforcement learning to estimate action-value functions has proven to be highly unstable. Luckily, there's a couple of techniques that can be employed to avoid instabilities. These techniques consist of using a ***Target Network*** and ***Experience Replay***. We will explore these two techniques in the following sections.

### Target Network

We can train the $Q$-Network by adjusting it's weights at each iteration to minimize the mean-squared error in the Bellman equation, where the target values are given by:

$$
y = R + \gamma \max_{a'}Q(s',a';w)
$$

where $w$ are the weights of the $Q$-Network. This means that we are adjusting the weights $w$ at each iteration to minimize the following error:

$$
\overbrace{\underbrace{R + \gamma \max_{a'}Q(s',a'; w)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}
$$

Notice that this forms a problem because the $y$ target is changing on every iteration. Having a constantly moving target can lead to oscillations and instabilities. To avoid this, we can create
a separate neural network for generating the $y$ targets. We call this separate neural network the **target $\hat Q$-Network** and it will have the same architecture as the original $Q$-Network. By using the target $\hat Q$-Network, the above error becomes:

$$
\overbrace{\underbrace{R + \gamma \max_{a'}\hat{Q}(s',a'; w^-)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}
$$

where $w^-$ and $w$ are the weights the target $\hat Q$-Network and $Q$-Network, respectively.

In practice, we will use the following algorithm: every $C$ time steps we will use the $\hat Q$-Network to generate the $y$ targets and update the weights of the target $\hat Q$-Network using the weights of the $Q$-Network. We will update the weights $w^-$ of the the target $\hat Q$-Network using a **soft update**. This means that we will update the weights $w^-$ using the following rule:
 
$$
w^-\leftarrow \tau w + (1 - \tau) w^-
$$

where $\tau\ll 1$. By using the soft update, we are ensuring that the target values, $y$, change slowly, which greatly improves the stability of our learning algorithm.

### Initializing Q-Networks

```
# Create the Q-Network.
q_network = Sequential([
    Input(state_size),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_actions, activation='linear')
    ])

# Create the target Q^-Network.
target_q_network = Sequential([
    Input(state_size),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_actions, activation='linear')
    ])

optimizer = Adam(learning_rate=ALPHA)
```

### Experience Replay

When an agent interacts with the environment, the states, actions, and rewards the agent experiences are sequential by nature. If the agent tries to learn from these consecutive experiences it can run into problems due to the strong correlations between them. To avoid this, we employ a technique known as **Experience Replay** to generate uncorrelated experiences for training our agent. Experience replay consists of storing the agent's experiences (i.e the states, actions, and rewards the agent receives) in a memory buffer and then sampling a random mini-batch of experiences from the buffer to do the learning. The experience tuples $(S_t, A_t, R_t, S_{t+1})$ will be added to the memory buffer at each time step as the agent interacts with the environment.

By using experience replay we avoid problematic correlations, oscillations and instabilities. In addition, experience replay also allows the agent to potentially use the same experience in multiple weight updates, which increases data efficiency.

```
# store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
```

## Deep Q-Learning with Experience Replay

![image](https://user-images.githubusercontent.com/73081144/200744524-c5d8d143-7308-4def-908a-cb5980d973e9.png)

## Loss

```
def compute_loss(experiences, gamma, q_network, target_q_network):

    """ 
    calculates the loss
    
    parameters
    ----------
    experiences (tuple)
        tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
    gamma (float)
        the discount factor.
    q_network (tf.keras.Sequential)
        keras model for predicting the q_values
    target_q_network (tf.keras.Sequential)
        keras model for predicting the targets
          
    returns
    -------
    loss (TensorFlow Tensor(shape=(0,), dtype=int32))
        the Mean-Squared Error between
            the y targets and the Q(s,a) values
    """
    
    # unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    
    # compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    # set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a)
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    
    # get the q_values
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
        
    # compute the loss
    loss = MSE(q_values, y_targets)
    
    return loss
```

## Updating Weights

```
@tf.function
def agent_learn(experiences, gamma):

    """
    updates the weights of the Q networks
    
    parameters
    ----------
    experiences (tuple)
        tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
    gamma (float)
        the discount factor
    """
    
    # calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # get the gradients of the loss with respect to the weights
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    # update the weights of the q_network
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    update_target_network(q_network, target_q_network)
```

## Training Agent

* **Line 1**: We initialize the `memory_buffer` with a capacity of $N =$ `MEMORY_SIZE`. Notice that we are using a `deque` as the data structure for our `memory_buffer`.


* **Line 2**: Initialize Q-Network.


* **Line 3**: Initialize Target Q-Network.


* **Line 4**: We start the outer loop. Notice that we have set $M =$ `num_episodes = 2000`. This number is reasonable because the agent should be able to solve the Lunar Lander environment in less than `2000` episodes using this notebook's default parameters.


* **Line 5**: We use the `.reset()` method to reset the environment to the initial state and get the initial state.


* **Line 6**: We start the inner loop. Notice that we have set $T =$ `max_num_timesteps = 1000`. This means that the episode will automatically terminate if the episode hasn't terminated after `1000` time steps.


* **Line 7**: The agent observes the current `state` and chooses an `action` using an $\epsilon$-greedy policy. Our agent starts out using a value of $\epsilon =$ `epsilon = 1` which yields an $\epsilon$-greedy policy that is equivalent to the equiprobable random policy. This means that at the beginning of our training, the agent is just going to take random actions regardless of the observed `state`. As training progresses we will decrease the value of $\epsilon$ slowly towards a minimum value using a given $\epsilon$-decay rate. We want this minimum value to be close to zero because a value of $\epsilon = 0$ will yield an $\epsilon$-greedy policy that is equivalent to the greedy policy. This means that towards the end of training, the agent will lean towards selecting the `action` that it believes (based on its past experiences) will maximize $Q(s,a)$. We will set the minimum $\epsilon$ value to be `0.01` and not exactly 0 because we always want to keep a little bit of exploration during training.


* **Line 8**: We use the `.step()` method to take the given `action` in the environment and get the `reward` and the `next_state`. 


* **Line 9**: We store the `experience(state, action, reward, next_state, done)` tuple in our `memory_buffer`. Notice that we also store the `done` variable so that we can keep track of when an episode terminates. This allowed us to set the $y$ targets in [Exercise 2](#ex02).


* **Line 10**: We check if the conditions are met to perform a learning update. This function checks if $C =$ `NUM_STEPS_FOR_UPDATE = 4` time steps have occured and if our `memory_buffer` has enough experience tuples to fill a mini-batch. For example, if the mini-batch size is `64`, then our `memory_buffer` should have more than `64` experience tuples in order to pass the latter condition. If the conditions are met, then thefunction will return a value of `True`, otherwise it will return a value of `False`.


* **Lines 11 - 14**: If the `update` variable is `True` then we perform a learning update. The learning update consists of sampling a random mini-batch of experience tuples from our `memory_buffer`, setting the $y$ targets, performing gradient descent, and updating the weights of the networks. We will use the `agent_learn` function we defined in [Section 8](#8) to perform the latter 3.


* **Line 15**: At the end of each iteration of the inner loop we set `next_state` as our new `state` so that the loop can start again from this new state. In addition, we check if the episode has reached a terminal state (i.e we check if `done = True`). If a terminal state has been reached, then we break out of the inner loop.


* **Line 16**: At the end of each iteration of the outer loop we update the value of $\epsilon$, and check if the environment has been solved. We consider that the environment has been solved if the agent receives an average of `200` points in the last `100` episodes. If the environment has not been solved we continue the outer loop and start a new episode.

Finally, we include some extra variables to keep track of the total number of points the agent received in each episode. This will help us determine if the agent has solved the environment and it will also allow us to see how our agent performed during training. We also use the `time` module to measure how long the training takes. 

```
start = time.time()

num_episodes = 2000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy

# create a memory buffer D with capacity N
memory_buffer = deque(maxlen=MEMORY_SIZE)

# set the target network weights equal to the Q-Network weights
target_q_network.set_weights(q_network.get_weights())

for i in range(num_episodes):
    
    # reset the environment to the initial state and get the initial state
    state = env.reset()
    total_points = 0
    
    for t in range(max_num_timesteps):
        
        # from the current state S choose an action A using an ε-greedy policy
        state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
        q_values = q_network(state_qn)
        action = get_action(q_values, epsilon)
        
        # take action A and receive reward R and the next state S'
        next_state, reward, done, _ = env.step(action)
        
        # store experience tuple (S,A,R,S') in the memory buffer
        # we store the done variable as well for convenience
        memory_buffer.append(experience(state, action, reward, next_state, done))
        
        # only update the network every NUM_STEPS_FOR_UPDATE time steps
        update = check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
        
        if update:
            # smple random mini-batch of experience tuples (S,A,R,S') from D
            experiences = get_experiences(memory_buffer)
            
            # set the y targets, perform a gradient descent step,
            # and update the network weights
            agent_learn(experiences, GAMMA)
        
        state = next_state.copy()
        total_points += reward
        
        if done:
            break
            
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    
    # update the ε value
    epsilon = get_new_eps(epsilon)

    print(f"\repisode {i+1} | total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\repisode {i+1} | total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # we will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes
    if av_latest_points >= 200.0:
        print(f"\n\nenvironment solved in {i+1} episodes!")
        q_network.save('lunar_lander_model.h5')
        break
        
tot_time = time.time() - start

print(f"\ntotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
```

## Plot Points History

```
# plot the total point history along with the moving average
plot_history(total_point_history)
```

## See Agent in Action

In the cell below we create a video of our agent interacting with the Lunar Lander environment using the trained *q_network*. The video is saved to the videos folder with the given filename. We use the *embed_mp4* function to embed the video in the Jupyter Notebook so that we can see it here directly without having to download it.

Note that since the lunar lander starts with a random initial force applied to its center of mass, every time you run the cell below you will see a different video. If the agent was trained properly, it should be able to land the lunar lander in the landing pad every time, regardless of the initial force applied to its center of mass.

```
filename = "./videos/lunar_lander.mp4"

utils.create_video(filename, env, q_network)
utils.embed_mp4(filename)
```
