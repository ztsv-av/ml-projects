# Lunar Lander

Lunar lander is a reinforcement learning problem, where the job of the algorithm is to land a spacecraft into a specified zone.

The goal of this reinforcement learning algorithm is to learn a policy $\pi$ that, when given $s$ (chech below), picks action $a=\pi(s)$ so as to maximize the return.

## Actions

- do nothing;
- left thruster;
- main thruster;
- right thruster.

## State

$s$ is going to consist of: $x, y, \dot{x}, \dot{y}, \theta, \dot{\theta}, l, r$, where

- $x =$ position on $x$ axis;
- $y =$ position on $y$ axis;
- $\theta =$ angle / how far lunar lander is tilted to left/right;
- $\dot{\theta} =$ how fast lunar lander is tilting to left/right;
- $l =$ whether left leg of the lunar lander is on the ground (binary value);
- $r =$ whether right leg of the lunar lander is on the ground (binary value).

## Reward Function

- getting to landing pad: 100 - 140;
- additional reward for moving toward/away from pad;
- crash: -100;
- soft landing: +100;
- leg grounded: +10;
- fire main engine: -0.3;
- fire size thruster: -0.03.
