# Temporal Difference

This project implements Temporal Difference (TD) learning algorithms
for reinforcement learning, including Monte Carlo, TD(λ), and SARSA(λ),
applied to the FrozenLake environment from `gymnasium`.

## Tasks

### 0. Monte Carlo

`0-monte_carlo.py` contains the function `monte_carlo(env, V, policy,
episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)` that performs the
first-visit Monte Carlo algorithm to estimate the value function `V` of
a given policy.

* `env` is the environment instance
* `V` is a `numpy.ndarray` of shape `(s,)` containing the value estimate
* `policy` is a function that takes in a state and returns the next
  action to take
* `episodes` is the total number of episodes to train over
* `max_steps` is the maximum number of steps per episode
* `alpha` is the learning rate
* `gamma` is the discount rate

Returns the updated value estimate `V`.

#### Usage

```bash
./0-main.py
```

## Requirements

* Python 3.9
* numpy
* gymnasium
