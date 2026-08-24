# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium

---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

The experiment uses the `FrozenLake-v1` environment from Gymnasium with a 4 × 4 grid.

The environment is represented as:

```text
S F F F
F H F H
F F F H
H F F G
```

Where:

* `S` represents the starting state.
* `F` represents a safe frozen tile.
* `H` represents a hole.
* `G` represents the goal state.

The environment is configured with `is_slippery=True`, which makes the environment stochastic. Therefore, the agent may not always move in the intended direction.

The agent receives a reward of `1` when it reaches the goal and a reward of `0` for other transitions.

---

## Theory

Monte Carlo methods learn from complete episodes. An episode is a sequence of states, actions, and rewards. Unlike Temporal Difference methods, Monte Carlo methods wait until the episode is completed before updating the action-value estimates.

Monte Carlo Control estimates the action-value function $Q(s,a)$ using the returns obtained from complete episodes. The Q-values are gradually improved based on the difference between the observed return and the current Q-value.

The incremental update allows the agent to update its Q-values after every episode while using the complete return obtained from that episode.

---

## Epsilon-Greedy Policy

Monte Carlo Control uses an epsilon-greedy policy to balance exploration and exploitation.

With probability $\epsilon$, the agent selects a random action to explore the environment.

With probability $1-\epsilon$, the agent selects the action having the highest Q-value for the current state.

Initially, epsilon is high so that the agent explores more. As training progresses, epsilon is gradually reduced until it reaches a minimum value. This allows the agent to increasingly exploit the knowledge it has learned.

---

## Algorithm

1. Create the `FrozenLake-v1` environment.
2. Initialize the Q-table with zeros.
3. Set the number of training episodes, discount factor, learning rate, and epsilon parameters.
4. Start with a high epsilon value to encourage exploration.
5. Generate a complete episode using the epsilon-greedy policy.
6. Store the state, action, and reward at every step.
7. Calculate the return by processing the episode backwards.
8. Update the corresponding Q-value using the observed return.
9. Reduce epsilon after each episode while maintaining the minimum epsilon value.
10. Repeat the process for the specified number of episodes.
11. Extract the greedy policy by selecting the action with the highest Q-value for every state.
12. Calculate the estimated state-value function from the maximum Q-value of each state.
13. Display the Q-table, state-value function, learned policy, and average reward.
14. Plot the moving average of rewards to visualize the learning progress.

---

## Python Program

#### Monte Carlo Control

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Environment
# -------------------------------------------------
env_desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

env = gym.make(
    "FrozenLake-v1",
    desc=env_desc,
    is_slippery=True
)

n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------
Q = np.zeros((n_states, n_actions))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------
def epsilon_greedy_action(state, epsilon):

    if np.random.random() < epsilon:
        # Exploration
        return env.action_space.sample()
    else:
        # Exploitation
        return np.argmax(Q[state])


# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------
def generate_episode(epsilon):

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):

        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode


# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------
epsilon = epsilon_start

for episode_num in range(num_episodes):

    # Generate a complete episode
    episode = generate_episode(epsilon)

    # Calculate total reward
    total_reward = sum(
        reward for state, action, reward in episode
    )

    episode_rewards.append(total_reward)

    # Calculate return backward
    G = 0

    for state, action, reward in reversed(episode):

        G = reward + gamma * G

        # Incremental Monte Carlo update
        Q[state, action] += alpha * (
            G - Q[state, action]
        )

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------
optimal_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Display Results
# -------------------------------------------------
def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("Name: Suresh S")
    print("Register Number: 212223040215")

    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):

    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(optimal_policy)

success_rate = np.mean(episode_rewards[-1000:])

print(
    "\nAverage reward over last 1000 episodes:",
    success_rate
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")

plt.grid(True)
plt.show()

env.close()
```

---

## Output

```text
Final Q-table:

[[Q-values for 16 states and 4 actions]]

Estimated State-Value Function:

[[V-values arranged in a 4 × 4 grid]]

Learned Policy:

[['L' 'D/R/U' ...]
 [... ... ... ...]
 [... ... ... ...]
 [... ... ... ...]]

Average reward over last 1000 episodes: <value>
```

The exact Q-table, state-value function, policy, and average reward can vary between executions because the FrozenLake environment is stochastic and the Monte Carlo algorithm uses random exploration.

---

## Result

```text
The On-Policy Monte Carlo Control algorithm was successfully
implemented using the Gymnasium FrozenLake-v1 environment.
The agent generated complete episodes and updated the Q-table
using the returns obtained from those episodes. An epsilon-
greedy policy was used to balance exploration and exploitation.
The learned Q-values were used to obtain the final greedy policy
and estimated state-value function.
```

---

## Inference

```text
Monte Carlo Control learns the action values by observing
complete episodes and using the returns obtained from them.
The epsilon-greedy strategy allows the agent to explore different
actions initially and gradually exploit the actions with higher
Q-values. As training progresses, the learned policy improves
and the average reward generally increases. The experiment
demonstrates how an agent can learn an effective policy directly
from experience without requiring an explicit model of the
environment.
```
