March 2024

## Deep Car
This repository contains an implementation of an environment where cars can race on a track, and reinforcement learning methods are used to train an autonomous car. The cars can accelerate, decelerate, and turn right or left with a turning radius that depends on the friction of the road and the velocity of the car. The track can be arbitrarily constructed through a series of discrete points that are interpolated to form a continuous path. The reinforcement learning algorithm used is the Deep Deterministic Policy Gradient (DDPG), as described in the paper https://arxiv.org/abs/1509.02971.

### Project Structure:
The project is organized as follows:
- **racing_env.py**: defines the car and the track environment for the simulation;
- **RL_library.py**: implements the DDPG agent, including the actor and critic models, experience replay buffer and target network updates;
- **RL_main.py**: contains the main training loop and integrates the environment with the reinforcement learning agent.
