March 2024

## Deep Car
This repository contains an implementation of an enviroment where car can race in a track and reinforcement learning methods are exploited to train an autonomous car. \n
The cars can accelerate, decelerate, turn right or left with turning radius depending by the friction of the road and the velocity of the car.
The track can be arbitrarly built through a series of discrete points that will be interpolated. \\
The reinforcement learning algorithm used is the Deep Deterministic Policy Gradient (DDPG) described in the paper https://arxiv.org/abs/1509.02971.

### Project Structure:
The project is organized as follows:
- **racing_env.py**: defines the car and the track environment for the simulation;
- **RL_library.py**: implements the DDPG agent, including the actor and critic models, experience replay buffer and target network updates;
- **RL_main.py**: contains the main training loop and integrates the environment with the reinforcement learning agent.
