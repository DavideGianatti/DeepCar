import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from RL_library import *
from racing_env import *

"""
This script trains an autonomous car to navigate an racing track using a reinforcement learning algorithm.
"""

# Set random seed for reproducibility
seed_value = 1234
keras.utils.set_random_seed(seed_value)
np.random.seed(seed_value)

# Configuration parameters
total_episodes = 50  # Number of episodes for training
n_t = 1000           # Number of time steps per episode
gamma = 0.99         # Discount factor for future rewards
tau = 0.005          # Soft update factor for target networks

# Initialize lists to store reward history
ep_reward_list = []   # To store reward history of each episode
avg_reward_list = []  # To store average reward history of recent episodes

# Define the racing track (this example is an oval track)
track = Track([[0,0], [0, 200], [5, 225], [15, 250],[30, 275], [50, 275], [70, 250], [85, 225], [95, 200], [95, -100],
               [90, -125], [80, -150], [65, -175], [45, -175], [25, -150], [15, -125], [5, -100], [0, -75]], 10)

# Define angles for observation
n_angles = 64
angles = np.linspace(-np.pi / 2, np.pi / 2, n_angles + 1)

# Initialize the actor and critic models
actor_model = get_actor(n_angles=len(angles))
critic_model = get_critic(n_angles=len(angles), n_actions=2)
target_actor = get_actor(n_angles=len(angles))
target_critic = get_critic(n_angles=len(angles), n_actions=2)
best_actor = get_actor(n_angles=len(angles))
best_critic = get_critic(n_angles=len(angles), n_actions=2)

# Initialize target networks with weights of the current models
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Set learning rates and weight decay for actor and critic optimizers
critic_lr = 0.01
critic_wd = 0.0001
actor_lr = 0.0001
actor_wd = 0.0001

# Initialize optimizers for actor and critic models
critic_optimizer = tf.keras.optimizers.Adam(critic_lr, weight_decay=critic_wd)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr, weight_decay=actor_wd)

# Initialize experience replay buffer
buffer = Buffer(buffer_capacity=50000, batch_size=1024, target_actor=target_actor, target_critic=target_critic,
                critic_model=critic_model, actor_model=actor_model, critic_optimizer=critic_optimizer,
                actor_optimizer=actor_optimizer, gamma=gamma, num_states=len(angles), num_actions=2)

best_reward = -10**6  # Initialize best reward to a very low value

# Main training loop
for ep in tqdm(range(total_episodes)):

    # Initialize car and state
    car = Car(pos=[0., 0.], v=[0., 10.], track=track)
    prev_state = track(car, angles)
    prev_v = car.v
    prev_dir = track.dir[car.progress]

    episodic_reward = 0
    episodic_v = 0
    episodic_crash = 0

    # Time steps within an episode
    for r in range(n_t):

        # Prepare tensors for the model
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        tf_prev_v = tf.expand_dims(tf.convert_to_tensor(prev_v), 0)
        tf_prev_dir = tf.expand_dims(tf.convert_to_tensor(prev_dir), 0)

        # Get action from the actor model
        action = tf.squeeze(actor_model([tf_prev_state, tf_prev_v, tf_prev_dir]))
        np_action = np.array(action)  # Convert to numpy array

        try:
            # Execute action and observe new state
            reward = car.move(left_right=np_action[0], gas_break=np_action[1], track=track)
            state = track(car, angles)
        except ValueError as err:
            print(f'Warning: {err}')
            break

        # Update buffer with the new experience
        dir = track.dir[car.progress]
        v = car.v
        buffer.record((prev_state, prev_v, prev_dir, action, reward, state, v, dir))

        # Update episodic rewards and statistics
        episodic_reward += reward
        episodic_v += np.linalg.norm(v)
        if reward == crash_reward:
            episodic_crash += 1

        # Save the model if it performs better
        if episodic_reward > best_reward:
            best_actor.set_weights(actor_model.get_weights())
            best_critic.set_weights(critic_model.get_weights())
            best_reward = episodic_reward

        # Train the models with experience from the buffer
        buffer.learn()

        # Update target networks
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        prev_state = state

    # Log and print episode results
    avg_reward = episodic_reward / n_t
    print(f'Episode {ep}')
    print(f'Mean reward: {avg_reward}')
    print(f'Mean v: {episodic_v / n_t * 3.6:.1f} km/h')
    print(f'Number of crashes: {episodic_crash} \n')

    avg_reward_list.append(avg_reward)

# Save the best models
best_actor.save(f'actor_model_{name}')
best_critic.save(f'critic_model_{name}')

# Plot average rewards over episodes
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Average Episodic Reward")
plt.title("Episode vs Average Episodic Reward")
plt.show()
