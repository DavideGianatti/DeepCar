import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

"""
This script defines a Deep Deterministic Policy Gradient (DDPG) agent for reinforcement learning,
including functions to create the actor and critic models,
an experience replay buffer class to store and sample experiences.
"""

def get_actor(n_angles):
    """
    Defines and returns the actor model for the reinforcement learning agent.

    Args:
        n_angles (int): Number of angles representing the distance measurements.

    Returns:
        tf.keras.Model: The actor model.
    """
    # Define input layers for distances, velocity, and direction
    input_dis = layers.Input(shape=(n_angles,))
    input_dis = layers.BatchNormalization()(input_dis)
    input_v = layers.Input(shape=(2,))
    input_v = layers.BatchNormalization()(input_v)
    input_dir = layers.Input(shape=(2,))

    # Concatenate all inputs into a single tensor
    inputs = layers.Concatenate()([input_dis, input_v, input_dir])

    # Define gas/break output branch
    gas_break = layers.Dense(64, activation="selu",
                       kernel_initializer=tf.zeros_initializer(),
                       bias_initializer=tf.zeros_initializer())(inputs)
    gas_break = layers.Dense(1, activation="tanh")(gas_break)

    # Define left/right output branch
    left_right = layers.Dense(64, activation="selu",
                       kernel_initializer=tf.zeros_initializer(),
                       bias_initializer=tf.zeros_initializer())(inputs)
    left_right = layers.Dense(1, activation="tanh")(left_right)

    # Concatenate the two branches to form the final output
    outputs = layers.Concatenate()([left_right, gas_break])

    # Create the actor model
    model = tf.keras.Model([input_dis, input_v, input_dir], outputs)

    return model

def get_critic(n_angles, n_actions):
    """
    Defines and returns the critic model for the reinforcement learning agent.

    Args:
        n_angles (int): Number of angles representing the distance measurements.
        n_actions (int): Number of actions.

    Returns:
        tf.keras.Model: The critic model.
    """
    # Define input layers for state
    input_dis = layers.Input(shape=(n_angles,))
    input_dis = layers.BatchNormalization()(input_dis)
    input_v = layers.Input(shape=(2,))
    input_v = layers.BatchNormalization()(input_v)
    input_dir = layers.Input(shape=(2,))

    # Process distances through dense layers
    distances = layers.Dense(n_angles, activation='selu')(input_dis)
    distances = layers.Dense(n_angles, activation='selu')(distances)

    # Process velocity through dense layers
    v = layers.Dense(4, activation='selu')(input_v)
    v = layers.Dense(8, activation='selu')(v)

    # Process direction through dense layers
    dir = layers.Dense(4, activation='selu')(input_dir)
    dir = layers.Dense(8, activation='selu')(dir)

    # Concatenate all processed state inputs
    state_inputs = layers.Concatenate()([distances, v, dir])

    # Further process concatenated state inputs
    state_out = layers.Dense(16, activation="selu")(state_inputs)
    state_out = layers.Dense(32, activation="selu")(state_out)

    # Define input layer for actions
    action_input = layers.Input(shape=(n_actions,))
    action_out = layers.Dense(32, activation="selu")(action_input)

    # Concatenate state and action outputs
    concat = layers.Concatenate()([state_out, action_out])

    # Process concatenated tensor through dense layers
    out = layers.Dense(64, activation="selu")(concat)
    out = layers.Dense(64, activation="selu")(out)
    out = layers.Dense(64, activation="selu",
                       kernel_initializer=tf.zeros_initializer(),
                       bias_initializer=tf.zeros_initializer())(out)
    outputs = layers.Dense(1)(out)  # Final output layer

    # Create the critic model
    model = tf.keras.Model([input_dis, input_v, input_dir, action_input], outputs)

    return model

class Buffer:
    """
    Experience replay buffer for storing and sampling experience tuples.

    Args:
        num_states (int): Number of state features.
        num_actions (int): Number of action features.
        target_actor (tf.keras.Model): Target actor model.
        target_critic (tf.keras.Model): Target critic model.
        critic_model (tf.keras.Model): Critic model.
        actor_model (tf.keras.Model): Actor model.
        critic_optimizer (tf.keras.optimizers.Optimizer): Optimizer for critic model.
        actor_optimizer (tf.keras.optimizers.Optimizer): Optimizer for actor model.
        gamma (float): Discount factor.
        buffer_capacity (int): Maximum number of experiences to store in the buffer. Defaults to 100000.
        batch_size (int): Number of experiences to sample from the buffer during training. Defaults to 64.
    """
    def __init__(self, num_states, num_actions, target_actor, target_critic, critic_model, actor_model,
                 critic_optimizer, actor_optimizer, gamma, buffer_capacity=100000, batch_size=64):
        # Buffer capacity and batch size
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        # Counter to track buffer usage
        self.buffer_counter = 0

        # Initialize buffers for states, actions, rewards, and next states
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.v_buffer = np.zeros((self.buffer_capacity, 2))
        self.dir_buffer = np.zeros((self.buffer_capacity, 2))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.next_v_buffer = np.zeros((self.buffer_capacity, 2))
        self.next_dir_buffer = np.zeros((self.buffer_capacity, 2))

        # Assign models and optimizers
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.critic_model = critic_model
        self.actor_model = actor_model
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.gamma = gamma

    def record(self, obs_tuple):
        """
        Record a new experience in the buffer.

        Args:
            obs_tuple (tuple): A tuple containing state, velocity, direction, action, reward, next state,
                               next velocity, and next direction.
        """
        # Determine index for storing the new experience
        index = self.buffer_counter % self.buffer_capacity

        # Store the experience in the respective buffers
        self.state_buffer[index] = obs_tuple[0]
        self.v_buffer[index] = obs_tuple[1]
        self.dir_buffer[index] = obs_tuple[2]
        self.action_buffer[index] = obs_tuple[3]
        self.reward_buffer[index] = obs_tuple[4]
        self.next_state_buffer[index] = obs_tuple[5]
        self.next_v_buffer[index] = obs_tuple[6]
        self.next_dir_buffer[index] = obs_tuple[7]

        # Increment buffer counter
        self.buffer_counter += 1

    @tf.function
    def update(self, state_batch, v_batch, dir_batch, action_batch, reward_batch, next_state_batch, next_v_batch, next_dir_batch):
        """
        Update the actor and critic networks using a batch of experiences.

        Args:
            state_batch (tf.Tensor): Batch of states.
            v_batch (tf.Tensor): Batch of velocities.
            dir_batch (tf.Tensor): Batch of directions.
            action_batch (tf.Tensor): Batch of actions.
            reward_batch (tf.Tensor): Batch of rewards.
            next_state_batch (tf.Tensor): Batch of next states.
            next_v_batch (tf.Tensor): Batch of next velocities.
            next_dir_batch (tf.Tensor): Batch of next directions.
        """
        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            # Predict target actions using target actor network
            target_actions = self.target_actor([next_state_batch, next_v_batch, next_dir_batch], training=True)
            # Compute target critic value
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, next_v_batch, next_dir_batch, target_actions], training=True
            )
            # Compute current critic value
            critic_value = self.critic_model([state_batch, v_batch, dir_batch, action_batch], training=True)
            # Compute critic loss
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        # Compute gradients for critic
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        # Apply gradients to critic optimizer
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            # Predict actions using actor network
            actions = self.actor_model([state_batch, v_batch, dir_batch], training=True)
            # Compute critic value using predicted actions
            critic_value = self.critic_model([state_batch, v_batch, dir_batch, actions], training=True)
            # Compute actor loss (negative of critic value)
            actor_loss = -tf.math.reduce_mean(critic_value)

        # Compute gradients for actor
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        # Apply gradients to actor optimizer
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def learn(self):
        """
        Sample a batch of experiences from the buffer and perform a learning step.
        """
        # Determine the range of records available
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Sample random batch indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert sampled experiences to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        v_batch = tf.convert_to_tensor(self.v_buffer[batch_indices])
        dir_batch = tf.convert_to_tensor(self.dir_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        next_v_batch = tf.convert_to_tensor(self.next_v_buffer[batch_indices])
        next_dir_batch = tf.convert_to_tensor(self.next_dir_buffer[batch_indices])

        # Perform update using the sampled batch
        self.update(state_batch, v_batch, dir_batch, action_batch, reward_batch, next_state_batch, next_v_batch, next_dir_batch)

@tf.function
def update_target(target_weights, weights, tau):
    """
    Update the target network parameters using a soft update method.

    Args:
        target_weights (tf.Tensor): Weights of the target network.
        weights (tf.Tensor): Weights of the current network.
        tau (float): Interpolation parameter for soft update.
    """
    for (a, b) in zip(target_weights, weights):
        # Interpolate between target and current weights using tau
        a = tf.multiply(b, tau) + tf.multiply(a, (1 - tau))
