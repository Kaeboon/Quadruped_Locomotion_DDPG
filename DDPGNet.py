# Code written by ME5418 AY23/24 Group 18
# This python define a class to create the networks and define their forward / backward pass for DDPG application for our project.

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import random

# Default values for training parameters. You may pass your own values from the LA demo code.
GAMMA = 0.99
TAU = 1e-3
ACTOR_LR = 1e-4
ACTOR_L2_REG = 1e-5
ACTOR_GRAD_CLIP = 1
CRITIC_LR = 1e-3
CRITIC_L2_REG = 2e-4
CRITIC_GRAD_CLIP = 1

# For dealing with experience buffer sampling:
BATCH_SIZE = 128

# For OU's parameter:
MU = 0.0
THETA = 0.15
SIGMA = 0.2
DECAY_R = 0.99
RESET_INTERVAL = 100

# Functions to scale the actions and observations value: These functions are used in the sample_batch() method below.

def scale_actions(actions): # This function is used to scale the range of our actions to [-1, 1].
    scaled_action = []
    for i in range(len(actions)):
        scaled_action.append((actions[i] - (-26.5))*(1 - (-1)) / (26.5 - (-26.5)) + (-1)) # The maximum torque of aliengo motors is found to be 26.5 Nm.
    return np.array(scaled_action)

def scale_observation(observations, previous_action): # This function is used to scale the range of observatios to [-1, 1]
    scaled_observation = []
    scaled_observation.append(observations['robot_torso_position'][1]) # y torso position
    scaled_observation.append(observations['robot_torso_position'][2]) # z torso position
    scaled_observation.append( ( observations['robot_torso_velocity'][0] - (-1.667)) * (1 + (-1)) / (1.667 - (-1.667)) + (-1) )
    scaled_observation.append(observations['robot_torso_velocity'][1])
    scaled_observation.append(observations['robot_torso_velocity'][2])
    scaled_observation.append(observations['robot_torso_orientation'][0])
    scaled_observation.append(observations['robot_torso_orientation'][1])
    scaled_observation.append(observations['robot_torso_orientation'][2])
    scaled_observation.append(observations['robot_torso_orientation'][3])
    scaled_observation.append(observations['robot_torso_ori_rate'][0])
    scaled_observation.append(observations['robot_torso_ori_rate'][1])
    scaled_observation.append(observations['robot_torso_ori_rate'][2])
    for joint in ["FR_upper_joint", "FR_lower_joint", "FL_upper_joint", "FL_lower_joint", "RR_upper_joint", "RR_lower_joint", "RL_upper_joint", "RL_lower_joint"]:
        scaled_observation.append( (observations["each_joint_angle"][joint] - (-3.14159)) * (1 - (-1)) / (3.14159- (- 3.14159)) + (-1) ) # The joint angle ranges from -pi to pi.
    for joint in ["FR_upper_joint", "FR_lower_joint", "FL_upper_joint", "FL_lower_joint", "RR_upper_joint", "RR_lower_joint", "RL_upper_joint", "RL_lower_joint"]:
        scaled_observation.append(observations["each_joint_vel"][joint]) # The joint angle ranges from -pi to pi.
    for joint in ["FR_toe", "FL_toe", "RR_toe", "RL_toe"]:
        scaled_observation.append(observations["leg_contact_with_ground"][joint])
    for i in range(8):
        scaled_observation.append(previous_action[i])

    return np.array(scaled_observation)

# Define our Ornstein-Uhlenbeck noise class
class OUNoise:
    def __init__(self, mu=MU, theta=THETA, sigma = SIGMA, decay_r= DECAY_R):
        self.mu = mu # mean
        self.theta = theta # rate of mean reversion
        self.sigma = sigma
        self.decay_rate = decay_r # decay period
        self.action_dim = 8
        self.low = -1
        self.high = 1
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def get_action(self, action): # t is sampling time
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        self.sigma *= self.decay_rate
        return np.clip(action + self.state, self.low, self.high)

class DDPGNet:
    def __init__(self, observe_size, action_size, TRAINING, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, actor_l2_reg=ACTOR_L2_REG, actor_grad_clip=ACTOR_GRAD_CLIP, critic_lr=CRITIC_LR, critic_l2_reg=CRITIC_L2_REG, critic_grad_clip=CRITIC_GRAD_CLIP, batch_size=BATCH_SIZE):

        # Observation and Action dimension
        self.action_size = action_size # should be 32 for our case
        self.observe_size = observe_size # should be 8 for our case


        # Input placeholders
        self.observation_input_placeholder = tf.placeholder(shape=[None, observe_size], dtype=tf.float32)
        self.next_observation_input_placeholder = tf.placeholder(shape=[None, observe_size], dtype=tf.float32)
        self.action_input_placeholder = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        # Weight and Bias initializer
        self.w_init = tf.random_normal_initializer(mean = 0, stddev = 0.0005)
        self.b_init = tf.constant_initializer(0.1)

        # Build Actor, Target_Actor, Critic, Target Critic 1, Target Critic 2
        self.actor_net_output = self.build_actor_net()
        self.target_actor_net_output = self.build_target_actor_net()
        self.critic_net_output = self.build_critic_net()
        self.target_critic_1_net_output = self.build_target_critic_1_net()
        self.target_critic_2_net_output = self.build_target_critic_2_net()

        if TRAINING:

            # Initialize some training parameters
            self.gamma = gamma
            self.tau = tau
            self.actor_lr = actor_lr
            self.actor_l2_reg = actor_l2_reg
            self.actor_grad_clip = actor_grad_clip
            self.critic_lr = critic_lr
            self.critic_l2_reg = critic_l2_reg
            self.critic_grad_clip = critic_grad_clip
            self.batch_size = batch_size

            # Replay Buffer
            self.replay_buffer = []

            # Reward Placeholder
            self.reward_placeholder = tf.placeholder(shape=[None], dtype=tf.float32)

            # Get the respective trainable variables from each network
            self.actor_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
            self.critic_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
            self.target_actor_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')
            self.target_critic_1_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic_1')
            self.target_critic_2_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic_2')

            # Define loss and trainer for actor network
            self.actor_weights = [var for var in self.actor_trainable_vars]
            self.actor_l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in self.actor_weights])


            self.actor_loss = -tf.reduce_mean(self.target_critic_1_net_output) + ACTOR_L2_REG * self.actor_l2_loss  # add a negative sign, because optimizer always minimizes only. So here we minimize negative Q value
            self.actor_optimizer = tf.train.AdamOptimizer(ACTOR_LR)

            self.actor_gradients = self.actor_optimizer.compute_gradients(self.actor_loss, var_list=self.actor_trainable_vars)
            self.clipped_actor_gradients, _ = tf.clip_by_global_norm([grad for grad, var in self.actor_gradients], clip_norm = ACTOR_GRAD_CLIP)
            self.actor_trainer = self.actor_optimizer.apply_gradients(zip(self.clipped_actor_gradients, self.actor_trainable_vars))


            # Define loss and trainer for critic network
            self.target_Q = self.reward_placeholder + GAMMA * self.target_critic_2_net_output # Target_Q = Reward + Discount * Q_next
            self.TD_error = self.target_Q - self.critic_net_output # TD_error = Target_Q - Q


            self.critic_weights = [var for var in self.critic_trainable_vars]
            self.critic_l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in self.critic_weights])

            self.critic_loss = tf.reduce_mean(tf.square(self.TD_error)) + CRITIC_L2_REG * self.critic_l2_loss    # same as the MSE error between Target_Q and Q
            self.critic_optimizer = tf.train.AdamOptimizer(CRITIC_LR)

            self.critic_gradients = self.critic_optimizer.compute_gradients(self.critic_loss, var_list=self.critic_trainable_vars)
            self.clipped_critic_gradients, _ = tf.clip_by_global_norm([grad for grad, var in self.critic_gradients], clip_norm = CRITIC_GRAD_CLIP)

            self.critic_trainer = self.critic_optimizer.apply_gradients(zip(self.clipped_critic_gradients, self.critic_trainable_vars))

            # self.critic_trainer = self.critic_optimizer.minimize(self.critic_loss, var_list = self.critic_trainable_vars) # Only update Critic trainable variables

            # Define the soft update for targets network
            self.update_target_ops = []
            for i in range(len(self.target_actor_trainable_vars)):
                self.update_target_ops.append(self.target_actor_trainable_vars[i].assign(TAU * self.actor_trainable_vars[i] + (1-TAU) * self.target_actor_trainable_vars[i])) #Target network new weight = TAU * Main network weight + (1 - TAU) * Target network old weight.
            for i in range(len(self.target_critic_1_trainable_vars)):
                self.update_target_ops.append(self.target_critic_1_trainable_vars[i].assign(TAU * self.critic_trainable_vars[i] + (1-TAU) * self.target_critic_1_trainable_vars[i])) # Target network new weight = TAU * Main network weight + (1 - TAU) * Target network old weight.
            for i in range(len(self.target_critic_1_trainable_vars)):
                self.update_target_ops.append(self.target_critic_2_trainable_vars[i].assign(self.target_critic_1_trainable_vars[i])) # We use updated Target Critic 1 to update Target Critic 1

    def build_actor_net(self):

        with tf.variable_scope("actor", reuse=False):
            actor_fc_1 = layers.fully_connected(inputs = self.observation_input_placeholder, num_outputs = 400 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = tf.nn.relu)
            actor_fc_2 = layers.fully_connected(inputs = actor_fc_1, num_outputs = 300 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = tf.nn.relu)
            return layers.fully_connected(inputs = actor_fc_2, num_outputs = self.action_size , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = tf.nn.tanh)

    def build_target_actor_net(self):

        with tf.variable_scope("target_actor", reuse=False):
            target_actor_fc_1 = layers.fully_connected(inputs = self.next_observation_input_placeholder, num_outputs = 400 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = tf.nn.relu)
            target_actor_fc_2 = layers.fully_connected(inputs = target_actor_fc_1, num_outputs = 300 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = tf.nn.relu)
            return layers.fully_connected(inputs = target_actor_fc_2, num_outputs = self.action_size , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = tf.nn.tanh)

    def build_critic_net(self):

        with tf.variable_scope("critic", reuse=False):
            critic_fc_1 = layers.fully_connected(inputs = self.observation_input_placeholder, num_outputs = 400 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = tf.nn.relu)
            critic_fc_2 = layers.fully_connected(inputs = critic_fc_1, num_outputs = 300 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = None)
            critic_fc_3 = layers.fully_connected(inputs = self.action_input_placeholder, num_outputs = 300 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = None)
            critic_add = tf.add(critic_fc_2, critic_fc_3)
            critic_add = tf.nn.relu(critic_add)
            return layers.fully_connected(inputs = critic_add, num_outputs = 1 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = None)

    def build_target_critic_1_net(self):

        with tf.variable_scope("target_critic_1", reuse=False):
            target_critic_1_fc_1 = layers.fully_connected(inputs = self.observation_input_placeholder, num_outputs = 400 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = tf.nn.relu)
            target_critic_1_fc_2 = layers.fully_connected(inputs = target_critic_1_fc_1, num_outputs = 300 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = None)
            target_critic_1_fc_3 = layers.fully_connected(inputs = self.actor_net_output, num_outputs = 300 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = None)
            target_critic_1_add = tf.add(target_critic_1_fc_2, target_critic_1_fc_3)
            target_critic_1_add = tf.nn.relu(target_critic_1_add)
            return layers.fully_connected(inputs = target_critic_1_add, num_outputs = 1 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = None)

    def build_target_critic_2_net(self):

        with tf.variable_scope("target_critic_2", reuse=False):
            target_critic_2_fc_1 = layers.fully_connected(inputs = self.next_observation_input_placeholder, num_outputs = 400 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = tf.nn.relu)
            target_critic_2_fc_2 = layers.fully_connected(inputs = target_critic_2_fc_1, num_outputs = 300 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = None)
            target_critic_2_fc_3 = layers.fully_connected(inputs = self.target_actor_net_output, num_outputs = 300 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = None)
            target_critic_2_add = tf.add(target_critic_2_fc_2, target_critic_2_fc_3)
            target_critic_2_add = tf.nn.relu(target_critic_2_add)
            return layers.fully_connected(inputs = target_critic_2_add, num_outputs = 1 , weights_initializer = self.w_init, biases_initializer = self.b_init, activation_fn = None)

    # The below three methods are actually not used in our Learning Agent. We define it for our NN demo code only.

    def store_to_buffer(self, current_state, action, next_state, reward, terminate):
        self.replay_buffer.append(((scale_observation(current_state)), scale_actions(action), scale_observation(next_state), np.array(reward), np.array(terminate))) # Directly append the experiences gotten from step() from OpenAI gym

    def sample_batch(self): # Randomly sample a few of experience from the replay buffer, then do some reshaping of their dimension to fit the dimension of our input layers.
        assert(len(self.replay_buffer) > BATCH_SIZE)
        batch = np.array(random.sample(self.replay_buffer, BATCH_SIZE), dtype=object)
        current_observation = np.stack(batch[:, 0])
        action = np.stack(batch[:, 1])
        next_observation = np.stack(batch[:, 2])
        reward = np.stack(batch[:, 3])
        return current_observation, action, next_observation, reward

    def clear_replay_buffer(self): # To prevent list overflow. Good to run this once in a while when actual training.
        self.replay_buffer.clear()