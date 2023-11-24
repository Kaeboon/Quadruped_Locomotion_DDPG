
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
import imageio
import argparse

# Import gym environment and DDPG (actor, critic, target) networks
from aliengo_gym import AliengoEnv
from DDPGNet import DDPGNet
from DDPGNet import scale_actions
from DDPGNet import scale_observation
from DDPGNet import OUNoise

# check your computer information
from tensorflow.python.client import device_lib
dev_list = device_lib.list_local_devices()
print(dev_list)

# Path to the folders saving the model parameters, render gifs, tf summary, and episode buffer informations
model_path = os.path.dirname(os.path.abspath('__file__')) +  "/model_save"
gifs_path = os.path.dirname(os.path.abspath('__file__')) + "/gifs_save"
train_path = os.path.dirname(os.path.abspath('__file__')) + "/tf_summary_save"
episode_buffer_path = os.path.dirname(os.path.abspath('__file__')) + "/episode_buffer_save"

parser = argparse.ArgumentParser(description="Process input")
parser.add_argument('--TRAINING', type=int, help="Value for TRAINING variable")
parser.add_argument('--Policy', type=int, help="Value for Policy variable")

args = parser.parse_args()

if args.TRAINING is None or args.Policy is None:
    parser.error("You must provide values for both TRAINING and Policy")

TRAINING = bool(args.TRAINING)
Policy = args.Policy

# To load previously trained model?

if TRAINING == True:
    Policy = 0 # Set to 0 if you want to start from scratch. Else, set your policy value.

if Policy == 0:
    load_model = False # Toggle True to load previously trained model
elif Policy == 1:
    load_model = True
    MODEL_NUMBER = 2600 # which model number you want to load?
elif Policy == 2:
    load_model = True
    MODEL_NUMBER = 500 # which model number you want to load?
elif Policy == 3:
    load_model = True
    MODEL_NUMBER = 200 # which model number you want to load?
else:
    load_model = False
    MODEL_NUMBER = 0

# Parameters that decides when to save the model checkpoint, render gifs, tf summary and episode buffer informations.

# Save GIF
OUTPUT_GIFS = True # toggle True to save GIFS
NEXT_GIF_COUNT = 100 # To save GIFS at every NEXT_GIFS_COUNT episodes interval

# Save episode buffer, tf summary and model checkpoint
SAVE_EPISODE_BUFFER = True # toggle True to save episode buffer
SUMMARY_WINDOW = 5  # To save tf summary, and save model checkpoint at every SUMMARY_WINDOW episodes interval

# global variables used tensorboard visualization. tf summary will be saved to these two variable and regularly  written out to local files
episode_rewards = [] # do not edit
episode_lengths = [] # do not edit

# TRAINING and TESTING:
MAX_EPISODE_LENGTH = 1000 # maximum steps to run in each episode during testing

episode_count = 0 # (do not edit this variable) gloal variable to save how much episodes has already been gone through

# TESTING
NUM_EXPS = 1 # How many episodes you want to test?

# TRAINING: for initializing actor, critic, target actor, target critic network
observe_size = 40
action_size = 8

# TRAINING: for initializing actor, critic, target actor, target critic's trainer

# TRAINING: for storing experience buffer and extracting a batch of experiences
EXPERIENCE_BUFFER_SIZE = 10 # How much experience should be stored in the experience buffer before being used for backward pass.
BATCH_SIZE = 10 # How much experiences should be extracted out from the experience buffer to be used for backward pass

# TRAINING: for OU Noise parameter
MU = 0.0
THETA = 0.15
SIGMA = 0.2
DECAY_R = 0.99
RESET_INTERVAL = 100

# - for updating actor network
ACTOR_LR = 1e-3
ACTOR_L2_REG = 1e-5
ACTOR_GRAD_CLIP = 1
# - for updating critic network
CRITIC_LR = 1e-3
CRITIC_L2_REG = 2e-4
CRITIC_GRAD_CLIP = 1
# - for soft updating target network
GAMMA = 0.99
TAU = 1e-3

# Save render to gif helper function
def make_gif(images, fname):
    imageio.mimwrite(fname, images, subrectangles=True)
    print(f"Episode {episode_count - 1}. GIF is saved")

class Worker:

    def __init__(self, game, global_network, noise):
        self.env = game # Give it access to our GYM environment
        self.global_network = global_network # Give it access to our actor, critic, target actor, target critic network
        self.noise = noise # Give it access to our noise instance

    def train(self, rollout, sess):
        global episode_count

        # To randomly sample a few of experience from the replay buffer, then do some reshaping of their dimension to fit the dimension of our input layers.
        batch = np.array(random.sample(rollout, min(BATCH_SIZE, len(rollout))), dtype=object)
        current_observations = np.stack(batch[:, 0])
        actions = np.stack(batch[:, 1])
        next_observations = np.stack(batch[:, 2])
        rewards = np.stack(batch[:, 3])

        # Backward pass for actor network, obtain actor loss
        feed_dict = {self.global_network.observation_input_placeholder: current_observations}

        actor_loss, _ = sess.run([self.global_network.actor_loss, self.global_network.actor_trainer], feed_dict = feed_dict)

        # Backward pass for critic network, obtain critic loss
        feed_dict = {self.global_network.observation_input_placeholder: current_observations,
                     self.global_network.action_input_placeholder: actions,
                     self.global_network.reward_placeholder: rewards,
                     self.global_network.next_observation_input_placeholder: next_observations}

        critic_loss, _ = sess.run([self.global_network.critic_loss, self.global_network.critic_trainer], feed_dict=feed_dict)

        # Soft update for Target Actor and Target Critics
        _ = sess.run(self.global_network.update_target_ops)

        return actor_loss, critic_loss

    def shouldRun(self, coord, episode_count): # Just a method to decide when to stop the training / testing operation
        if TRAINING:
            return (not coord.should_stop()) # For training, this is just an infinity loop
        else:
            return (episode_count < NUM_EXPS) # For testing, it will only run for NUM_EXPS episodes.

    def work(self, sess, coord, saver, summary):

        global episode_count, episode_rewards, episode_lengths # global variable for saving information that would be passed to global_summary at fixed interval
        # episode_count -> how many episode has been gone through
        # episode_rewards -> a list of (the sum of all reward obtain at each step at current episode)
        # episode_lengths -> a list of (how many steps are taken at current episode)

        self.global_summary = summary # our tf summary file writer -> for tensorboard visualization use

        self.nextGIF = episode_count # nextGIF will decide at which episode should the training render be saved as GIF

        with sess.as_default(), sess.graph.as_default(): # session (sess) and its associated graph become the default session and default graph.

            while self.shouldRun(coord, episode_count): # LOOP THRU EPISODES

                episode_buffer = [] # Clean out the episode buffer. Used to save our experiences
                episode_reward = 0 # Clean out the episode reward. Used to save all

                d = False # Has it reached the termination state? Later on this value will be re-evaluated after each env.step()
                saveGIF = False # Do I need to save GIF for this episode? Later on this value will be re-evaluated according to the nextGIF value.

                self.env.reset(reload_urdf=False) # Reset environemnt -> return aliengo robot back to initial position orientation.
                self.noise.reset() # reset OU noise state

                if OUTPUT_GIFS:

                    if (not TRAINING) or (episode_count >= self.nextGIF): # To evaluate whether saving GIF is needed. Always True for testing. Only once a NEXT_GIF_COUNT for training
                        saveGIF = True
                        self.nextGIF = episode_count + NEXT_GIF_COUNT
                        GIF_episode = int(episode_count)
                        episode_frames = [self.env.render()] # save the render RGB array into our episode_frames

                episode_step_count = 0 # Number of steps taken in one episode

                while True: # LOOP THRU STEPS

                    # Please note that:
                    # s: current observation. a: action taken. s1: next observation. r: reward. d: is it termination state?

                    if episode_step_count == 0:
                        s = self.env.get_state() # Get current state
                        a = [0,0,0,0,0,0,0,0]

                    # Just reshaping the state data
                    s_list = []
                    s_list.append((scale_observation(s, a), )) # normalize state values
                    s_list = np.array(s_list, dtype=object)
                    s = np.stack(s_list[:, 0])

                    # Obtain the action as output by our actor network based on current state
                    feed_dict = {self.global_network.observation_input_placeholder: s}
                    a = sess.run(self.global_network.actor_net_output, feed_dict = feed_dict)

                    if TRAINING:
                        a = self.noise.get_action(action=a) # Add noise to our state using OU Noise

                    a = (a + 1)*(26.5+26.5)/2 - 26.5 # undo the normalization
                    a = tuple(a[0]) # reshaping the action value

                    # Perform a step to obtain next state, reward, is it termination state
                    s, _, s1, r, d, pa = self.env.step(a , episode_step_count)

                    if saveGIF:
                        episode_frames.append(self.env.render()) # only add a new RGB array of current render if saving GIF is required

                    episode_buffer.append((scale_observation(s, scale_actions(pa)), scale_actions(a), scale_observation(s1, scale_actions(a)), np.array(r), np.array(d))) # then append the experience (s, a, s1, r, d) to our episode buffer
                    episode_reward += r # add value to our episode_reward
                    s = s1 # now next observation will be our current observation for the next loop

                    episode_step_count +=1 # update episode step count

                    if TRAINING and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d): # once it episode buffer has stored as much as the EXPERIENCE BUFFER SIZE, backward pass will be performed.
                        if (len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE):
                            episode_buffer_training = episode_buffer[-EXPERIENCE_BUFFER_SIZE:] # Only obtain the last EXPERIENCE_BUFFER_SIZE of experiences for backward pass
                        else:
                            episode_buffer_training = episode_buffer[:] # Get all of the experience buffer to be used for training

                        actor_loss, critic_loss = self.train(episode_buffer_training, sess) # calling the train method

                        print(f"Episode {episode_count}. Training Carried out at step = {episode_step_count}. Actorloss = {actor_loss}, Critic_loss = {critic_loss}")

                    if episode_step_count >= MAX_EPISODE_LENGTH or d: # check if termination state is reached
                        if d == True:
                            print(f"Episode: {episode_count}. Termination state hit at step = {episode_step_count}")
                        break

                # Update our global variables used for tf.summary
                episode_lengths.append(episode_step_count)
                episode_rewards.append(episode_reward)

                episode_count += 1

                if not TRAINING: # Print our testing information
                    print(f"Episode {episode_count-1} Completed. Total step = {episode_step_count}. \n")
                    print("Saving GIF ......")
                    GIF_episode = int(episode_count)

                else: # if TRAINING

                    if episode_count % SUMMARY_WINDOW == 0: # check if needed to save tf summary for tensorboard visualization, and update our model checkpoint

                        # Save Model Checkpoint

                        saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.ckpt')
                        print(f"Episode {episode_count-1}. Model checkpoint is saved \n")

                        # To update our tf.summary and Tensorboard Visualization

                        summary = tf.Summary()

                        mean_reward = np.nanmean(episode_rewards[-SUMMARY_WINDOW:])
                        mean_length = np.nanmean(episode_lengths[-SUMMARY_WINDOW:])
                        summary.value.add(tag='Perf/Reward', simple_value = mean_reward)
                        summary.value.add(tag='Perf/Length', simple_value = mean_length)
                        summary.value.add(tag="Losses/Actor Loss", simple_value = actor_loss )
                        summary.value.add(tag="Losses/Critic Loss", simple_value = critic_loss )# In truth, within one episode, there will be multiple backward pass. Here only takes in the last backward pass loss

                        self.global_summary.add_summary(summary, int(episode_count))
                        self.global_summary.flush()

                        print(f"Episode {episode_count-1}. tf.summary -> Tensorboard is updated \n")

                if saveGIF:

                    # To generate GIF file. file name format: episode_<which episodes>_<how many steps taken>_<how much is the reward in this episode>

                    images = np.array(episode_frames)
                    if TRAINING:
                        make_gif(images, "{}/episode_{:d}_{:d}_{:.1f}.gif".format(gifs_path, GIF_episode, episode_step_count, episode_reward))
                    else:
                        make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path, GIF_episode, episode_step_count))

                if SAVE_EPISODE_BUFFER:

                    # To save the experience buffer. file name format: episode_<which episode>

                    with open(episode_buffer_path + "/episode_{}.dat".format(GIF_episode), "wb") as file:
                        pickle.dump(episode_buffer, file)

        self.env.disconnect() # To terminate the PyBullet window after training is over. This would not run if training is stop manually, thus the extra last cell in this notebook.


tf.reset_default_graph()
print("Hello World!")
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(episode_buffer_path):
    os.makedirs(episode_buffer_path)

config = tf.ConfigProto(allow_soft_placement = True) #  to allow the placement of an operation on a device other than the one explicitly requested

with tf.device("/cpu:0"): # for now just use cpu to run. Might consider changing to gpu in the future to speed up training process.

    master_network = DDPGNet(observe_size=observe_size, action_size=action_size, TRAINING=TRAINING, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, actor_l2_reg=ACTOR_L2_REG, actor_grad_clip=ACTOR_GRAD_CLIP, critic_lr=CRITIC_LR, critic_l2_reg=CRITIC_L2_REG, critic_grad_clip=CRITIC_GRAD_CLIP, batch_size=BATCH_SIZE)

    gameEnv = AliengoEnv(render = True, Policy = Policy) # For us to visualize how each episode runs
    gameEnv.reset(reload_urdf=True) # To import the model in
    noise = OUNoise(MU, THETA, SIGMA, DECAY_R) # initializing OU Noise
    noise.reset() # reset the OU noise
    worker = Worker(gameEnv, master_network, noise) # Initialize our Worker Agent

    global_summary = tf.summary.FileWriter(train_path) # Initializing our tf summary -> for tensorboard visualisation
    saver = tf.train.Saver(max_to_keep = 2) # Initializing our tf saver -> for saving -> for saving model checkpoint

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()

        if load_model == True: # To load the model back from our tf checkpoint files
            print("Loading Model...")
            if not TRAINING:
                with open(model_path+'/checkpoint', 'w') as file:
                    file.write('model_checkpoint_path: "model-{}.ckpt"'.format(MODEL_NUMBER))
                    file.close()
            ckpt = tf.train.get_checkpoint_state(model_path)
            p=ckpt.model_checkpoint_path
            p=p[p.find('-')+1:]
            p=p[:p.find('.')]
            if TRAINING:
                episode_count = int(p)
            else:
                episode_count = 0
            saver.restore(sess,ckpt.model_checkpoint_path)
            print("episode_count set to ", episode_count)

        worker.work(sess, coord, saver, global_summary) # Start Training / Testing operation

# if not TRAINING:
#     print([np.mean(episode_lengths), np.sqrt(np.var(episode_lengths)), np.mean(np.asarray(np.asarray(episode_lengths) < MAX_EPISODE_LENGTH, dtype=float))])
