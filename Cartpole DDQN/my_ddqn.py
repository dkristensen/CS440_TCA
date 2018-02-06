"""
Example of using DDQN to solve Atari games using OpenAO
gym environments. Followed the example by Ewan Li https://ewanlee.github.io/2017/07/09/Using-Tensorflow-and-Deep-Q-Network-Double-DQN-to-Play-Breakout/ 
"""
from __future__ import print_function

import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

from collections import deque, namedtuple

env = gym.envs.make("CartPole-v0")

action_space_size = env.action_space.n
observation_space_size = env.reset().shape[0]
possible_actions = [0,1]
learning_rate = 0.001
print("Action Space of size {}".format(action_space_size))
print("Observation Space of size {}".format(observation_space_size))

class StateProcessor():
    """
    Processes the input state for use with the network
    """
    
    def __init__(self):
        """
        Constructs the layout of the state processor. Not to be used in this example but including
        for use with my actual project to keep it clean
        """
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[observation_space_size],
                                                dtype=tf.float32)
            self.output = self.input_state
    def process(self, sess, state):
        return sess.run(self.output, {self.input_state:state})


class Estimator():
    """
    The actual estimator of the network
    """
    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope

        self.summary_writer = None
        with tf.variable_scope(scope):
            self._build_model()

            #if we specified a summary directory, write to it
            if(summaries_dir):
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        #Placeholders for the input
        self.x_pl = tf.placeholder(shape=[None,observation_space_size],
                                    dtype=tf.float32,
                                    name="x")

        self.target_pl = tf.placeholder(shape=[None],
                                    dtype=tf.float32,
                                    name='target')

        self.actions_pl = tf.placeholder(shape=[None],
                                    dtype=tf.int32,
                                    name="actions")

        batch_size = tf.shape(self.x_pl)[0]
        
        # Create the fully connected layer for the model
        fc1 = tf.contrib.layers.fully_connected(self.x_pl, 50, scope = 'fc1', activation_fn=tf.sigmoid)
        fc2 = tf.contrib.layers.fully_connected(fc1, 50, scope='fc2', activation_fn=tf.sigmoid)
        self.predictions = tf.contrib.layers.fully_connected(fc2, action_space_size, activation_fn=None, scope = 'out')

        # Gather the predictions for the action that the network chose for each input
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions,[-1]), gather_indices)

        # Compute the loss
        self.losses = tf.squared_difference(self.target_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer wiuth params from the original DDQN paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9)
        self.train_op = self.optimizer.minimize(self.loss, global_step = tf.contrib.framework.get_global_step())

        # Summaries
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, state):
        """
        Returns the prediction of the network for the state(s)
        """
        return sess.run(self.predictions, {self.x_pl: state})

    
    def update(self, sess, states, actions, targets):
        """
        Trains the network towards the targets
        """
        assert(len(states) == len(actions) and len(states) == len(targets)),\
            "Must input the same number of states, targets, and actions"
        
        feed_dict = {self.x_pl: states,
                    self.target_pl: targets,
                    self.actions_pl: actions}
        summaries, global_step, _, loss = sess.run(
                        [self.summaries,
                        tf.contrib.framework.get_global_step(),
                        self.train_op,
                        self.loss],
                        feed_dict)
        if(self.summary_writer):
            self.summary_writer.add_summary(summaries, global_step)
        return loss

def copy_model_parameters(sess, estimator_from, estimator_to):
    """
    Copies the weights from one network to another (the method behind DDQN)
    """
    from_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator_from.scope)]
    from_params = sorted(from_params, key=lambda v: v.name)
    to_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator_to.scope)]
    to_params = sorted(to_params, key=lambda v: v.name)

    update_ops =[]
    for from_v, to_v in zip(from_params,to_params):
        op = to_v.assign(from_v)
        update_ops.append(op)

    sess.run(update_ops)

    
def make_epsilon_greedy_policy(estimator, num_actions):
    """
    Creates an epsilon-greedy policy using the Q-value approximator network
    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        q_values = estimator.predict(sess, np.expand_dims(observation,0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=50000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor = 0.99,
                    epsilon_start = 1.0,
                    epsilon_end=0.05,
                    epsilon_decay_steps=500000,
                    batch_size = 64,
                    record_video_every = 5000):
    
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # Create the directories for the checkpoints, summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if(not os.path.exists(checkpoint_dir)):
        os.makedirs(checkpoint_dir)
    if(not os.path.exists(monitor_path)):
        os.makedirs(monitor_path)
    
    saver = tf.train.Saver()
    # Load previous checkpoint if one exists
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if(latest_checkpoint):
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    
    # Get current time step
    current_time = sess.run(tf.contrib.framework.get_global_step())

    # Set the epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy to follow
    policy = make_epsilon_greedy_policy(q_estimator, action_space_size)

    # set up the replay memory
    replay_memory = []

    # Populate replay memory with the initial experience 
    print("Populating replay memory...")
    state = env.reset()
    state = state_processor.process(sess, state)
    for  i in range(replay_memory_init_size):
        action_probs = policy(sess,state,epsilons[min(current_time, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(possible_actions[action])
        next_state = state_processor.process(sess, next_state)
        replay_memory.append(Transition(state,action,reward,next_state, done))
        if(done):
            state = env.reset()
            state = state_processor.process(sess, state)
        else:
            state = next_state

    env = Monitor(env, directory = monitor_path, video_callable = lambda count:count%1 == 0, resume = True)
    with open("results.csv", "w") as results:
        for i_episode in range(num_episodes):
            
            #Save the checkpoint
            saver.save(tf.get_default_session(), checkpoint_path)

            # Reset the environment
            state = env.reset()
            state = state_processor.process(sess, state)
            loss = None

            for t in itertools.count():

                #Get the epsilon for the current time
                epsilon = epsilons[min(current_time, epsilon_decay_steps-1)]

                # Add this epsilon to tensorboard
                episode_summary = tf.Summary()
                episode_summary.value.add(simple_value=epsilon, tag="epsilon")
                q_estimator.summary_writer.add_summary(episode_summary, current_time)

                # If at the step, update the target estimator
                if( current_time % update_target_estimator_every == 0):
                    copy_model_parameters(sess, q_estimator, target_estimator)
                    print("\nCopied model parameters to target network.")
                
                # Print the current step
                print("\r Step {} ({}) @ Episode {}/{}, loss: {}".format(t,
                                                                            current_time,
                                                                            i_episode+1,
                                                                            num_episodes,
                                                                            loss),
                                                                            end="")
                sys.stdout.flush()

                # Take a step
                action_probs = policy(sess, state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
                next_state, reward, done, _ = env.step(possible_actions[action])
                next_state = state_processor.process(sess, next_state)
                
                # if replay memory is full, make space by popping first
                if(len(replay_memory) == replay_memory_size):
                    replay_memory.pop(0)
                
                # add to replay memory
                replay_memory.append(Transition(state,action,reward,next_state,done))

                # Get minibatch for training
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch =map(np.array, zip(*samples))

                # DQN settings
                # compute q values and targets
                # q_values_next = target_estimator.predict(sess, next_states_batch)
                # targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_values_next, axis=1)

                # DDQN Settings
                # Compute q-values and targets
                q_values_next = q_estimator.predict(sess,next_states_batch)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = target_estimator.predict(sess,next_states_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * q_values_next_target[np.arange(batch_size), best_actions]



                # Gradient Descent Update
                states_batch = np.array(states_batch)
                loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

                if(done):
                    print("{},{}".format(i_episode,t),file=results)
                    break
            
                state = next_state
                current_time+=1

    return stats

tf.reset_default_graph()

experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# Make a global step
global_step = tf.Variable(0, name="global_step", trainable=False)

# Initialize estimators 
q_estimator = Estimator(scope='q', summaries_dir=experiment_dir)
target_estimator = Estimator(scope='target_q')

state_processor = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t,stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator = target_estimator,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=1000,
                                    replay_memory_size=5000,
                                    replay_memory_init_size=400,
                                    update_target_estimator_every=1000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=10000,
                                    discount_factor=1.0,
                                    batch_size=32):
        print("\nEpisode Reward: {}".format(stats.episode_reward[-1]))





    




