import tensorflow as tf
import simmodel
import numpy as np


class ExperienceReplayMemory(object):

    def __init__(self, replay_memory_size):
        self.replay_memory_size = replay_memory_size
        self.consolidated_experience = EpisodeExperience()

    def store_experience(self, episode_history):
        self.consolidated_experience.consolidate(episode_history)

        while self.consolidated_experience.size() > self.replay_memory_size:
            self.consolidated_experience.purge()

    def sample_transitions(self, batch_size):
        batch = EpisodeExperience()

        if self.consolidated_experience.size() == 0:
            raise Exception("No transitions for sampling.")

        selected_indexes = np.arange(self.consolidated_experience.size())
        np.random.shuffle(selected_indexes)
        selected_indexes = selected_indexes[:batch_size]

        for index in selected_indexes:
            batch.observe_action_effects(state=self.consolidated_experience.states[index],
                                         action=self.consolidated_experience.actions[index],
                                         reward=self.consolidated_experience.rewards[index],
                                         new_state=self.consolidated_experience.new_states[index])

        return np.array(batch.states), np.array(batch.actions), np.array(batch.rewards), np.array(batch.new_states)


class EpisodeExperience(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []

    def size(self):
        return len(self.states)

    def observe_action_effects(self, state, action, reward, new_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)

    def consolidate(self, episode_history):
        self.states += episode_history.states
        self.actions += episode_history.actions
        self.rewards += episode_history.rewards
        self.new_states += episode_history.new_states

    def purge(self):
        self.states.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.new_states.pop(0)


class DeepQLearner(object):

    def __init__(self, name, input_number, hidden_units, logger, learning_rate=None, discount_factor=None,
                 counter_for_learning=None,
                 initial_epsilon=None, final_epsilon=None, decay_steps=None, replay_memory_size=None):
        self.logger = logger
        self.name = name
        self.episode_experience = EpisodeExperience()
        self.metric_catalogue = []

        self.actions = [simmodel.CLEAN_ACTION, simmodel.SLOPPY_ACTION]

        self.prediction_scope = self.name + '-prediction_network'
        self.pred_states, self.pred_q_values = self.build_network(self.prediction_scope, input_number, hidden_units)

        if learning_rate is not None:
            self.initial_epsilon = initial_epsilon
            self.final_epsilon = final_epsilon
            self.decay_steps = decay_steps
            self.replay_memory = ExperienceReplayMemory(replay_memory_size=replay_memory_size)
            self.discount_factor = discount_factor
            self.counter_for_learning = counter_for_learning

            self.target_scope = self.name + '-target_network'
            self.target_states, self.target_q_values = self.build_network(self.target_scope, input_number,
                                                                          hidden_units)

            self.train_target_q, self.train_actions, self.train_loss, self.train_operation = self.build_training_operation(
                learning_rate)

    def new_episode(self):
        self.episode_experience = EpisodeExperience()

    def observe_action_effects(self, state, action, reward, new_state):
        self.episode_experience.observe_action_effects(state, action, reward, new_state)

    def sample_transitions(self, batch_size):
        self.logger.debug(self.name + "-Memory size: " + str(
            self.replay_memory.consolidated_experience.size()) + " .Sampling: " + str(batch_size))
        return self.replay_memory.sample_transitions(batch_size)

    def store_experience(self):
        self.logger.debug(self.name + "-Consolidating experience: " + str(self.episode_experience.size()))
        self.replay_memory.store_experience(self.episode_experience)

    def build_network(self, variable_scope, input_number, hidden_units):
        with tf.variable_scope(variable_scope):
            states = tf.placeholder(tf.float32, shape=[None, input_number], name="state")

            initializer = tf.variance_scaling_initializer()
            hidden_layer_1 = tf.layers.dense(states, hidden_units, activation=tf.nn.elu, kernel_initializer=initializer,
                                             name="hidden_1")
            hidden_layer_2 = tf.layers.dense(hidden_layer_1, hidden_units, activation=tf.nn.elu,
                                             kernel_initializer=initializer,
                                             name="hidden_2")
            outputs = tf.layers.dense(hidden_layer_2, len(self.actions), kernel_initializer=initializer,
                                      name="q_values")

        return states, outputs

    def get_current_epsilon(self, global_counter):

        if hasattr(self, 'decay_steps'):
            return max(self.final_epsilon,
                       self.initial_epsilon - (
                               self.initial_epsilon - self.final_epsilon) * global_counter / self.decay_steps)
        else:
            return None

    def select_action(self, system_state, global_counter, session):
        prob_random = self.get_current_epsilon(global_counter)

        q_values_from_pred = session.run(self.pred_q_values, feed_dict={self.pred_states: [system_state]})

        # TODO Also check the need of this
        if (prob_random is not None and np.random.random() < prob_random) or (
                hasattr(self, 'counter_for_learning') and global_counter < self.counter_for_learning):
            self.logger.debug(self.name + "-system state: %s prob_random: %.2f", str(system_state), prob_random)

            action = np.random.randint(len(self.actions))
            self.logger.debug(self.name + "-Behaving randomly: %s", str(action))
            return action
        else:
            action = np.argmax(q_values_from_pred)
            self.logger.debug(self.name + "-Behaving greedy: %s q_values_from_pred: %s", str(action),
                              str(q_values_from_pred))
            return action

    def build_training_operation(self, learning_rate):
        train_target_q = tf.placeholder(tf.float32, [None], name="target_q_values")
        train_actions = tf.placeholder(tf.int64, [None], name="actions")

        actions_one_hot = tf.one_hot(train_actions, len(self.actions), 1.0, 0.0, name="actions_one_hot")
        action_q_values = tf.reduce_sum(self.pred_q_values * actions_one_hot, axis=1, name="action_q_values")

        delta = tf.square(train_target_q - action_q_values)
        loss = tf.reduce_mean(delta, name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        return train_target_q, train_actions, loss, optimizer.minimize(loss)

    def calculate_transition_targets(self, session, reward_list, next_state_list):
        next_q_values = session.run(self.target_q_values, feed_dict={self.target_states: next_state_list})
        max_next_q_value = np.max(next_q_values, axis=1)
        target_q_values = reward_list + self.discount_factor * max_next_q_value

        return target_q_values

    def train_network(self, session, target_q_values, action_list, state_list):

        _, q_values, loss = session.run([self.train_operation, self.pred_q_values, self.train_loss], feed_dict={
            self.train_target_q: target_q_values,
            self.train_actions: action_list,
            self.pred_states: state_list})

        return q_values

    def update_target_weights(self, session):
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_scope)
        pred_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.prediction_scope)

        for target_variable, pred_variable in zip(target_vars, pred_vars):
            weight_from_pred = tf.placeholder(tf.float32, name="weight_from_pred")
            session.run(target_variable.assign(weight_from_pred),
                        feed_dict={weight_from_pred: pred_variable.eval()})
