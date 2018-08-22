import tensorflow as tf
import simmodel
import numpy as np


class DeepQLearner(object):

    def __init__(self, learning_rate, discount_factor, input_number, hidden_units, counter_for_learning,
                 logger, initial_epsilon, final_epsilon, decay_steps):
        self.logger = logger
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps

        self.learning_rate = learning_rate
        self.actions = [simmodel.CLEAN_ACTION, simmodel.SLOPPY_ACTION]
        self.discount_factor = discount_factor
        self.counter_for_learning = counter_for_learning

        self.target_scope = 'target_network'
        self.prediction_scope = 'prediction_network'
        self.pred_states, self.pred_q_values = self.build_network(self.target_scope, input_number, hidden_units)
        self.target_states, self.target_q_values = self.build_network(self.prediction_scope, input_number, hidden_units)
        self.train_target_q, self.train_actions, self.train_loss, self.train_operation = self.build_training_operation()

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
        return max(self.final_epsilon,
                   self.initial_epsilon - (
                           self.initial_epsilon - self.final_epsilon) * global_counter / self.decay_steps)

    def select_action(self, system_state, global_counter, session):
        prob_random = self.get_current_epsilon(global_counter)
        self.logger.debug("system state: %s prob_random: %.2f", str(system_state), prob_random)

        q_values_from_pred = session.run(self.pred_q_values, feed_dict={self.pred_states: [system_state]})

        # TODO Also check the need of this
        if np.random.random() < prob_random or global_counter < self.counter_for_learning:
            action = np.random.randint(len(self.actions))
            self.logger.debug("Behaving randomly: %s", str(action))
            return action
        else:
            action = np.argmax(q_values_from_pred)
            self.logger.debug("Behaving greedy: %s q_values_from_pred: %s", str(action), str(q_values_from_pred))
            return action

    def build_training_operation(self):
        train_target_q = tf.placeholder(tf.float32, [None], name="target_q_values")
        train_actions = tf.placeholder(tf.int64, [None], name="actions")

        actions_one_hot = tf.one_hot(train_actions, len(self.actions), 1.0, 0.0, name="actions_one_hot")
        action_q_values = tf.reduce_sum(self.pred_q_values * actions_one_hot, axis=1, name="action_q_values")

        delta = tf.square(train_target_q - action_q_values)
        loss = tf.reduce_mean(delta, name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

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
