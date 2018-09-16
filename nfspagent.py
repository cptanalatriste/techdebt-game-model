import tensorflow as tf
import rlagent
import numpy as np
import random


class ExperienceReservoir(object):

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.states = []
        self.actions = []

    def size(self):
        return len(self.states)

    def observe_action_effects(self, state, action):
        self.states.append(state)
        self.actions.append(action)

        while self.size() > self.memory_size:
            self.states.pop(0)
            self.actions.pop(0)

    def sample_behaviour(self, batch_size):
        states = []
        actions = []

        for index, pair in enumerate(zip(self.states, self.actions)):
            state, action = pair

            if index < batch_size:
                states.append(state)
                actions.append(action)
            else:
                selected_index = random.randin(0, index)
                if selected_index < batch_size:
                    states[selected_index] = state
                    actions[selected_index] = action

        return np.array(states), np.array((actions))


class NFSPAgent(object):

    def __init__(self, name, logger, anticipatory_parameter, input_number, actions, hidden_units, rl_learning_rate,
                 sl_learning_rate, global_step, replay_memory_size, sl_memory_size):
        self.logger = logger
        self.name = name
        self.anticipatory_parameter = anticipatory_parameter
        self.reservoir_memory = ExperienceReservoir(memory_size=sl_memory_size)

        self.followed_best_response = None
        self.actions = actions

        self.dq_learner = rlagent.DeepQLearner(logger=logger, input_number=input_number, hidden_units=hidden_units,
                                               name="RL_" + name, learning_rate=rl_learning_rate,
                                               global_step=global_step, replay_memory_size=replay_memory_size)

        self.states, self.logits, self.probabilities = self.build_network(input_number=input_number,
                                                                          hidden_units=hidden_units,
                                                                          variable_scope="SL_" + self.name + "-network")
        self.build_training_operation(variable_scope="SL_" + self.name + "-train", learning_rate=sl_learning_rate,
                                      global_step=global_step)

    def sample_behaviour(self, batch_size):
        return self.reservoir_memory.sample_behaviour(batch_size)

    def build_network(self, input_number, hidden_units, variable_scope):

        with tf.variable_scope(variable_scope):
            states = tf.placeholder(tf.float32, shape=[None, input_number], name="state")

            hidden_layer = tf.layers.dense(states, hidden_units, activation=tf.nn.relu)
            logits = tf.layers.dense(hidden_layer, len(self.actions), name="outputs")
            probabilities = tf.nn.softmax(logits)

        return states, logits, probabilities

    def build_training_operation(self, variable_scope, global_step, learning_rate):
        with tf.variable_scope(variable_scope):
            train_labels = tf.placeholder(tf.float32, [None], name="train_labels")

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels, logits=self.logits)
            loss = tf.reduce_mean(cross_entropy, name="loss")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        return optimizer.minimize(loss, global_step=global_step)

    def select_action(self, system_state, global_counter, session):
        if self.anticipatory_parameter >= np.random.random():
            action = self.dq_learner.select_action(system_state=system_state, global_counter=global_counter,
                                                   session=session)
            self.followed_best_response = True
        else:
            action = self.select_action_from_average_strategy(system_state=system_state, session=session)
            self.followed_best_response = False
        return action

    def select_action_from_average_strategy(self, system_state, session):
        action_probabilities = session.run(self.probabilities, feed_dict={self.states: [system_state]})
        action = np.random.choice(np.arange(len(self.actions)), p=action_probabilities)
        return action

    def observe_action_effects(self, state, action, reward, new_state):
        self.dq_learner.observe_action_effects(state=state, action=action, reward=reward, new_state=new_state)

        if self.followed_best_response:
            self.reservoir_memory.observe_action_effects(state=state, action=action)


def main():
    print("Start!")


if __name__ == "__main__":
    main()
