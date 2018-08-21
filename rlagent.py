import tensorflow as tf
import simmodel
import numpy as np
import logging
from tqdm import tqdm


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


# TODO: All this class can be moved to the agent
class ExperienceReplayMemory(object):

    def __init__(self, table_size=100):
        self.table_size = table_size
        self.consolidated_experience = EpisodeExperience()

    def store_experience(self, episode_history):
        self.consolidated_experience.consolidate(episode_history)

        while self.consolidated_experience.size() > self.table_size:
            self.consolidated_experience.purge()

    def sample_transitions(self, batch_size):
        batch = EpisodeExperience()

        selected_indexes = np.arange(self.consolidated_experience.size())
        np.random.shuffle(selected_indexes)
        selected_indexes = selected_indexes[:batch_size]

        for index in selected_indexes:
            batch.observe_action_effects(state=self.consolidated_experience.states[index],
                                         action=self.consolidated_experience.actions[index],
                                         reward=self.consolidated_experience.rewards[index],
                                         new_state=self.consolidated_experience.new_states[index])

        return np.array(batch.states), np.array(batch.actions), np.array(batch.rewards), np.array(batch.new_states)


class DeveloperAgent(object):

    def __init__(self, session, learning_rate, discount_factor, input_number, hidden_units, counter_for_learning,
                 logger):
        self.session = session
        self.logger = logger

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
            hidden_layer = tf.layers.dense(states, hidden_units, activation=tf.nn.elu, kernel_initializer=initializer,
                                           name="hidden")
            outputs = tf.layers.dense(hidden_layer, len(self.actions), kernel_initializer=initializer, name="q_values")

        return states, outputs

    def select_action(self, system_state, global_counter, initial_epsilon=1.0, final_epsilon=0.1, decay_steps=10000):
        prob_random = max(final_epsilon,
                          initial_epsilon - (initial_epsilon - final_epsilon) * global_counter / decay_steps)

        logger.debug("prob_random: %.2f", prob_random)

        q_values_from_pred = self.session.run(self.pred_q_values, feed_dict={self.pred_states: [system_state]})

        # TODO Also check the need of this
        if np.random.random() < prob_random or global_counter < self.counter_for_learning:
            action = np.random.randint(len(self.actions))
            logger.debug("Behaving randomly: %s", str(action))
            return action
        else:
            action = np.argmax(q_values_from_pred)
            logger.debug("Behaving greedy: %s", str(action))
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

    def calculate_transition_targets(self, reward_list, next_state_list):
        next_q_values = self.session.run(self.target_q_values, feed_dict={self.target_states: next_state_list})
        max_next_q_value = np.max(next_q_values, axis=1)
        target_q_values = reward_list + self.discount_factor * max_next_q_value

        return target_q_values

    def train_network(self, target_q_values, action_list, state_list):

        logger.debug("target_q_values: %s action_list: %s state_list: %s ", str(target_q_values), str(action_list),
                     str(state_list))

        _, q_values, loss = self.session.run([self.train_operation, self.pred_q_values, self.train_loss], feed_dict={
            self.train_target_q: target_q_values,
            self.train_actions: action_list,
            self.pred_states: state_list})

        return q_values

    def update_target_weights(self):
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_scope)
        pred_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.prediction_scope)

        for target_variable, pred_variable in zip(target_vars, pred_vars):
            weight_from_pred = tf.placeholder(tf.float32, name="weight_from_pred")
            self.session.run(target_variable.assign(weight_from_pred),
                             feed_dict={weight_from_pred: pred_variable.eval()})


def main(logger):
    total_episodes = 1000
    train_frequency = 4
    transfer_frequency = 10 * 60
    batch_size = 32
    logging_frequency = 50

    discount_factor = 0.99
    learning_rate = 1e-4
    input_number = 2
    hidden_units = 2

    time_units = 60
    counter_for_learning = total_episodes * time_units * 0.1
    avg_resolution_time = 1 / 3.0
    prob_new_issue = 0.9
    prob_rework = 0.05

    with tf.Session() as session:
        developer_agent = DeveloperAgent(session=session, learning_rate=learning_rate,
                                         discount_factor=discount_factor, counter_for_learning=counter_for_learning,
                                         input_number=input_number, hidden_units=hidden_units, logger=logger)
        logger.debug("RL agent initialized")

        session.run(tf.global_variables_initializer())

        global_counter = 0
        replay_memory = ExperienceReplayMemory()
        q_values_list = []
        episode_reward_list = []

        for episode_index in tqdm(range(1, total_episodes + 1)):
            simulation_environment = simmodel.SimulationEnvironment(logger=logger,
                                                                    time_units=time_units,
                                                                    avg_resolution_time=avg_resolution_time,
                                                                    prob_new_issue=prob_new_issue,
                                                                    prob_rework=prob_rework)
            developer = simmodel.Developer(agent=developer_agent)
            episode_experience = EpisodeExperience()
            episode_reward = 0.0
            previous_state = simulation_environment.get_system_state()

            for time_step in range(simulation_environment.time_units):
                action_performed, new_state = simulation_environment.step(developer, time_step, global_counter)
                reward = developer.issues_delivered

                if action_performed is not None:
                    episode_experience.observe_action_effects(previous_state, action_performed, reward, new_state)
                    previous_state = new_state
                    episode_reward += reward
                    global_counter += 1

                if global_counter > counter_for_learning:
                    if global_counter % train_frequency == 0:
                        state_list, action_list, reward_list, next_state_list = replay_memory.sample_transitions(
                            batch_size)

                        logger.debug("Starting transition target calculations...")
                        target_q_values = developer_agent.calculate_transition_targets(reward_list, next_state_list)

                        logger.debug("Starting training ...")
                        q_values = developer_agent.train_network(target_q_values, action_list, state_list)

                        q_values_list.append(q_values)

                    if global_counter % transfer_frequency:
                        developer_agent.update_target_weights()

            replay_memory.store_experience(episode_experience)
            episode_reward_list.append(episode_reward)

            logger.debug("EPISODE %s: Developer fixes: %.2f Final pending issues: %.2f", str(episode_index),
                         developer.issues_delivered, simulation_environment.pending_issues)
            log_progress(logging_frequency, episode_index, episode_reward_list, q_values_list)


def log_progress(logging_frequency, episode_index, episode_reward_list, q_values_list):
    if logging_frequency % episode_index == 0:
        last_100_rewards = episode_reward_list[-100:]
        logger.info("EPISODE %s Reward stats (min, max, median, mean): %.2f %.2f %.2f %.2f", str(episode_index),
                    np.min(last_100_rewards),
                    np.max(last_100_rewards),
                    np.median(last_100_rewards), np.mean(last_100_rewards))

        if q_values_list:
            last_100_qvalues = q_values_list[-100:]
            logger.info("EPISODE %s Q value stats (min, max, median, mean): %.2f %.2f %.2f %.2f",
                        str(episode_index), np.min(last_100_qvalues),
                        np.max(last_100_qvalues),
                        np.median(last_100_qvalues), np.mean(last_100_qvalues))


if __name__ == "__main__":
    logging_level = logging.INFO
    # logging_level = logging.DEBUG

    logging.basicConfig(level=logging_level, filename='tech_debt_rl.log', filemode='w')
    logger = logging.getLogger("DQNetwork-Training->")
    logger.debug("Starting script")
    main(logger)
