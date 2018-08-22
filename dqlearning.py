import tensorflow as tf
import os
import numpy as np
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


class DeepQLearning(object):

    def __init__(self, logger, total_episodes, decay_steps, train_frequency, batch_size, counter_for_learning,
                 transfer_frequency, save_frequency, checkpoint_path):
        self.total_episodes = total_episodes
        self.decay_steps = decay_steps

        self.train_frequency = train_frequency
        self.batch_size = batch_size

        self.counter_for_learning = counter_for_learning
        self.transfer_frequency = transfer_frequency
        self.save_frequency = save_frequency

        self.checkpoint_path = checkpoint_path
        self.logger = logger

    def start(self, simulation_environment, agent_wrapper, enable_restore):

        with tf.Session() as session:

            ql_agent = agent_wrapper.agent

            self.logger.debug("RL agent initialized")

            initializer = tf.global_variables_initializer()
            saver = tf.train.Saver()

            if os.path.isfile(self.checkpoint_path + ".index") and enable_restore:
                saver.restore(session, self.checkpoint_path)
            else:
                session.run(initializer)

            global_counter = 0
            replay_memory = ExperienceReplayMemory()

            for episode_index in tqdm(range(1, self.total_episodes + 1)):

                simulation_environment.reset()
                agent_wrapper.reset()

                episode_experience = EpisodeExperience()
                episode_reward = 0.0
                previous_state = simulation_environment.get_system_state()

                for time_step in range(simulation_environment.time_units):
                    self.logger.debug("Global counter: %s", str(global_counter))

                    action_performed, new_state = simulation_environment.step(agent_wrapper, time_step, global_counter,
                                                                              session)
                    reward = agent_wrapper.get_reward()

                    if action_performed is not None:
                        episode_experience.observe_action_effects(previous_state, action_performed, reward, new_state)
                        previous_state = new_state
                        episode_reward += reward

                    global_counter += 1

                    if global_counter > self.counter_for_learning:
                        if global_counter % self.train_frequency == 0:
                            state_list, action_list, reward_list, next_state_list = replay_memory.sample_transitions(
                                self.batch_size)

                            self.logger.debug("Starting transition target calculations...")
                            target_q_values = ql_agent.calculate_transition_targets(session=session,
                                                                                    reward_list=reward_list,
                                                                                    next_state_list=next_state_list)

                            self.logger.debug("Starting training ...")
                            _ = ql_agent.train_network(session=session, target_q_values=target_q_values,
                                                       action_list=action_list, state_list=state_list)

                        if global_counter % self.transfer_frequency:
                            ql_agent.update_target_weights(session)

                        if global_counter % self.save_frequency:
                            saver.save(session, self.checkpoint_path)

                replay_memory.store_experience(episode_experience)
                agent_wrapper.log_progress(self.logger, episode_index, global_counter)
