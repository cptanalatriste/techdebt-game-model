import tensorflow as tf
import os
from tqdm import tqdm


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

    def start(self, simulation_environment, agent_wrappers, enable_restore):

        with tf.Session() as session:

            initializer = tf.global_variables_initializer()
            saver = tf.train.Saver()

            if os.path.isfile(self.checkpoint_path + ".index") and enable_restore:
                saver.restore(session, self.checkpoint_path)
            else:
                session.run(initializer)

            global_counter = 0

            for episode_index in tqdm(range(1, self.total_episodes + 1)):

                simulation_environment.reset()
                for wrapper in agent_wrappers:
                    wrapper.reset()

                previous_state = simulation_environment.get_system_state()

                for time_step in range(simulation_environment.time_units):
                    self.logger.debug("Episode: %s Time step: %s  Global counter: %s", str(episode_index),
                                      str(time_step), str(global_counter))

                    actions_performed, new_state = simulation_environment.step(agent_wrappers, time_step,
                                                                               global_counter,
                                                                               session)

                    for agent_wrapper in agent_wrappers:
                        if agent_wrapper.name in actions_performed:
                            reward = agent_wrapper.get_reward()
                            action_performed = actions_performed[agent_wrapper.name]
                            agent_wrapper.agent.observe_action_effects(previous_state, action_performed, reward,
                                                                       new_state)
                    previous_state = new_state
                    global_counter += 1

                    if global_counter > self.counter_for_learning:
                        if global_counter % self.train_frequency == 0:

                            for agent_wrapper in agent_wrappers:
                                self.logger.debug("Attempting training on agent " + agent_wrapper.name)

                                ql_agent = agent_wrapper.agent

                                state_list, action_list, reward_list, next_state_list = ql_agent.sample_transitions(
                                    self.batch_size)

                                self.logger.debug(agent_wrapper.name + "-Starting transition target calculations...")
                                target_q_values = ql_agent.calculate_transition_targets(session=session,
                                                                                        reward_list=reward_list,
                                                                                        next_state_list=next_state_list)

                                self.logger.debug(agent_wrapper.name + "-Starting training ...")
                                _ = ql_agent.train_network(session=session,
                                                           target_q_values=target_q_values,
                                                           action_list=action_list, state_list=state_list)

                                if global_counter % self.transfer_frequency:
                                    ql_agent.update_target_weights(session)

                        if global_counter % self.save_frequency:
                            saver.save(session, self.checkpoint_path)

                self.logger.debug("Episode %s finished: ", str(episode_index))
                for wrapper in agent_wrappers:
                    wrapper.agent.store_experience()
                    wrapper.log_progress(episode_index, global_counter)
