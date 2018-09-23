import tensorflow as tf
import os


class DeepQLearning(object):

    def __init__(self, logger, total_training_steps, decay_steps, train_frequency, batch_size, counter_for_learning,
                 transfer_frequency, save_frequency, checkpoint_path):
        self.total_training_steps = total_training_steps
        self.decay_steps = decay_steps

        self.train_frequency = train_frequency
        self.batch_size = batch_size

        self.counter_for_learning = counter_for_learning
        self.transfer_frequency = transfer_frequency
        self.save_frequency = save_frequency

        self.checkpoint_path = checkpoint_path
        self.logger = logger

        self.scope = 'train'
        self.training_step_var = self.get_global_step_variable()

    def get_global_step_variable(self):
        with tf.variable_scope(self.scope):
            global_step_variable = tf.Variable(0, trainable=False, name='training_step')

        return global_step_variable

    def train_agents(self, agent_wrappers, training_step, session):
        for agent_wrapper in agent_wrappers:
            self.logger.debug("Attempting training on agent " + agent_wrapper.name)

            ql_agent = agent_wrapper.agent

            ql_agent.train(session=session, batch_size=self.batch_size)

            if training_step % self.transfer_frequency:
                ql_agent.update_target_weights(session)

    def start(self, simulation_environment, agent_wrappers, enable_restore):

        with tf.Session() as session:
            initializer = tf.global_variables_initializer()
            saver = tf.train.Saver()

            if os.path.isfile(self.checkpoint_path + ".index") and enable_restore:
                saver.restore(session, self.checkpoint_path)
            else:
                session.run(initializer)

            episode_finished = False
            global_counter = 0

            simulation_environment.reset(agent_wrappers)

            while True:
                training_step = self.training_step_var.eval()
                if global_counter >= self.total_training_steps:
                    break

                global_counter += 1
                if episode_finished:
                    self.logger.debug("Episode finished!")
                    for wrapper in agent_wrappers:
                        wrapper.agent.store_experience()
                        wrapper.log_progress(global_counter=global_counter, training_step=training_step)

                    simulation_environment.reset(agent_wrappers)

                previous_state = simulation_environment.get_system_state()

                self.logger.debug("Training step: %s  Global counter: %s",
                                  str(training_step), str(global_counter))

                actions_performed, new_state, episode_finished, rewards = simulation_environment.step(
                    developers=agent_wrappers,
                    global_counter=global_counter,
                    session=session)

                self.logger.debug("actions_performed %s new_state %s episode_finished %s", actions_performed, new_state,
                                  episode_finished)

                for agent_wrapper in agent_wrappers:
                    action_performed = actions_performed[agent_wrapper.name]
                    reward = rewards[agent_wrapper.name]
                    agent_wrapper.agent.observe_action_effects(previous_state, action_performed, reward,
                                                               new_state)

                if global_counter > self.counter_for_learning and global_counter % self.train_frequency == 0:
                    self.train_agents(agent_wrappers=agent_wrappers, training_step=training_step, session=session)

                if training_step % self.save_frequency:
                    saver.save(session, self.checkpoint_path)
