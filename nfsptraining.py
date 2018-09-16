import tensorflow as tf


class NFSPTrainer(object):

    def __init__(self, logger):
        self.logger = logger

    def start(self, simulation_environment, agent_wrappers):
        with tf.Session() as session:
            initializer = tf.global_variables_initializer()

            session.run(initializer)

        episode_finished = False

        simulation_environment.reset(agent_wrappers)

        while True:
            if episode_finished:
                pass

            previous_state = simulation_environment.get_system_state()

            actions_performed, new_state, episode_finished = simulation_environment.step(developers=agent_wrappers,
                                                                                         session=session)


def main():
    pass


if __name__ == "__main__":
    main()
