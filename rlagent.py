import tensorflow as tf
import simmodel
import numpy as np


class DeveloperAgent(object):

    def __init__(self, session, initial_epsilon=1.0, final_epsilon=0.1):
        self.session = session
        self.actions = ["CLEAN", "SLOPPY"]

        self.q_values_operation = None

    def select_action(self, system_state, epsilon_decrease, initial_epsilon=1.0, final_epsilon=0.1):
        action_distribution = self.session.run(self.q_values_operation, feed_dict={"states": [system_state]})[0]

        prob_random = initial_epsilon * (1 - epsilon_decrease) + final_epsilon * epsilon_decrease

        if np.random.random() < prob_random:
            return np.argmax(np.random.random(action_distribution.shape))
        else:
            return np.argmax(action_distribution)


def main():
    total_episodes = 10
    counter_for_learning = 60

    with tf.Session() as session:
        developer_agent = DeveloperAgent(session=session)
        session.run(tf.global_variables_initializer())

        global_counter = 0

        for episode_index in range(total_episodes):
            simulation_environment = simmodel.SimulationEnvironment(time_units=60, avg_resolution_time=1 / 5.0,
                                                                    prob_new_issue=0.2, prob_rework=0.05)

            # TODO Verifify these behaves as expected
            epsilon_decrease = 1 / float(simulation_environment.time_units)

            system_state = simulation_environment.get_system_state(developer_agent)
            for time_step in range(simulation_environment.time_units):
                action = developer_agent.select_action(system_state, epsilon_decrease)

                # TODO Also check the need of this
                if global_counter < counter_for_learning:
                    action = np.argmax(np.random.random((len(developer_agent.actions))))


if __name__ == "__main__":
    main()
