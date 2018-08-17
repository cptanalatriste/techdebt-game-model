import tensorflow as tf
import simmodel
import numpy as np


class DeveloperAgent(object):

    def __init__(self, session, counter_for_learning):
        self.session = session
        self.counter_for_learning = counter_for_learning
        self.actions = [simmodel.CLEAN_ACTION, simmodel.SLOPPY_ACTION]

        self.q_values_operation = None

    def select_action(self, system_state, epsilon_decrease, global_counter, initial_epsilon=1.0, final_epsilon=0.1):
        action_distribution = self.session.run(self.q_values_operation, feed_dict={"states": [system_state]})[0]

        prob_random = initial_epsilon * (1 - epsilon_decrease) + final_epsilon * epsilon_decrease

        # TODO Also check the need of this
        if np.random.random() < prob_random or global_counter < self.counter_for_learning:
            return np.argmax(np.random.random(action_distribution.shape))
        else:
            return np.argmax(action_distribution)


def main():
    total_episodes = 10
    counter_for_learning = 60

    with tf.Session() as session:
        developer_agent = DeveloperAgent(session=session, counter_for_learning=counter_for_learning)
        developer = simmodel.Developer(developer_agent=developer_agent)

        session.run(tf.global_variables_initializer())

        global_counter = 0

        for episode_index in range(total_episodes):
            simulation_environment = simmodel.SimulationEnvironment(time_units=60, avg_resolution_time=1 / 5.0,
                                                                    prob_new_issue=0.2, prob_rework=0.05)

            for time_step in range(simulation_environment.time_units):
                simulation_environment.step(developer, time_step, global_counter)


if __name__ == "__main__":
    main()
