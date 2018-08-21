import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLEAN_ACTION = 0
SLOPPY_ACTION = 1


class StubbornAgent(object):

    def __init__(self, only_action):
        self.only_action = only_action

    def select_action(self, system_state, epsilon_decrease, global_counter):
        return self.only_action


class DevelopmentIssue(object):

    def __init__(self, avg_resolution_time, prob_rework):
        self.avg_resolution_time = avg_resolution_time
        self.prob_rework = prob_rework


class Developer(object):

    def __init__(self, agent):
        self.current_issue = None
        self.issues_delivered = 0
        self.agent = agent

    def start_coding(self, simulation_environment, global_counter):
        system_state = simulation_environment.get_system_state()

        action = self.agent.select_action(system_state=system_state,
                                          global_counter=global_counter)

        self.carry_out_action(action, simulation_environment)
        return action

    def carry_out_action(self, action, simulation_environment):
        if CLEAN_ACTION == action:
            self.code_clean(simulation_environment)
        elif SLOPPY_ACTION == action:
            self.code_sloppy(simulation_environment)
        else:
            raise Exception("The action " + str(action) + " is not supported.")

    def code_clean(self, simulation_environment):
        self.current_issue = DevelopmentIssue(avg_resolution_time=simulation_environment.avg_resolution_time,
                                              prob_rework=simulation_environment.prob_rework * 0.9)

    def code_sloppy(self, simulation_config):
        self.current_issue = DevelopmentIssue(avg_resolution_time=simulation_config.avg_resolution_time * 0.75,
                                              prob_rework=simulation_config.prob_rework * 1.1)

    def notify(self, current_state, action_performed, next_state):
        self.dqn_agent.store_transition(current_state, action_performed, next_state)


class SimulationEnvironment(object):

    def __init__(self, time_units, avg_resolution_time, prob_new_issue, prob_rework, logger):
        self.time_units = time_units
        self.logger = logger
        self.avg_resolution_time = avg_resolution_time
        self.prob_new_issue = prob_new_issue
        self.prob_rework = prob_rework

        self.pending_issues = 0
        self.current_time = 0

    def register_new_issue(self):
        self.pending_issues += 1

    def remove_issue(self, developer):
        self.pending_issues -= 1
        developer.issues_delivered += 1

    def get_system_state(self):
        return self.current_time, self.pending_issues

    def step(self, developer, time_step, global_counter):
        self.current_time = time_step
        action_performed = None

        if self.pending_issues > 0:

            if not developer.current_issue:
                action_performed = developer.start_coding(self, global_counter)
            else:

                random_output = np.random.random()

                if random_output < developer.current_issue.avg_resolution_time:
                    self.prob_rework = developer.current_issue.prob_rework

                    random_output = np.random.random()

                    if random_output >= developer.current_issue.prob_rework:
                        developer.current_issue = None
                        self.remove_issue(developer)

        if np.random.random() < self.prob_new_issue:
            self.register_new_issue()

        return action_performed, self.get_system_state()


def run_simulation():
    pending_issues = []
    simulation_env = SimulationEnvironment(time_units=60, avg_resolution_time=1 / 5.0,
                                           prob_new_issue=0.1, prob_rework=0.05)

    developer = Developer(agent=StubbornAgent(only_action="SLOPPY"))

    for time_step in range(simulation_env.time_units):
        simulation_env.step(developer, time_step=time_step, global_counter=time_step)
        pending_issues.append(simulation_env.pending_issues)

    return pd.Series(pending_issues)


if __name__ == "__main__":
    pending_issues = run_simulation()
    print("Mean issues in the system: ", pending_issues.mean())

    plt.plot(pending_issues.index, pending_issues.values)
    plt.show()
