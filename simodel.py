import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DevelopmentIssue(object):

    def __init__(self, avg_resolution_time, prob_rework):
        self.avg_resolution_time = avg_resolution_time
        self.prob_rework = prob_rework


class Developer(object):

    def __init__(self):
        self.current_issue = None
        self.coding_clean = None

    def code_clean(self, simulation_config):
        self.current_issue = DevelopmentIssue(avg_resolution_time=simulation_config.avg_resolution_time,
                                              prob_rework=simulation_config.prob_rework * 0.9)

    def code_sloppy(self, simulation_config):
        self.current_issue = DevelopmentIssue(avg_resolution_time=simulation_config.avg_resolution_time * 0.75,
                                              prob_rework=simulation_config.prob_rework * 1.1)


class SimulationConfig(object):

    def __init__(self, time_units, avg_resolution_time, prob_new_issue, prob_rework):
        self.time_units = time_units
        self.avg_resolution_time = avg_resolution_time
        self.prob_new_issue = prob_new_issue
        self.prob_rework = prob_rework

        self.pending_issues = 0

    def register_new_issue(self):
        self.pending_issues += 1

    def remove_issue(self):
        self.pending_issues -= 1


def update_state(simulation_config, developer):
    if simulation_config.pending_issues > 0:

        if not developer.current_issue:
            developer.code_sloppy(simulation_config)
        else:
            if np.random.random() < developer.current_issue.avg_resolution_time:
                simulation_config.prob_rework = developer.current_issue.prob_rework

                if np.random.random() < developer.current_issue.prob_rework:
                    developer.code_clean(simulation_config)
                else:
                    developer.current_issue = None
                    simulation_config.remove_issue()

    if np.random.random() < simulation_config.prob_new_issue:
        simulation_config.register_new_issue()


def run_simulation():
    pending_issues = []
    simulation_config = SimulationConfig(time_units=60, avg_resolution_time=1 / 5.0,
                                         prob_new_issue=0.1, prob_rework=0.05)

    developer = Developer()

    for time_step in range(simulation_config.time_units):
        update_state(simulation_config, developer)
        pending_issues.append(simulation_config.pending_issues)

    return pd.Series(pending_issues)


if __name__ == "__main__":
    pending_issues = run_simulation()
    print(pending_issues)

    plt.plot(pending_issues.index, pending_issues.values)
    plt.show()
