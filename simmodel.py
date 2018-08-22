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

    def __init__(self, avg_resolution_time, prob_rework, code_impact):
        self.avg_resolution_time = avg_resolution_time
        self.prob_rework = prob_rework
        self.code_impact = code_impact


class CodingApproach(object):

    def __init__(self, resolution_factor, rework_factor, code_impact):
        self.resolution_factor = resolution_factor
        self.rework_factor = rework_factor
        self.code_impact = code_impact

    def get_development_issue(self, simulation_environment):
        avg_resolution_time = min(1.0, simulation_environment.avg_resolution_time * self.resolution_factor)
        prob_rework = min(1.0, simulation_environment.prob_rework * self.rework_factor)

        return DevelopmentIssue(
            avg_resolution_time=avg_resolution_time,
            prob_rework=prob_rework,
            code_impact=self.code_impact)


class Developer(object):

    def __init__(self, agent, approach_map):
        self.current_issue = None
        self.approach_map = approach_map
        self.agent = agent

        self.issues_delivered = None
        self.sloppy_counter = None
        self.attempted_deliveries = None

        self.reset()

    def get_reward(self):
        return self.issues_delivered

    def reset(self):
        self.current_issue = None
        self.issues_delivered = 0
        self.sloppy_counter = 0
        self.attempted_deliveries = 0

    def start_coding(self, simulation_environment, global_counter, session):
        system_state = simulation_environment.get_system_state()

        action = self.agent.select_action(system_state=system_state,
                                          global_counter=global_counter,
                                          session=session)

        self.carry_out_action(action, simulation_environment)
        return action

    def carry_out_action(self, action, simulation_environment):

        coding_approach = self.approach_map[action]

        if coding_approach is not None:
            # TODO Hacky but temporary solution
            if action == SLOPPY_ACTION:
                self.sloppy_counter += 1
            self.current_issue = coding_approach.get_development_issue(simulation_environment)
        else:
            raise Exception("The action " + str(action) + " is not supported.")

    def log_progress(self, logger, episode_index, global_counter):

        logger.info(
            "EPISODE %s: Developer fixes: %.2f Sloppy commits: %.2f Attempted Deliveries: %.2f  "
            "Sloppy ratio: %.2f Next epsilon: %.2f ",
            str(episode_index),
            self.issues_delivered,
            self.sloppy_counter,
            self.attempted_deliveries,
            float(self.sloppy_counter) / max(self.attempted_deliveries, 1),
            self.agent.get_current_epsilon(global_counter))


class SimulationEnvironment(object):

    def __init__(self, time_units, avg_resolution_time, prob_new_issue, prob_rework, logger):
        self.time_units = time_units
        self.logger = logger

        self.init_avg_resolution_time = avg_resolution_time
        self.init_prob_new_issue = prob_new_issue
        self.init_prob_rework = prob_rework

        self.avg_resolution_time = None
        self.prob_new_issue = None
        self.prob_rework = None
        self.to_do_issues = None
        self.doing_issues = None
        self.done_issues = None
        self.current_time = None

        self.reset()

    def reset(self):
        self.avg_resolution_time = self.init_avg_resolution_time
        self.prob_new_issue = self.init_prob_new_issue
        self.prob_rework = self.init_prob_rework

        self.to_do_issues = 0
        self.doing_issues = 0
        self.done_issues = 0

        self.current_time = 0

    def add_to_backlog(self):
        self.to_do_issues += 1

    def move_to_in_progress(self, developer, global_counter, session):
        action_performed = developer.start_coding(self, global_counter, session)
        self.to_do_issues -= 1
        self.doing_issues += 1

        return action_performed

    def move_to_done(self, developer):
        self.doing_issues -= 1
        self.done_issues += 1

        developer.issues_delivered += 1
        developer.current_issue = None

    def code_submitted(self, developer):
        self.avg_resolution_time = min(1.0, self.avg_resolution_time * developer.current_issue.code_impact)
        developer.attempted_deliveries += 1

    def get_system_state(self):
        return self.time_units - self.current_time, self.to_do_issues, self.doing_issues, self.done_issues

    def step(self, developer, time_step, global_counter, session):
        self.current_time = time_step
        action_performed = None

        if self.to_do_issues > 0:

            if developer.current_issue is None:
                action_performed = self.move_to_in_progress(developer, global_counter, session)
            else:

                random_output = np.random.random()
                if random_output < developer.current_issue.avg_resolution_time:
                    # Deliver issue, but verify rework first
                    self.code_submitted(developer)

                    random_output = np.random.random()
                    if random_output >= developer.current_issue.prob_rework:
                        # No rework needed
                        self.move_to_done(developer)

        if np.random.random() < self.prob_new_issue:
            self.add_to_backlog()

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
