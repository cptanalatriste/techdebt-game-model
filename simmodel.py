import csv
import gymenvironment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

CLEAN_ACTION = gymenvironment.Actions.CodeClean
SLOPPY_ACTION = gymenvironment.Actions.CodeSloppy
IN_PROGRESS_ACTION = 2

FIX_READY_REWARD = +10
IN_PROGRESS_REWARD = -0.1

PENDING_TIME_INDEX = 0
PENDING_ITEMS_INDEX = 1


def last_minute_patcher(agent, system_state):
    action = CLEAN_ACTION
    pending_time = system_state[PENDING_TIME_INDEX]

    if pending_time <= agent.panic_threshold:
        action = SLOPPY_ACTION

    return action


def stressed_patcher(agent, system_state):
    action = CLEAN_ACTION
    pending_items = system_state[PENDING_ITEMS_INDEX]

    if pending_items > agent.panic_threshold:
        action = SLOPPY_ACTION
    return action


class BaseDeveloper(object):

    def __init__(self, logger, name, panic_threshold, action_selector):
        self.name = name
        self.logger = logger
        self.panic_threshold = panic_threshold
        self.metric_catalogue = []

        self.actions = [CLEAN_ACTION, SLOPPY_ACTION]
        self.action_selector = action_selector

    def record_metric(self, metric_value):
        self.metric_catalogue.append(metric_value)

    def clear_metrics(self):
        self.metric_catalogue = []

    def select_action(self, system_state, global_counter=None, session=None):
        return self.action_selector(self, system_state)

    def new_episode(self):
        pass


class PerformanceMetrics:

    def __init__(self, developer):
        self.sloppy_counter = developer.sloppy_counter
        self.action_counter = developer.action_counter
        self.attempted_deliveries = developer.attempted_deliveries
        self.issues_delivered = developer.issues_delivered

    def get_sloppy_ratio(self):
        return float(self.sloppy_counter) / self.action_counter if self.action_counter > 0 else 0.0


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
        self.name = agent.name

        self.issues_delivered = None
        self.sloppy_counter = None
        self.action_counter = None
        self.attempted_deliveries = None

        self.agent.clear_metrics()
        self.reset()

    def get_reward(self):
        return self.issues_delivered

    def reset(self):
        self.current_issue = None
        self.issues_delivered = 0

        self.sloppy_counter = 0
        self.action_counter = 0
        self.attempted_deliveries = 0

        self.agent.new_episode()

    def start_coding(self, simulation_environment, global_counter, session):
        system_state = simulation_environment.get_system_state()

        action = self.agent.select_action(system_state=system_state,
                                          global_counter=global_counter,
                                          session=session)

        self.carry_out_action(action, simulation_environment)
        self.action_counter += 1

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

    def log_progress(self, training_step=None, global_counter=None):

        performance_metrics = PerformanceMetrics(developer=self)
        current_epsilon = None
        if hasattr(self.agent, 'get_current_epsilon'):
            current_epsilon = self.agent.get_current_epsilon(global_counter)

        csv_filename = "training_log_" + self.name + ".csv"

        if not os.path.isfile(csv_filename):
            with open(csv_filename, 'w', newline="") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(
                    ['training_step', 'sloppy_counter', 'action_counter', 'attempted_deliveries', 'issues_delivered',
                     'sloppy_ratio',
                     'current_epsilon'])

        with open(csv_filename, 'a', newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([training_step, self.sloppy_counter, self.action_counter, self.attempted_deliveries,
                                 self.issues_delivered, performance_metrics.get_sloppy_ratio(), current_epsilon])

        self.agent.record_metric(performance_metrics)


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

    def reset(self, agent_wrappers=None):
        self.avg_resolution_time = self.init_avg_resolution_time
        self.prob_new_issue = self.init_prob_new_issue
        self.prob_rework = self.init_prob_rework

        self.to_do_issues = 0
        self.doing_issues = 0
        self.done_issues = 0

        self.current_time = 0

        if agent_wrappers is not None:
            for wrapper in agent_wrappers:
                wrapper.reset()

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

    def step(self, developers, session, global_counter=None):
        self.current_time += 1
        actions_performed = {}
        rewards = {}

        for developer in developers:

            actions_performed[developer.name] = IN_PROGRESS_ACTION
            rewards[developer.name] = IN_PROGRESS_REWARD

            if developer.current_issue is None:
                action_performed = self.move_to_in_progress(developer, global_counter, session)
                actions_performed[developer.name] = action_performed
            else:
                random_output = np.random.random()
                if random_output < developer.current_issue.avg_resolution_time:
                    # Deliver issue, but verify rework first
                    self.code_submitted(developer)

                    random_output = np.random.random()
                    if random_output >= developer.current_issue.prob_rework:
                        # No rework needed
                        self.move_to_done(developer)
                        rewards[developer.name] = FIX_READY_REWARD

        if np.random.random() < self.prob_new_issue:
            self.add_to_backlog()

        episode_finished = self.current_time == self.time_units

        return actions_performed, self.get_system_state(), episode_finished, rewards


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
