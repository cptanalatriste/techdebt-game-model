import enum
import gym
import numpy as np


class Actions(enum.Enum):
    CodeClean = 0
    CodeSloppy = 1


class EnvironmentState:

    def __init__(self, action_resolution_times, action_code_impacts, avg_resolution_time, action_probs_rework,
                 in_progress_reward, fix_ready_reward, prob_new_issue, time_units):
        self.to_do_issues = None
        self.doing_issues = None
        self.done_issues = None
        self.current_time = None
        self.time_units = None

        self.action_resolution_times = action_resolution_times
        self.action_code_impacts = action_code_impacts
        self.action_probs_rework = action_probs_rework

        self.avg_resolution_time = avg_resolution_time
        self.in_progress_reward = in_progress_reward
        self.fix_ready_reward = fix_ready_reward
        self.prob_new_issue = prob_new_issue

    @property
    def shape(self):
        return 2,

    def reset(self):
        self.to_do_issues = 0
        self.doing_issues = 0
        self.done_issues = 0
        self.current_time = 0

    def encode(self):
        return self.time_units - self.current_time, self.to_do_issues, self.doing_issues, self.done_issues

    def move_to_in_progress(self):
        self.to_do_issues -= 1
        self.doing_issues += 1

    def code_submitted(self, action):
        code_impact = self.action_code_impacts[action]
        self.avg_resolution_time = min(1.0, self.avg_resolution_time * code_impact)

    def move_to_done(self):
        self.doing_issues -= 1
        self.done_issues += 1

    def add_to_backlog(self):
        self.to_do_issues += 1

    def step(self, action):
        self.current_time += 1
        self.move_to_in_progress()
        reward = self.in_progress_reward

        avg_resolution_time = self.action_resolution_times[action]

        if np.random.random() < avg_resolution_time:
            self.code_submitted(action)

            action_prob_rework = self.action_probs_rework[action]
            if np.random.random() >= action_prob_rework:
                self.move_to_done()
                reward = self.fix_ready_reward

        if np.random.random() < self.prob_new_issue:
            self.add_to_backlog()

        episode_finished = self.current_time == self.time_units
        return reward, episode_finished


class BPIEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._state = EnvironmentState()
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)

    def reset(self):
        self._state.reset()
        return self._state.encode()

    def step(self, action_index):
        action = Actions(action_index)
        reward, done = self._state.step(action)
        observation = self._state.encode()
        information = {}

        return observation, reward, done, information

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass
