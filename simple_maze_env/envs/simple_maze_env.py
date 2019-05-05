import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class Maze44(gym.Env):

    def __init__(self):
        #print('initing')
        self.end_state = np.array([5, 5])
        self.start_state = np.array([0, 0])
        self.current_state = np.array([0, 0])
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([5, 5]), dtype=np.float32)

    def get_reward(self):
        #print('getting reward')
        if (self.current_state == self.end_state).all():
            return 1
        else:
            return -1

    def get_state(self):
        #print('getting state')
        return self.current_state

    def step(self, action):
        #print('stepping')
        self.take_action(action)
        reward = self.get_reward()
        obs = self.get_state()
        #print('obs', obs)
        episode_over = (self.current_state == self.end_state).all()
        if episode_over:
            print('DONE')
            self.reset()
        return obs, reward, episode_over, {}

    def take_action(self, action):
        #print('taking action')
        if action == 0:
            # up
            self.current_state[1] = np.min([self.current_state[1] + 1, 5])
        if action == 1:
            # down
            self.current_state[1] = np.max([self.current_state[1] - 1, 0])
        if action == 2:
            # left
            self.current_state[0] = np.max([self.current_state[0] - 1, 0])
        if action == 3:
            # right
            self.current_state[0] = np.min([self.current_state[0] + 1, 5])

    def reset(self):
        self.current_state = np.array([0, 0])
        return np.array([0, 0])
