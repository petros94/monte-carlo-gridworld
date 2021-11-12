import random

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
plt.ion()

rotate_cw = np.matrix([[0, -1], [1, 0]])
rotate_ccw = np.matrix([[0, 1], [-1, 0]])

class SimpleGridworldEnv(gym.Env):
    def __init__(self):
        self.height = 3
        self.width = 4

        self.moves = {
            0: (-1, 0),  # up
            1: (0, 1),  # right
            2: (1, 0),  # down
            3: (0, -1),  # left
        }

        # initialize plot
        self.init_plot()

        # begin in start state
        self.reset()

    def step(self, action):
        movement_vector = np.array(self.moves[action])

        # Stochastic environment
        p = random.uniform(0, 1)
        if 0.8 < p <= 0.9:
            movement_vector = np.array(movement_vector*rotate_cw)[0]
        elif 0.9 < p:
            movement_vector = np.array(movement_vector*rotate_ccw)[0]

        s_ = self.s + movement_vector

        # position (1,1) is blocked
        if (s_ != (1, 1)).any():
            self.s = s_

        # stay inside map
        self.s = max(0, self.s[0]), max(0, self.s[1])
        self.s = (min(self.s[0], self.height - 1),
                  min(self.s[1], self.width - 1))

        # check for terminal states
        if self.s == (0, 3):
            return self.s, 1, True, {}
        elif self.s == (1, 3):
            return self.s, -1, True, {}
        return self.s, -0.04, False, {}

    def reset(self):
        self.s = (2, 0)
        self.render()
        return self.s

    def render(self, mode="human"):
        black = [0, 0, 0]
        white = [255, 255, 255]
        red = [255, 0, 0]
        green = [0, 255, 0]
        blue = [0, 0, 255]

        world = [[white, white, white, green],
                 [white, black, white, red],
                 [white, white, white, white]]


        y, x = self.s
        world[y][x] = blue

        self.im.set_data(world)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def init_plot(self):
        black = [0, 0, 0]
        white = [255, 255, 255]
        red = [255, 0, 0]
        green = [0, 255, 0]


        world = np.array([[white, white, white, green],
                 [white, black, white, red],
                 [white, white, white, white]])
        self.fig, self.ax = plt.subplots(1, 1)
        self.im = self.ax.imshow(world)
        plt.show()

