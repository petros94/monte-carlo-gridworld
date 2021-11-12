import random

import numpy as np
import pandas as pd

rotate_cw = np.matrix([[0, -1], [1, 0]])
rotate_ccw = np.matrix([[0, 1], [-1, 0]])

def merge_two_dicts(x, y):
    z = x.copy()  # start with keys and values of x
    z.update(y)  # modifies z with keys and values of y
    return z

class MCAgent:
    def __init__(self):
        self.height = 3
        self.width = 4
        self.gamma = 0.9
        self.e = 0.2
        self.visited_states = []
        self.current_episode_rewards = {}
        self.all_episodes_rewards = pd.DataFrame()
        # initial utilities set to 0 for all states
        self.utilities = dict([(str((y, x)), 0) for y in range(3) for x in range(4)])
        self.moves = {
            0: (-1, 0),  # up
            1: (0, 1),  # right
            2: (1, 0),  # down
            3: (0, -1),  # left
        }
        self.policy = dict([(str((y, x)), random.choice(list(self.moves.keys()))) for y in range(3) for x in range(4)])


    def choose_action(self, s):
        return self.policy[str(s)] if random.uniform(0, 1) < self.e else random.choice(list(self.moves.keys()))

    def policy_evaluation(self):
        old_utilities = self.utilities.copy()
        self.utilities = merge_two_dicts(self.utilities, self.all_episodes_rewards.applymap(lambda it: it[0], na_action='ignore').mean().to_dict())
        return np.std(np.array(list(old_utilities.values())) - np.array(list(self.utilities.values())))


    def policy_iteration(self):
        # Recalculate the best action to take for each state,
        # considering the expected utilities of the possible next states that the action will lead me to.
        #
        # This approach is only possible because the dynamics of the environment are known to the agents,
        # otherwise we would have to calculate the values for each pair (s,a).

        for s in self.visited_states:
            policy = [0] * len(self.moves.keys())
            for action in self.moves.keys():
                # possible outcomes (s') from this move
                s_straight, s_cw, s_ccw = self.calculate_outcomes(s, action)

                # a = π(s) = argmax[ Σp(s'|s,a)*U(s') ] - since p is known to the agents we can use this to determine the policy
                policy[action] = 0.8 * self.utilities[str(s_straight)] + \
                            0.1 * self.utilities[str(s_cw)] + \
                            0.1 * self.utilities[str(s_ccw)]

            self.policy[str(s)] = np.argmax(policy)

        self.e = min(self.e + 0.1, 0.95)


    def store(self, r, s_, done):
        # Update previous states
        for key in self.current_episode_rewards:
            U, n = self.current_episode_rewards[key][0]
            n += 1
            U += r * self.gamma ** n
            self.current_episode_rewards[key][0] = (U, n)

        # Add state to memory
        if s_ not in self.visited_states:
            self.visited_states.append(s_)
        key = str(s_)
        if key not in self.current_episode_rewards:
            self.current_episode_rewards[key] = [(r, 0)]

        # check if done
        if done:
            entry = pd.DataFrame(self.current_episode_rewards)
            self.current_episode_rewards = {}
            self.all_episodes_rewards = self.all_episodes_rewards.append(entry)

    def calculate_outcomes(self, s, action):
        s_straight, s_cw, s_ccw = s, s, s

        if ((s + np.array(self.moves[action])) != (1, 1)).any():
            s_straight = np.array(s) + np.array(self.moves[action])

        if (s + np.array(self.moves[action], int) * rotate_cw != (1, 1)).any():
            s_cw = s + np.array(np.array(self.moves[action], int) * rotate_cw)[0]

        if (s + np.array(self.moves[action], int) * rotate_ccw != (1, 1)).any():
            s_ccw = s + np.array(np.array(self.moves[action], int) * rotate_ccw)[0]

        s_straight = max(0, s_straight[0]), max(0, s_straight[1])
        s_cw = max(0, s_cw[0]), max(0, s_cw[1])
        s_ccw = max(0, s_ccw[0]), max(0, s_ccw[1])
        s_straight = (min(s_straight[0], self.height - 1),
                      min(s_straight[1], self.width - 1))
        s_cw = (min(s_cw[0], self.height - 1),
                min(s_cw[1], self.width - 1))
        s_ccw = (min(s_ccw[0], self.height - 1),
                 min(s_ccw[1], self.width - 1))
        return s_straight, s_cw, s_ccw