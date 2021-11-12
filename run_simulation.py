from agents import MCAgent
import pandas as pd
import gym
import envs   # noqa
import matplotlib.pyplot as plt

env = gym.make('SimpleGridworldEnv-v0')
agent = MCAgent()

def test_mc_agent(windy_gridworld):
    utilities_error = pd.DataFrame()
    epsilon = 0.001
    delta = 1
    mc_sample_episodes = 5
    episode_number = 0
    step_count = 0
    while delta > epsilon:
        for episode in range(mc_sample_episodes):
            step_count = 0
            s = windy_gridworld.reset()
            episode_number += 1

            while True:
                a = agent.choose_action(s)
                s_, r, done, info = windy_gridworld.step(a)
                agent.store(r, s_, done)

                windy_gridworld.render()

                s = s_
                step_count += 1

                if done:
                    break
        # Evaluate policy
        delta = agent.policy_evaluation()
        # Change policy
        agent.policy_iteration()

        print("Policy changed, Utilities changed ", delta)
        utilities_error = utilities_error.append(pd.DataFrame({'steps': [step_count], 'episode': [episode_number], 'error': [delta]}))

    print("Utilities diff:")
    print(utilities_error)
    print("Utilities:")
    print(agent.utilities)
    print("Policy:")
    print(agent.policy)
    return utilities_error, agent.utilities, agent.policy


if __name__ == "__main__":
    utilities_error, agent.utilities, agent.policy = test_mc_agent(env)
    plt.figure()
    plt.plot(utilities_error.episode, utilities_error.error)
    plt.savefig("episode_utilities_error_" + str(agent.gamma) + ".png")
