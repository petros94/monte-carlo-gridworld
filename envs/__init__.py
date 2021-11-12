from envs.simple_gridworld_env import SimpleGridworldEnv    # noqa
from gym.envs.registration import register

register(
        id='SimpleGridworldEnv-v0',
        entry_point='envs:SimpleGridworldEnv',
        )
