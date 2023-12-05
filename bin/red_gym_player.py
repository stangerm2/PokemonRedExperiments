from red_env_constants import *
from ram_reader.red_memory_map import *

from red_gym_obs_tester import RedGymObsTester


class RedGymPlayer:
    def __init__(self, env):
        if env.debug:
            print('**** RedGymPlayer ****')