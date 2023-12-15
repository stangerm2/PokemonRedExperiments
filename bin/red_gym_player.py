from red_env_constants import *
from ram_reader.red_memory_map import *


class RedGymPlayer:
    def __init__(self, env):
        self.env = env
        if env.debug:
            print('**** RedGymPlayer ****')
        
        self.current_badges = 0

    def get_badge_reward(self):
        badges = self.env.game.player.get_badges()
        if badges > self.current_badges:
            self.current_badges = badges
            return 1000
            
        return 0