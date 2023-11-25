import math
from red_env_constants import *


class TestSubEnv:
    def __init__(self, env):

        def _get_movement_reward(self):
            reward = 0
            bonus = math.log10(len(self.env.visited_pos)) if len(self.env.visited_pos) > 0 else 0
            bonus += math.log10(2049 - self.env.step_count) if (2048 - self.env.step_count) > 0 and len(
                self.env.visited_pos) >= MAX_STEP_MEMORY - 1 else 0

            x_pos, y_pos, map_n = self.get_current_location()
            if self.env.game.get_memory_value(0xD35E) == 12:
                reward = -.5
            elif not self.env.moved_location:
                reward = 0
            elif (x_pos, y_pos, map_n) in self.env.visited_pos:
                reward = 1
            else:
                reward = 1 + bonus
                self.env.steps_discovered += 1

            return reward
