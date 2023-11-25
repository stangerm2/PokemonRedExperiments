import math

import numpy as np
from collections import deque

# Assuming these constants are defined in red_env_constants
from red_env_constants import POS_HISTORY_SIZE, POS_BYTES, MAX_STEP_MEMORY, MAP_VALUE_PALLET_TOWN

class RedGymMap:
    def __init__(self, env):
        self.env = env

        self.x_pos_org, self.y_pos_org, self.n_map_org = None, None, None
        self.pos_memory = np.zeros((POS_HISTORY_SIZE * POS_BYTES,), dtype=np.uint8)
        self.pos = np.zeros((POS_BYTES,), dtype=np.uint8)
        self.steps_discovered = 0
        self.visited_pos = {}
        self.visited_pos_order = deque()
        self.new_map = False
        self.moved_location = False

    def pallet_town_explorer_reward(self):
        reward, bonus = 0, self._calculate_exploration_bonus()

        x_pos, y_pos, map_n = self.get_current_location()
        if map_n == MAP_VALUE_PALLET_TOWN:
            reward = -.5
        elif not self.moved_location:
            reward = 0
        elif (x_pos, y_pos, map_n) in self.visited_pos:
            reward = 1
        else:
            reward = 1 + bonus
            self.steps_discovered += 1

        return reward

    def _calculate_exploration_bonus(self):
        bonus = math.log10(len(self.visited_pos)) if len(self.visited_pos) > 0 else 0

        steps_remaining = (self.env.max_steps + 1) - self.env.step_count
        if self.steps_discovered >= len(self.visited_pos):
            bonus += math.log10(steps_remaining)

        return bonus

    def update_pos_obs(self):
        new_x_pos, new_y_pos, new_map_n = self.get_current_location()
        self.pos = np.array([new_x_pos, new_y_pos, new_map_n])

        self.pos_memory = np.roll(self.pos_memory, POS_BYTES)
        self.pos_memory[:POS_BYTES] = self.pos[:POS_BYTES]

        print(f"Moved: {self.moved_location}")
        print(f"Start location: {self.x_pos_org, self.y_pos_org, self.n_map_org}")
        print(f"New location: {new_x_pos, new_y_pos, new_map_n}")
        print()
        print(self.pos)
        print()
        print(self.pos_memory)

    def save_post_action_pos(self):
        x_pos_new, y_pos_new, n_map_new = self.get_current_location()
        self.moved_location = not (self.x_pos_org == x_pos_new and
                                   self.y_pos_org == y_pos_new and
                                   self.n_map_org == n_map_new)

        if self.moved_location:
            self.new_map = n_map_new != self.n_map_org

    def get_current_location(self):
        x_pos = self.env.game.get_memory_value(0xD362)
        y_pos = self.env.game.get_memory_value(0xD361)
        map_n = self.env.game.get_memory_value(0xD35E)
        return x_pos, y_pos, map_n

    def save_pre_action_pos(self):
        # TODO: Only save pos history if moved but need moved flag in obs
        self.x_pos_org, self.y_pos_org, self.n_map_org = self.get_current_location()

        if len(self.visited_pos_order) > MAX_STEP_MEMORY:
            del_key = self.visited_pos_order.popleft()
            del self.visited_pos[del_key]

        current_pos = (self.x_pos_org, self.y_pos_org, self.n_map_org)
        if current_pos not in self.visited_pos:
            self.visited_pos[current_pos] = self.env.step_count
            self.visited_pos_order.append(current_pos)

