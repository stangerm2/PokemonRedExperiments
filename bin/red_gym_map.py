import math

import numpy as np
from collections import deque

# Assuming these constants are defined in red_env_constants
from red_env_constants import *


TWO_MOVE_POS = [
    (-2, 0),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -2),
    (0, -1),
    (0, 0),
    (0, 1),
    (0, 2),
    (1, -1),
    (1, 0),
    (1, 1),
    (2, 0)
]

class RedGymMap:
    def __init__(self, env):
        if env.debug:
            print('**** RedGymMap ****')

        self.env = env
        self.x_pos_org, self.y_pos_org, self.n_map_org = None, None, None
        self.pos_memory = np.zeros((POS_HISTORY_SIZE * XYM_BYTES,), dtype=np.uint8)
        self.unseen_positions = np.zeros((NEXT_STEP_VISITED,), dtype=np.uint8)
        self.pos = np.zeros((POS_BYTES,), dtype=np.uint8)
        self.steps_discovered = 0
        self.visited_pos = {}
        self.visited_pos_order = deque()
        self.new_map = 0
        self.moved_location = False
        self.location_history = deque()

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

    def update_map_obs(self):
        x_pos_new, y_pos_new, n_map_new = self.get_current_location()

        self._update_pos_obs(x_pos_new, y_pos_new, n_map_new)
        self._update_unseen_pos_obs(x_pos_new, y_pos_new, n_map_new)

    def _update_pos_obs(self, x_pos_new, y_pos_new, n_map_new):
        facing_dir = self.env.game.get_memory_value(PLAYER_FACING_DIR)
        tile_above = self.env.game.get_memory_value(TILE_ABOVE_PLAYER)
        tile_below = self.env.game.get_memory_value(TILE_BELOW_PLAYER)
        tile_left = self.env.game.get_memory_value(TILE_LEFT_OF_PLAYER)
        tile_right = self.env.game.get_memory_value(TILE_RIGHT_OF_PLAYER)
        tile_bump = self.env.game.get_memory_value(TILE_CURRENT_AND_FRONT_BUMP_PLAYER)

        self.pos = np.array([x_pos_new, y_pos_new, n_map_new, facing_dir, tile_above,
                            tile_below, tile_left, tile_right, tile_bump])

        self.pos_memory = np.roll(self.pos_memory, POS_BYTES)
        self.pos_memory[:XYM_BYTES] = self.pos[:XYM_BYTES]

        self.update_map_stats()

    def _update_unseen_pos_obs(self, x_pos_new, y_pos_new, n_map_new):
        i = 0
        for x_offset, y_offset in TWO_MOVE_POS:
            current_pos = (x_pos_new + x_offset, y_pos_new + y_offset, n_map_new)

            if current_pos in self.visited_pos:
                self.unseen_positions[i] = True
            else:
                self.unseen_positions[i] = False

            i += 1

        if self.env.debug:
            print(self.unseen_positions)

    def save_post_action_pos(self):
        x_pos_new, y_pos_new, n_map_new = self.get_current_location()
        self.moved_location = not (self.x_pos_org == x_pos_new and
                                   self.y_pos_org == y_pos_new and
                                   self.n_map_org == n_map_new)

        if self.moved_location:
            # Bug check: AI is only allowed to move 0 or 1 spots per turn, new maps change x,y ref pos so don't count.
            # When the game goes to a new map, it changes m first, then y,x will update on the next turn
            if self.new_map:
                self.x_pos_org, self.y_pos_org, self.n_map_org = self.get_current_location()
                self.new_map = False
            elif n_map_new == self.n_map_org:
                if not (abs(self.x_pos_org - x_pos_new) + abs(self.y_pos_org - y_pos_new) <= 1):
                    self.update_map_stats()

                    print()
                    print()
                    print()
                    print(self.env.instance_id)

                    debug_str = ""
                    while len(self.location_history):
                        debug_str += self.location_history.popleft()
                    self.env.support.save_debug_string(debug_str)
                    # assert False
            else:
                self.new_map = True

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

    def update_map_stats(self):
        new_x_pos, new_y_pos, new_map_n = self.get_current_location()

        debug_str = f"Moved: {self.moved_location} \n"
        if self.new_map:
            debug_str = f"\nNew Map!\n"
        debug_str += f"Start location: {self.x_pos_org, self.y_pos_org, self.n_map_org} \n"
        debug_str += f"New location: {new_x_pos, new_y_pos, new_map_n} \n"
        debug_str += f"\n"
        debug_str += f"{self.pos}"
        debug_str += f"\n"
        debug_str += f"{self.pos_memory}"

        if len(self.location_history) > 10:
            self.location_history.popleft()
        self.location_history.append(debug_str)

        if self.env.debug:
            print(debug_str)

