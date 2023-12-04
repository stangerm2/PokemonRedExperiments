import math
import torch

import numpy as np
from collections import deque

# Assuming these constants are defined in red_env_constants
from red_env_constants import *
from ram_reader.red_memory_map import *

from red_gym_obs_tester import RedGymObsTester


class RedGymMap:
    def __init__(self, env):
        if env.debug:
            print('**** RedGymMap ****')

        self.env = env
        self.x_pos_org, self.y_pos_org, self.n_map_org = None, None, None
        self.steps_discovered = 0
        self.visited_pos = {}
        self.visited_pos_order = deque()
        self.new_map = 0
        self.moved_location = False
        self.location_history = deque()

        self.screen = np.zeros((SCREEN_VIEW_SIZE + 3, SCREEN_VIEW_SIZE), dtype=np.float32)
        self.visited = np.zeros((SCREEN_VIEW_SIZE + 3, SCREEN_VIEW_SIZE), dtype=np.uint8)

        self.tester = RedGymObsTester(self)

    def pallet_town_explorer_reward(self):
        reward, bonus = 0, self._calculate_exploration_bonus()

        x_pos, y_pos, map_n = self.get_current_location()
        if map_n == MAP_VALUE_PALLET_TOWN:
            reward = -.5
        elif not self.moved_location:
            reward = 0
        elif (x_pos, y_pos, map_n) in self.visited_pos:
            reward = 0.01
        else:
            reward = 1
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

        self._update_tile_obs(x_pos_new, y_pos_new, n_map_new)
        self._update_visited_obs(x_pos_new, y_pos_new, n_map_new)
        self._update_npc_obs(x_pos_new, y_pos_new, n_map_new)  # Overwrites '_update_visited_obs' for npc's, must stay in order
        self._update_pos_obs(x_pos_new, y_pos_new, n_map_new)

        self.update_map_stats()
        

    def _update_tile_obs(self, x_pos_new, y_pos_new, n_map_new):
        # Starting addresses for each row
        starting_addresses = [TILE_COL_0_ROW_0, TILE_COL_0_ROW_1, TILE_COL_0_ROW_2, TILE_COL_0_ROW_3,
                               TILE_COL_0_ROW_4, TILE_COL_0_ROW_5, TILE_COL_0_ROW_6]
        num_columns = SCREEN_VIEW_SIZE
        increment_per_column = 2

        for row, start_addr in enumerate(starting_addresses):
            for col in range(num_columns):
                address = start_addr + col * increment_per_column
                
                tile_val = self.env.game.get_memory_value(address)
                self.screen[row][col] = self.env.memory.byte_to_float_norm[tile_val] + 0.1  # Norm range 0 - 0.0622 for byte input, assign higher prio w/ .9


    def _update_visited_obs(self, x_pos_new, y_pos_new, n_map_new):
        callback = lambda x, y, pos: self._update_matrix_visited(x, y, pos)
        self._traverse_matrix(x_pos_new, y_pos_new, n_map_new, callback)

        # NOTE: it's critical to NOT set cur pos as visited on the obs until the next turn, it REALLY helps the AI
        # self.visited[3][3] = 0


    def _update_npc_obs(self, x_pos_new, y_pos_new, n_map_new):
        sprites = self._get_sprites(n_map_new)

        callback = lambda x, y, pos: self._update_matrix_npc(x, y, pos, sprites)
        self._traverse_matrix(x_pos_new, y_pos_new, n_map_new, callback)


    def _update_pos_obs(self, x_pos_new, y_pos_new, n_map_new):
        x_pos_binary = format(x_pos_new, f'0{SCREEN_VIEW_SIZE}b')
        y_pos_binary = format(y_pos_new, f'0{SCREEN_VIEW_SIZE}b')
        m_pos_binary = format(n_map_new, f'0{SCREEN_VIEW_SIZE}b')
    
        for i, bit in enumerate(x_pos_binary):
            self.screen[SCREEN_VIEW_SIZE][i] = bit
            self.visited[SCREEN_VIEW_SIZE][i] = bit

        for i, bit in enumerate(y_pos_binary):
            self.screen[SCREEN_VIEW_SIZE + 1][i] = bit
            self.visited[SCREEN_VIEW_SIZE + 1][i] = bit

        for i, bit in enumerate(m_pos_binary):
            self.screen[SCREEN_VIEW_SIZE + 2][i] = bit
            self.visited[SCREEN_VIEW_SIZE + 2][i] = bit


    def _traverse_matrix(self, x_pos_new, y_pos_new, n_map_new, callback):
        center_x = center_y = SCREEN_VIEW_SIZE // 2

        for y in range(SCREEN_VIEW_SIZE):
            for x in range(SCREEN_VIEW_SIZE):
                center_x = center_y = SCREEN_VIEW_SIZE // 2
                x_offset = x - center_x
                y_offset = y - center_y
                current_pos = x_pos_new + x_offset, y_pos_new + y_offset, n_map_new

                callback(x, y, current_pos)


    def _get_sprites(self, n_map_new):
        sprites = {}
        for i, sprite_addr in enumerate(SPRITE_STARTING_ADDRESSES):
            on_screen = self.env.game.get_memory_value(sprite_addr + 0x0002)

            if on_screen == 0xFF:
                continue

            # TODO: Moving sprites cause learning problems, figure out how to handle them later. This make's them invisible on the screen to the AI
            can_move = self.env.game.get_memory_value(sprite_addr + 0x0106)
            if can_move != 0xFF:
                continue
            
            picture_id = self.env.game.get_memory_value(sprite_addr)
            x_pos = self.env.game.get_memory_value(sprite_addr + 0x0105) - 4  # topmost 2x2 tile has value 4), thus the offset
            y_pos = self.env.game.get_memory_value(sprite_addr + 0x0104) - 4  # topmost 2x2 tile has value 4), thus the offset
            # NOTE: facing might be useful for trainer battle navigation but requires more complex matrix to add in
            # facing = self.env.game.get_memory_value(sprite_addr + 0x0009)

            sprites[(x_pos, y_pos, n_map_new)] = picture_id
            
        return sprites


    def _update_matrix_visited(self, x, y, pos):
        if pos in self.visited_pos:
            self.visited[y][x] = 0
        else:
            self.visited[y][x] = 1


    def _update_matrix_npc(self, x, y, pos, sprites):
        if pos in sprites:
            self.screen[y][x] = self.env.memory.byte_to_float_norm[sprites[pos]]  # Norm range 0 - 0.0622 for byte input, assign higher prio w/ .8


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
        x_pos = self.env.game.get_memory_value(PLAYER_LOCATION_X)
        y_pos = self.env.game.get_memory_value(PLAYER_LOCATION_Y)
        map_n = self.env.game.get_memory_value(PLAYER_MAP)
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
        debug_str += f"{self.tester.p2p_obs}"
        debug_str += f"\n"
        debug_str += f"{self.screen}"
        debug_str += f"\n"
        debug_str += f"{self.visited}"

        if len(self.location_history) > 10:
            self.location_history.popleft()
        self.location_history.append(debug_str)

        if self.env.debug:
            print(debug_str)

