import sys
import uuid
import os
from math import floor, sqrt
import json
from pathlib import Path
from collections import deque

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
from pyboy.logger import log_level
import hnswlib
import mediapy as media
import pandas as pd
from functools import lru_cache
import threading
import math
import torch

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

class RedGymEnv(Env):

    def __init__(
            self, config=None):

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320  # 1000
        self.headless = config['headless']
        self.num_elements = 20000  # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []
        self.pos_elem_size = 3
        self.obs_memory_size = 7
        self.pos_memory = np.zeros((self.obs_memory_size * self.pos_elem_size,), dtype=np.uint8)  # x,y * freq=8
        self.pos = np.zeros((1,), dtype=np.uint8)
        self.steps_discovered = 0

        self.max_step_memory = 350  # ~410 steps in pallet
        self.seen_cords_order = deque()
        self.seen_coords = {}  # dict
        self.new_map = False
        self.moved_location = False

        self.frame_stacks = 3
        self.output_shape = (36, 40, 3)
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0],
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)

        self.mem_padding = 2
        self.mem_padding = 2
        self.memory_height = 8
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2]
        )

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        if self.extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                WindowEvent.PASS
            ])

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.observation_space = spaces.Dict(
            {
                "pos": spaces.MultiDiscrete([256] * self.pos_elem_size, ),
                "pos_memory": spaces.MultiDiscrete([256] * self.obs_memory_size * self.pos_elem_size, )
            }
        )

        head = 'headless' if config['headless'] else 'SDL2'

        log_level("ERROR")
        self.pyboy = PyBoy(
            config['gb_path'],
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not config['headless']:
            self.pyboy.set_emulation_speed(6)  # TODO: Config for slowing down speed

        self.reset()

    def reset(self, seed=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.agent_stats = []

        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            self.model_frame_writer.__enter__()

        self.step_count = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        self.pos_memory = np.zeros((self.obs_memory_size * self.pos_elem_size,), dtype=np.uint8)
        self.pos = np.zeros((1,), dtype=np.uint8)
        self.steps_discovered = 0

        self.max_step_memory = 350  # ~410 steps in pallet
        self.seen_cords_order = deque()
        self.seen_coords = {}  # dict
        self.new_map = False
        self.moved_location = False

        return self._get_obs(), {}

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray()  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255 * resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
        return game_pixels_render

    def step(self, action):
        self.update_seen_coords(action)

        self.run_action_on_emulator(action)

        self.append_agent_stats(action)

        new_reward = self.update_reward(action)

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        # self.save_screenshot(self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

        self.save_and_print_info(step_limit_reached)

        self.step_count += 1

        # print(f'obs: {self.obs_memory}\n')

        return obs, new_reward * 0.1, False, step_limit_reached, {}

    def _get_obs(self):
        self.update_coord_obs() # pos + map

        observation = {
            "pos": self.pos,
            "pos_memory": self.pos_memory
        }

        return observation

    def update_coord_obs(self):
        new_x_pos, new_y_pos, new_map_n = self.get_current_location()

        self.pos = np.array([new_x_pos, new_y_pos, new_map_n])
        # Calculate the starting index in self.obs_memory for the new data

        self.pos_memory = np.roll(self.pos_memory, self.pos_elem_size)
        self.pos_memory[0: 0 + self.pos_elem_size] = self.pos

    def get_termination_action(self, action):
        match action:
            case WindowEvent.PRESS_ARROW_DOWN:
                return WindowEvent.RELEASE_ARROW_DOWN
            case WindowEvent.PRESS_ARROW_UP:
                return WindowEvent.RELEASE_ARROW_UP
            case WindowEvent.PRESS_ARROW_LEFT:
                return WindowEvent.RELEASE_ARROW_LEFT
            case WindowEvent.PRESS_ARROW_RIGHT:
                return WindowEvent.RELEASE_ARROW_RIGHT
            case WindowEvent.PRESS_BUTTON_A:
                return WindowEvent.RELEASE_BUTTON_A
            case WindowEvent.PRESS_BUTTON_B:
                return WindowEvent.RELEASE_BUTTON_B
            case _:
                return WindowEvent.PASS

    def update_movement(self, x_pos_cur, y_pos_cur, n_map_cur):
        x_pos_new, y_pos_new, n_map_new = self.get_current_location()
        self.moved_location = not (x_pos_cur == x_pos_new and
                                   y_pos_cur == y_pos_new and
                                   n_map_cur == n_map_new)

        # print(f'moved_location: {self.moved_location}, self.new_map:{self.new_map}')
        if self.moved_location:
            # Bug check: AI is only allowed to move 0 or 1 spots per turn, new maps change x,y ref pos so don't count.
            # When the game goes to a new map, it changes m first, then y,x will update on the next turn
            if self.new_map:
                self.new_map = False
            elif n_map_new == n_map_cur:
                pass
                # print(f'id: {id(self)}, new location: {self.get_location_str(x_pos_new, y_pos_new, n_map_new)}')
                # assert abs(x_pos_cur - x_pos_new) + abs(y_pos_cur - y_pos_new) <= 1
            else:
                self.new_map = True

    def run_action_on_emulator(self, action):
        x_pos_cur, y_pos_cur, n_map_cur = self.get_current_location()

        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)

        # print(f'chat: {self.read_m(self.dialog)}')

        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq - 1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

        self.update_movement(x_pos_cur, y_pos_cur, n_map_cur)

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))

    def append_agent_stats(self, action):
        self.agent_stats.append({
            'reward': self.total_reward,
            'last_action': action,
            'discovered': self.steps_discovered
        })

    def get_current_location(self):
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)

        return x_pos, y_pos, map_n

    def get_location_str(self, x_pos, y_pos, map_n):
        return f"x:{x_pos} y:{y_pos} m:{map_n}"

    def update_seen_coords(self, action):
        x_pox, y_pos, map_n = self.get_current_location()
        current_location = self.get_location_str(x_pox, y_pos, map_n)

        if len(self.seen_cords_order) > self.max_step_memory:
            del_key = self.seen_cords_order.popleft()
            # print(f'cord_count: {len(self.seen_cords_order)}, delete key: {del_key}')
            del self.seen_coords[del_key]

        # print(f'order length: {len(self.seen_cords_order)}\n tracking cords: {list(self.seen_cords_order)}')

        if current_location in self.seen_coords:
            return

        self.seen_coords[current_location] = self.step_count
        self.seen_cords_order.append(current_location)

    def update_reward(self, action):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(
            [val for _, val in self.progress_reward.items()])  # sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward

        self.total_reward = new_total

        return new_total

    def check_if_done(self):
        return self.step_count >= self.max_steps

    def save_and_print_info(self, done):
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            for key, val in self.progress_reward.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.total_reward:5.2f}'
            prog_string += f' reward: {self.total_reward:5.2f}, seen_cord_len: {len(self.seen_coords)}'
            prog_string += f' steps_discovered: {self.steps_discovered}'
            print(f'\r{prog_string}', end='', flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f'curframe_{self.instance_id}.jpeg'),
                self.render(reduce_res=False))

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'),
                    self.render())
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'),
                    self.render(reduce_res=False))

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.progress_reward)
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == '1'

    def get_movement_reward(self):
        reward = 0
        bonus = math.log10(len(self.seen_coords)) if len(self.seen_coords) > 0 else 0
        bonus += math.log10(2049 - self.step_count) if (2048 - self.step_count) > 0 and len(
            self.seen_coords) >= self.max_step_memory - 1 else 0

        x_pox, y_pos, map_n = self.get_current_location()
        # TEST: Hack to stay out of grass, stay in pallet town
        if self.read_m(0xD35E) == 12:
            print(f'***************STAY IN PALLET TOWN**************************')
            reward = -.5
        # Ran into a wall, person, sign, ext..
        elif not self.moved_location:
            reward = 0
        # Stayed too close to the same location for too long
        elif self.get_location_str(x_pox, y_pos, map_n) in self.seen_coords:
            reward = 1
        else:
            reward = 1 + bonus
            self.steps_discovered += 1

        return reward

    def get_game_state_reward(self, print_stats=False):
        state_scores = {
            'movement': self.reward_scale * self.get_movement_reward(),
        }

        return state_scores

    def save_screenshot(self, x_pos_cur, y_pos_cur, n_map_cur, image=None):
        if image is None:
            image = self.render(reduce_res=False)

        ss_dir = self.s_path / Path(f'screenshots/{n_map_cur}_{x_pos_cur}_{y_pos_cur}')
        ss_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'{threading.get_ident()}_{self.step_count}.jpeg'),
            image)  # TODO: Does this match exactly the same image when OBS reward is called

    def get_map_location(self, map_idx):
        map_locations = {
            0: "Pallet Town",
            1: "Viridian City",
            2: "Pewter City",
            3: "Cerulean City",
            12: "Route 1",
            13: "Route 2",
            14: "Route 3",
            15: "Route 4",
            33: "Route 22",
            37: "Red house first",
            38: "Red house second",
            39: "Blues house",
            40: "oaks lab",
            41: "Pokémon Center (Viridian City)",
            42: "Poké Mart (Viridian City)",
            43: "School (Viridian City)",
            44: "House 1 (Viridian City)",
            47: "Gate (Viridian City/Pewter City) (Route 2)",
            49: "Gate (Route 2)",
            50: "Gate (Route 2/Viridian Forest) (Route 2)",
            51: "viridian forest",
            52: "Pewter Museum (floor 1)",
            53: "Pewter Museum (floor 2)",
            54: "Pokémon Gym (Pewter City)",
            55: "House with disobedient Nidoran♂ (Pewter City)",
            56: "Poké Mart (Pewter City)",
            57: "House with two Trainers (Pewter City)",
            58: "Pokémon Center (Pewter City)",
            59: "Mt. Moon (Route 3 entrance)",
            60: "Mt. Moon",
            61: "Mt. Moon",
            68: "Pokémon Center (Route 4)",
            193: "Badges check gate (Route 22)"
        }
        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return "Unknown Location"