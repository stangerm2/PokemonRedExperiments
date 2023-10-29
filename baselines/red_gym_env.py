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
import hnswlib
import mediapy as media
import pandas as pd
from functools import lru_cache
import threading
import math



from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

class RedGymEnv(Env):


    def __init__(
        self, config=None):

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320 #1000
        self.headless = config['headless']
        self.num_elements = 20000 # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.explore_weight = 1 if 'explore_weight' not in config else config['explore_weight']
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']
        self.similar_frame_dist = config['sim_frame_dist']
        self.reward_scale = 1 if 'reward_scale' not in config else config['reward_scale']
        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.single_move = 4  # The amount of frames needed to move exactly 1 square, facing from any direction
        self.all_runs = []
        self.interaction_started = False
        self.battle_started = False
        self.reward_memory = 1
        self.obs_memory = np.zeros((self.reward_memory, 3), dtype=np.uint8)

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        # Movement data of all movable NPCs, when adr data is 0 they're still (animation timer, movement timer trigger)
        # The animation timer must not be > 0 and movement timer must not be close to 0
        self.NpcMovement = [
            (0xc200, 0xc208),
            (0xc210, 0xc218),
            (0xc220, 0xc228),
            (0xc230, 0xc238),
            (0xc240, 0xc248),
            (0xc250, 0xc258),
            (0xc260, 0xc268),
            (0xc270, 0xc278),
            (0xc280, 0xc288),
            (0xc290, 0xc298),
            (0xc2A0, 0xc2A8),
            (0xc2B0, 0xc2B8),
            (0xc2C0, 0xc2C8),
            (0xc2D0, 0xc2D8),
            (0xc2E0, 0xc2E8),
            (0xc2F0, 0xc2F8)
        ]

        self.ExperienceLocations = [
            (0xD179, 0xD18A, 0xD18B),
            (0xD1A5, 0xD1A6, 0xD1A7),
            (0xD1D1, 0xD1D2, 0xD1D3),
            (0xD1FD, 0xD1FE, 0xD1FF),
            (0xD229, 0xD22A, 0xD22B),
            (0xD255, 0xD256, 0xD257),
        ]

        # non-zero when dialog or battle started
        self.dialog = 0x8800
        self.battle = 0xD057

        # CC51-CC53, interaction happening(text inc) usually all 0's but rare cases where 52 stays 0x14 (pokecenter),
        # still the best flag to check text scrolling is done, there is nothing else whole memory scoured
        self.interaction_1 = 0xCC51
        self.interaction_1 = 0xCC52
        self.interaction_1 = 0xCC53

        # sprite
        self.sprite_animation = 0xC108

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

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.reward_memory, 3), dtype=np.uint8)

        head = 'headless' if config['headless'] else 'SDL2'

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
        
        if self.use_screen_explore:
            self.init_knn()
        else:
            self.init_movement_memory()

        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)
        
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)

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
       
        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.step_count = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        self.interaction_started = False
        self.obs_memory = np.zeros((self.reward_memory, 3), dtype=np.uint8)

        return self.obs_memory, {}

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3), 
                    dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(), 
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render
    
    def step(self, action):
        x_pos, y_pos, map_n = self.get_current_location()
        # print(f'id: {id(self)}, current location: {self.get_location_str(x_pos, y_pos, map_n)}')
        self.update_seen_coords(action)

        self.run_action_on_emulator(action)
        # self.no_run_action_on_emulator(self.valid_actions[action])
        # self.append_agent_stats(action)

        x_pos, y_pos, map_n = self.get_current_location()
        # print(f'id: {id(self)}, new location: {self.get_location_str(x_pos, y_pos, map_n)}')

        # while not self.npcs_are_still():
        #    print(f'******* Movement Waiting')
        #    #self.pyboy.tick()
        #    time.sleep(1/1000)

        new_reward = self.update_reward(action)

        step_limit_reached = self.check_if_done()

        new_x_pos, new_y_pos, new_map_n = self.get_current_location()

        # for i in range(self.reward_memory - 1, -1, -1):  # Start from 498 and go to 0
        #    self.obs_memory[i] = self.obs_memory[i - 1]

        self.obs_memory[0] = np.array([new_map_n, new_x_pos, new_y_pos], dtype=np.uint8)

        #self.save_screenshot(self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

        self.save_and_print_info(step_limit_reached)

        self.step_count += 1

        #print(f'obs: {self.obs_memory}\n')

        return self.obs_memory, new_reward * 0.1, False, step_limit_reached, {}

    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim)  # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16)

    def init_movement_memory(self):
        self.max_step_memory = 350 # ~410 steps in pallet
        self.seen_cords_order = deque()
        self.seen_coords = {}  # dict
        self.seen_zones = set()
        self.new_map = False
        self.moved_location = False

    @lru_cache(None)
    def npcs_are_still(self):
        for npc in self.NpcMovement:
            animation_timer = self.read_m(npc[0])
            move_timer = self.read_m(npc[1])
            # print(f'anam: {hex(npc[0])}, move: {hex(npc[1])}')
            # print(f'anam timer: {animation_timer}, move timer: {self.read_m(npc[1])}')
            if animation_timer > 0 or (0 < move_timer < 3):
                # print(f'NOT STILL anam: {animation_timer}, move: {move_timer}')
                return False

        return True

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

        #print(f'moved_location: {self.moved_location}, self.new_map:{self.new_map}')
        if self.moved_location:
            # Bug check: AI is only allowed to move 0 or 1 spots per turn, new maps change x,y ref pos so don't count.
            # When the game goes to a new map, it changes m first, then y,x will update on the next turn
            if self.new_map:
                self.new_map = False
            elif n_map_new == n_map_cur:
                pass
                #print(f'id: {id(self)}, new location: {self.get_location_str(x_pos_new, y_pos_new, n_map_new)}')
                #assert abs(x_pos_cur - x_pos_new) + abs(y_pos_cur - y_pos_new) <= 1
            else:
                self.new_map = True

    def run_button_cmd(self, action, termination_action, frames):
        # press button then release after some steps
        self.pyboy.send_input(action)

        #print(f'\nbutton')
        frames = self.single_move
        if action == WindowEvent.PRESS_BUTTON_A:
            frames = 80

        # TODO: Still need a way to verify button action is frame complete
        for i in range(frames):
            self.pyboy.tick()
            interaction_1 = self.pyboy.get_memory_value(0xCC51)
            interaction_2 = self.pyboy.get_memory_value(0xCC52)
            interaction_3 = self.pyboy.get_memory_value(0xCC53)

            # Act like a 10 year old and spam the A button
            if action == WindowEvent.PRESS_BUTTON_A and i != 0:
                if i % 10 == 0:
                    self.pyboy.send_input(action)
                elif i % 5 == 0:
                    self.pyboy.send_input(termination_action)

            interaction_started = self.read_m(self.dialog) != 0
            # Pressed A button but there is no interaction, allowed but inefficent or
            # there is already a completed text screen present
            if ((i > 10 and not interaction_started) or
                (interaction_1 == 1 and interaction_3 == 1) or
                (interaction_2 == 2 and interaction_3 == 2)):
                break

            #if interaction_started:
            #print(f'Moving: {moving_animation}, map: {self.read_m(0xD35E)}, frame: {i}')
            #    self.print_memory(i)

            if i == 79:
                x_pos_cur, y_pos_cur, n_map_cur = self.get_current_location()
                self.save_screenshot(x_pos_cur, y_pos_cur, n_map_cur)

        self.pyboy.send_input(termination_action)
        #self.print_memory(0)

    def print_memory(self, i):
        interaction_1 = self.pyboy.get_memory_value(0xCC51)
        interaction_2 = self.pyboy.get_memory_value(0xCC52)
        interaction_3 = self.pyboy.get_memory_value(0xCC53)
        interaction_4 = self.pyboy.get_memory_value(0xFF8C)

        print(
            f'i: {i}'
            f' c: {interaction_4},'
            f' 1: {interaction_1},'
            f' 2: {interaction_2},'
            f' 3: {interaction_3}')


    def run_dpad_cmd(self, action, termination_action):
        animation_started = False

        #print(f'\naction: {WindowEvent(action).__str__()}, x:{x_pos_cur}, y:{y_pos_cur}, map: {n_map_cur}')

        # press button then release after some steps
        self.pyboy.send_input(action)

        #print(f'chat: {self.read_m(self.dialog)}')

        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)

        frames = self.act_freq

        # AI sent a dpad cmd during a chat interaction, which is allowed but just unproductive. Don't burn more
        # resources than needed to run the cmd.
        if self.read_m(self.dialog) != 0:
            frames = self.single_move

        # Frames for animation vary, xy move ~22, wall collision ~13 & zone reload ~66. Wasted frames are wasted
        # training cycles, frames/tick is expensive. Also, try to prefect OBS output image with completed frame cycle.
        for i in range(frames):
            # self.save_screenshot(self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E), i)

            # wSpritePlayerStateData1AnimFrameCounter, non-zero when sprite anim frames are playing
            moving_animation = self.read_m(self.sprite_animation)

            if animation_started and moving_animation == 0:
                break

            # Release the key once the animation starts, thus it should only be possible to advance 1 position.
            if moving_animation > 0:
                animation_started = True
                self.pyboy.send_input(termination_action)

            if self.save_video and not self.fast_video:
                self.add_video_frame()

            self.pyboy.tick()

        # Didn't get term key so do it now
        if not animation_started:
            self.pyboy.send_input(termination_action)

        if not self.save_video and self.headless:
            self.pyboy._rendering(False)

        if self.save_video and self.fast_video:
            self.add_video_frame()

        # self.save_screenshot(x_pos_new, y_pos_new, n_map_new)

    def no_run_action_on_emulator(self, action):
        termination_action = self.get_termination_action(action)

        x_pos_cur, y_pos_cur, n_map_cur = self.get_current_location()
        print(f'\naction: {WindowEvent(action).__str__()}, x:{x_pos_cur}, y:{y_pos_cur}, map: {n_map_cur}')

        if termination_action == WindowEvent.PASS:
            # print(f'ignoring command')
            return

        if termination_action == WindowEvent.RELEASE_BUTTON_A or termination_action == WindowEvent.RELEASE_BUTTON_B:
            self.run_button_cmd(action, termination_action, 24)
        else:
            self.run_dpad_cmd(action, termination_action)
            self.update_movement(x_pos_cur, y_pos_cur, n_map_cur)

        self.pyboy.tick()

        self.interaction_started = self.read_m(self.dialog) != 0
        self.battle_started = self.read_m(self.battle) != 0
        # print(f'interaction: {self.interaction_started}, battle: {self.battle_started}')

    def run_action_on_emulator(self, action):
        x_pos_cur, y_pos_cur, n_map_cur = self.get_current_location()

        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)

        #print(f'chat: {self.read_m(self.dialog)}')

        for i in range(self.act_freq):
            #print(
            #    f'v1: {self.pyboy.get_memory_value(0xCC51)},'
            #    f' 2: {self.pyboy.get_memory_value(0xCC52)},'
            #    f' 3: {self.pyboy.get_memory_value(0xCC53)}')

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
            if i == self.act_freq-1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

        self.update_movement(x_pos_cur, y_pos_cur, n_map_cur)

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))
    
    def append_agent_stats(self, action):
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)
        levels = [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        if self.use_screen_explore:
            expl = ('frames', self.knn_index.get_current_count())
        else:
            expl = ('coord_count', len(self.seen_coords))
        self.agent_stats.append({
            'step': self.step_count, 'x': x_pos, 'y': y_pos, 'map': map_n,
            'last_action': action,
            'pcount': self.read_m(0xD163), 'levels': levels, 'ptypes': self.read_party(),
            'hp': self.read_hp_fraction(),
            expl[0]: expl[1],
            'deaths': self.died_count, 'badge': self.get_badges(),
            'event': self.progress_reward['event'], 'healr': self.total_healing_rew
        })

    def update_frame_knn_index(self, frame_vec):
        
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current 
            labels, distances = self.knn_index.knn_query(frame_vec, k = 1)
            if distances[0][0] > self.similar_frame_dist:
                # print(f"distances[0][0] : {distances[0][0]} similar_frame_dist : {self.similar_frame_dist}")
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )


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
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        new_total = sum(
            [val for _, val in self.progress_reward.items()])  # sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        # if new_step < 0:
        # print(f'\n\nreward went down! {self.progress_reward}\n\n')
        # self.save_screenshot('neg_reward')

        self.total_reward = new_total

        return new_total

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        # these values are only used by memory
        return (0,
                0,
                prog['movement'] * 150 / (self.explore_weight * self.reward_scale))

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height
        
        def make_reward_channel(r_val):
            col_steps = self.col_steps
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered) 
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory
        
        level, hp, explore = self.group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore)
        ), axis=-1)
        
        if self.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def create_recent_memory(self):
        return rearrange(
            self.recent_memory, 
            '(w h) c -> h w c', 
            h=self.memory_height)

    def check_if_done(self):
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        #done = self.read_hp_fraction() == 0
        return done

    def save_and_print_info(self, done):
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            for key, val in self.progress_reward.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.total_reward:5.2f}'
            prog_string += f' reward: {self.total_reward:5.2f}, step_discovered: {len(self.seen_coords)}'
            print(f'\r{prog_string}', end='', flush=True)

        '''
        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f'curframe_{self.instance_id}.jpeg'),
                self.render(reduce_res=False))


        if self.print_rewards and done:
            #print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'), 
                    obs_memory)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'), 
                    self.render(reduce_res=False))

        '''

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
        return bin(256 + self.read_m(addr))[-bit-1] == '1'
    
    def get_levels_sum(self):
        poke_levels = [max(self.read_m(a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level
    
    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum-explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew
    
    def get_knn_reward(self):
        
        pre_rew = self.explore_weight * 0.005
        post_rew = self.explore_weight * 0.01
        cur_size = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        return base + post

    def get_movement_reward(self):
        reward = 0
        bonus = math.log10(len(self.seen_coords)) if len(self.seen_coords) > 0 else 0
        bonus += math.log10(2049 - self.step_count) if (2048 - self.step_count) > 0 and len(
            self.seen_coords) >= self.max_step_memory - 1 else 0

        x_pox, y_pos, map_n = self.get_current_location()
        # TEST: Hack to stay out of grass, stay in pallet town
        if self.read_m(0xD35E) == 12:
            print(f'***************STAY IN PALLET TOWN**************************')
            reward = -1
        # Ran into a wall, person, sign, ext..
        elif not self.moved_location:
            reward = 0
        # Stayed too close to the same location for too long
        elif self.get_location_str(x_pox, y_pos, map_n) in self.seen_coords:
            reward = 1
        else:
            reward = 1 + bonus

        return reward

    def get_zone_reward(self):
        x_pox, y_pos, map_n = self.get_current_location()

        if map_n in self.seen_zones:
            return 0

        self.seen_zones.add(map_n)

        return 7

    # def get_battle_reward(self):
    #    if self.exp_profile.empty:
    #        for pokemon in party:
    #            self.exp_profile += exp

    #    if self.battle_started && cur_exp != exp_now:

    # Existence is pain, use the time you have wisely
    def get_turn_reward(self):
        return -1

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]
    
    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                    self.save_screenshot('healing')
                self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1
                
    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        museum_ticket = (0xD754, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
        0,
    )

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        '''
        num_poke = self.read_m(0xD163)
        poke_xps = [self.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        #money = self.read_money() - 975 # subtract starting money
        seen_poke_count = sum([self.bit_count(self.read_m(i)) for i in range(0xD30A, 0xD31D)])
        all_events_score = sum([self.bit_count(self.read_m(i)) for i in range(0xD747, 0xD886)])
        oak_parcel = self.read_bit(0xD74E, 1) 
        oak_pokedex = self.read_bit(0xD74B, 5)
        opponent_level = self.read_m(0xCFF3)
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        enemy_poke_count = self.read_m(0xD89C)
        self.max_opponent_poke = max(self.max_opponent_poke, enemy_poke_count)
        
        if print_stats:
            print(f'num_poke : {num_poke}')
            print(f'poke_levels : {poke_levels}')
            print(f'poke_xps : {poke_xps}')
            #print(f'money: {money}')
            print(f'seen_poke_count : {seen_poke_count}')
            print(f'oak_parcel: {oak_parcel} oak_pokedex: {oak_pokedex} all_events_score: {all_events_score}')
        '''
        
        state_scores = {
            # 'event': self.reward_scale*self.update_max_event_rew(),
            # 'party_xp': self.reward_scale*0.1*sum(poke_xps),
            # 'level': self.reward_scale*self.get_levels_reward(),
            # 'heal': self.reward_scale*self.total_healing_rew,
            # 'op_lvl': self.reward_scale*self.update_max_op_level(),
            # 'dead': self.reward_scale*-0.1*self.died_count,
            # 'badge': self.reward_scale*self.get_badges() * 5,
            # 'op_poke': self.reward_scale*self.max_opponent_poke * 800,
            # 'money': self.reward_scale* money * 3,
            # 'seen_poke': self.reward_scale * seen_poke_count * 400,
            # 'explore': self.reward_scale * self.get_knn_reward()
            'movement': self.reward_scale * self.get_movement_reward(),
            # 'zone': self.reward_scale * self.get_zone_reward(),
            # 'turn': self.reward_scale * self.get_turn_reward(),
            # 'battle': self.reward_scale * self.get_battle_reward()
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

    def update_max_op_level(self):
        #opponent_level = self.read_m(0xCFE8) - 5 # base level
        opponent_level = max([self.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
        #if opponent_level >= 7:
        #    self.save_screenshot('highlevelop')
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.2
    
    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def read_hp_fraction(self):
        hp_sum = sum([self.read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
        max_hp_sum = sum([self.read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
    
    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
    def read_money(self):
        return (100 * 100 * self.read_bcd(self.read_m(0xD347)) + 
                100 * self.read_bcd(self.read_m(0xD348)) +
                self.read_bcd(self.read_m(0xD349)))
