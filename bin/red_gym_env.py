import uuid
from pathlib import Path
from collections import deque

import numpy as np
from gymnasium import Env, spaces

from red_gym_env_support import RedGymEnvSupport
from red_pyboy_manager import PyBoyManager
from red_gym_screen import RedGymScreen
from red_env_constants import *

def initialize_observation_space():
    return spaces.Dict(
        {
            "pos": spaces.MultiDiscrete([BYTE_SIZE] * POS_BYTES),
            "pos_memory": spaces.MultiDiscrete([BYTE_SIZE] * POS_HISTORY_SIZE * POS_BYTES)        }
    )


class RedGymEnv(Env):
    def __init__(self, config=None):
        self.debug = config.get('debug', False)
        self.s_path = Path(config['session_path'])
        self.save_final_state = config.get('save_final_state', False)
        self.print_rewards = config.get('print_rewards', False)
        self.headless = config.get('headless', True)
        self.init_state = config['init_state']
        self.act_freq = config.get('action_freq', 1)
        self.max_steps = config.get('max_steps', 1000)
        self.early_stopping = config.get('early_stop', False)
        self.save_video = config.get('save_video', False)
        self.fast_video = config.get('fast_video', False)
        self.reward_scale = config.get('reward_scale', 1)
        self.extra_buttons = config.get('extra_buttons', False)
        self.instance_id = config.get('instance_id', str(uuid.uuid4())[:8])
        self.frame_stacks = config.get('frame_stacks', FRAME_STACKS)
        self.output_shape = config.get('output_shape', OUTPUT_SHAPE)
        self.output_full = config.get('output_full', OUTPUT_FULL)
        self.rom_location = config.get('gb_path', '../PokemonRed.gb')

        self.support = RedGymEnvSupport(self)
        self.screen = RedGymScreen(self)
        self.game = PyBoyManager(self)

        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []
        self.agent_stats = []

        # Stable Baselines3 env config
        self.action_space = spaces.Discrete(len(self.game.valid_actions))
        self.observation_space = initialize_observation_space()

        self.reset()

        assert len(initialize_observation_space()) == len(self._get_observation())

    def reset(self, seed=None):
        self.seed = seed
        self.game.reload_game()
        self._reset_env_state()

        return self._get_observation(), {}

    def _reset_env_state(self):
        self.step_count = 0
        self.total_reward = 0
        self.pos_memory = np.zeros((POS_HISTORY_SIZE * POS_BYTES,), dtype=np.uint8)
        self.pos = np.zeros((POS_BYTES,), dtype=np.uint8)
        self.steps_discovered = 0
        self.visited_pos = {}
        self.visited_pos_order = deque()
        self.new_map = False
        self.moved_location = False
        self.reset_count += 1
        self.agent_stats = []

    def step(self, action):
        self._update_pre_action_memory()
        self.game.run_action_on_emulator(action)
        self._update_post_action_memory()

        self._update_rewards(action)
        self._append_agent_stats(action)

        observation = self._get_observation()

        step_limit_reached = self.get_check_if_done()
        self.support.save_and_print_info(step_limit_reached)

        self.step_count += 1

        return observation, self.total_reward * 0.1, False, step_limit_reached, {}

    def _update_pre_action_memory(self):
        x_pos_cur, y_pos_cur, n_map_cur = self.support.get_and_save_pos()

    def _update_post_action_memory(self):
        self.support.update_movement(x_pos_cur, y_pos_cur, n_map_cur)

    def get_check_if_done(self):
        return self.support.check_if_done()

    def _append_agent_stats(self, action):
        self.agent_stats.append({
            'reward': self.total_reward,
            'last_action': action,
            'discovered': self.steps_discovered
        })

    def _get_observation(self):
        self.support.update_coord_obs()
        observation = {
            "pos": self.pos,
            "pos_memory": self.pos_memory
        }
        return observation

    def _update_rewards(self, action):
        state_scores = {
            'movement': self.reward_scale * self._get_movement_reward(),
        }

        self.total_reward = sum(val for _, val in state_scores.items())
