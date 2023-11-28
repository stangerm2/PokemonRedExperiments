import uuid
from pathlib import Path

from gymnasium import Env, spaces

from red_gym_env_support import RedGymEnvSupport
from red_pyboy_manager import PyBoyManager
from red_gym_screen import RedGymScreen
from red_env_constants import *

from red_gym_map import *


def initialize_observation_space():
    return spaces.Dict(
        {
            "pos": spaces.MultiDiscrete([BYTE_SIZE] * POS_BYTES),
            #"pos_memory": spaces.MultiDiscrete([BYTE_SIZE] * POS_HISTORY_SIZE * XYM_BYTES),
            "unseen_positions": spaces.MultiBinary(NEXT_STEP_VISITED),
        }
    )


class RedGymEnv(Env):
    def __init__(self, thread_id, config=None):
        self.debug = config.get('debug', False)
        if self.debug:
            print('**** RedGymEnv ****')
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
        self.thread_id = thread_id

        self.screen = RedGymScreen(self)
        self.game = PyBoyManager(self)

        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0

        # Stable Baselines3 env config
        self.action_space = spaces.Discrete(len(self.game.valid_actions))
        self.observation_space = initialize_observation_space()

        # assert len(initialize_observation_space()) == len(self._get_observation())

    def reset(self, seed=None):
        self.seed = seed
        self._reset_env_state()

        return self._get_observation(), {}

    def _reset_env_state(self):
        self.support = RedGymEnvSupport(self)

        self.game.reload_game()

        self.step_count = 0
        self.total_reward = 0
        self.reset_count += 1
        self.agent_stats = []

    def step(self, action):
        self._run_pre_action_steps()
        self.game.run_action_on_emulator(self.game.valid_actions[action])
        self._run_post_action_steps()

        self._update_rewards(action)
        self._append_agent_stats(action)

        observation = self._get_observation()

        step_limit_reached = self.get_check_if_done()
        self.support.save_and_print_info(step_limit_reached)

        self.step_count += 1

        return observation, self.total_reward * 0.1, False, step_limit_reached, {}

    def _run_pre_action_steps(self):
        self.support.map.save_pre_action_pos()

    def _run_post_action_steps(self):
        self.support.map.save_post_action_pos()

    def get_check_if_done(self):
        return self.support.check_if_done()

    def _append_agent_stats(self, action):
        self.agent_stats.append({
            'reward': self.total_reward,
            # 'last_action': action,
            'discovered': self.support.map.steps_discovered,
            'p2p_found': self.support.map.tester.p2p_found,
        })

    def _get_observation(self):
        self.support.map.update_map_obs()

        observation = {
            "pos": self.support.map.pos,
            # "pos_memory": self.support.map.pos_memory,
            "unseen_positions": self.support.map.unseen_positions,
        }
        return observation

    def _update_rewards(self, action):
        state_scores = {
            'pallet_town_explorer': self.reward_scale * self.support.map.pallet_town_explorer_reward(),
            'pallet_town_point_nav': self.reward_scale * self.support.map.tester.pallet_town_point_nav(),
        }

        # TODO: If pass in some test flag run just a single test reward
        self.total_reward = sum(val for _, val in state_scores.items())
