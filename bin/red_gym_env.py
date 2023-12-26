import random
import uuid
from pathlib import Path

from gymnasium import Env, spaces

from red_gym_env_support import RedGymEnvSupport, RedGymGlobalMemory
from red_pyboy_manager import PyBoyManager, pyboy_init_actions
from red_gym_screen import RedGymScreen
from red_gym_player import RedGymPlayer
from red_gym_map import RedGymMap
from red_gym_battle import RedGymBattle
from red_gym_map import *
from red_env_constants import *

from ram_reader.red_ram_api import *
'''            # Game View:
            "screen": spaces.Box(low=0, high=1, shape=(SCREEN_VIEW_SIZE + 3, SCREEN_VIEW_SIZE), dtype=np.float32),
            "visited": spaces.Box(low=0, high=1, shape=(SCREEN_VIEW_SIZE + 3, SCREEN_VIEW_SIZE), dtype=np.uint8),
            "action": spaces.MultiDiscrete([len(pyboy_init_actions(extra_buttons)) + 1]),
            #"p2p": spaces.MultiBinary(150),

            # Game:
            "game_state": spaces.Discrete(MENU_TOTAL_SIZE + 1),
            "move_allowed": spaces.Discrete(2),  # True or False

            # Player:
            "pokemon_roster": spaces.Box(low=0, high=1, shape=(POKEMON_MAX_COUNT, POKEMON_TOTAL_ATTRIBUTES), dtype=np.float32),
            "money": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "badges": spaces.MultiBinary(4),  # 8 badges inside 4 bits

            # Items
            "bag_ids": spaces.Box(low=0, high=1, shape=(BAG_SIZE,), dtype=np.float32),
            "bag_quan": spaces.Box(low=0, high=1, shape=(BAG_SIZE,), dtype=np.float32),
            "pc_item_ids": spaces.Box(low=0, high=1, shape=(STORAGE_SIZE,), dtype=np.float32),
            "pc_item_quan": spaces.Box(low=0, high=1, shape=(STORAGE_SIZE,), dtype=np.float32),
            "pc_pokemon": spaces.Box(low=0, high=1, shape=(BOX_SIZE, 2), dtype=np.float32), # 2 = Pokemon ID & Level
            "item_selection_quan": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # Quantity of item selected (to buy/sell), 0-99

            # World
            "milestones": spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32),  # TODO: Import better milestone list
            "audio": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "pokemart_items": spaces.Box(low=0, high=1, shape=(POKEMART_AVAIL_SIZE,), dtype=np.float32),

            # Battle
            "battle_type": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "enemies_left": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "player_stats": spaces.Box(low=0, high=1, shape=(BATTLE_TOTAL_PLAYER_ATTRIBUTES,), dtype=np.float32),
            "enemy_stats": spaces.Box(low=0, high=1, shape=(BATTLE_TOTAL_ENEMIES_ATTRIBUTES,), dtype=np.float32),
            "turn_info": spaces.Box(low=0, high=1, shape=(BATTLE_TOTAL_TURN_ATTRIBUTES,), dtype=np.float32),'''

def initialize_observation_space(extra_buttons):
    return spaces.Dict(
        {
            # Game View:
            "screen": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.float32),
            "visited": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.uint8),
            "walkable": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.uint8),
            "coordinates": spaces.Box(low=0, high=1, shape=(3, 7), dtype=np.uint8),
            "action": spaces.Discrete(7),

            # Game:
            "game_state": spaces.Discrete(117),

            # Player:
            "pokemon_lineup": spaces.Box(low=0, high=255, shape=(6, 13), dtype=np.uint8),
            "pokemon_health": spaces.Box(low=0, high=703, shape=(6, 2), dtype=np.int16),
            "pokemon_stats": spaces.Box(low=0, high=458, shape=(6, 4), dtype=np.int16),
            #"badges": spaces.Box(low=0, high=255, shape=(1, ), dtype=np.uint8),

            # Battle
            "battle_type": spaces.Box(low=0, high=3, shape=(1,), dtype=np.uint8),
            "enemies_left": spaces.Box(low=0, high=6, shape=(1,), dtype=np.uint8),
            "player_stats": spaces.Box(low=0, high=255, shape=(7,), dtype=np.uint8),
            "enemy_stats": spaces.Box(low=0, high=255, shape=(13,), dtype=np.uint8),
            "turn_info": spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8),
            "type_hint": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
        }
    )

'''            



            # Items
            "bag_ids": spaces.Box(low=0, high=255, shape=(20,), dtype=np.uint8),
            "bag_quan": spaces.Box(low=0, high=255, shape=(20,), dtype=np.uint8),
            "pc_item_ids": spaces.Box(low=0, high=255, shape=(50,), dtype=np.uint8),
            "pc_item_quan": spaces.Box(low=0, high=255, shape=(50,), dtype=np.uint8),
            "pc_pokemon": spaces.Box(low=0, high=255, shape=(20, 2), dtype=np.uint8), # 2 = Pokemon ID & Level
            "item_selection_quan": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8), # Quantity of item selected (to buy/sell), 0-99

            # World
            "milestones": spaces.Box(low=0, high=255, shape=(9,), dtype=np.uint8),  # TODO: Import better milestone list
            "audio": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            "pokemart_items": spaces.Box(low=0, high=255, shape=(10,), dtype=np.uint8),

'''



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
        self.gameboy = PyBoyManager(self)
        self.memory = RedGymGlobalMemory()

        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0

        # Stable Baselines3 env config
        self.action_space = spaces.Discrete(len(self.gameboy.valid_actions))
        self.observation_space = initialize_observation_space(self.extra_buttons)

        np.set_printoptions(linewidth=np.inf)
        # assert len(initialize_observation_space()) == len(self._get_observation())

    def reset(self, seed=0):
        self._reset_env_state()

        return self._get_observation(), {}

    def _reset_env_state(self):
        self.support = RedGymEnvSupport(self)
        self.init_state = self.support.choose_random_game_load()
        print(f'Loading: {self.init_state}')

        self.map = RedGymMap(self)
        self.player = RedGymPlayer(self)
        self.battle = RedGymBattle(self)
        self.game = Game(self.gameboy.pyboy)

        self.gameboy.reload_game()

        self.step_count = 0
        self.total_reward = 0
        self.reset_count += 1
        self.agent_stats = []

    def step(self, action):
        self._run_pre_action_steps()
        self.gameboy.run_action_on_emulator(action)
        self.game.process_game_states()

        self._run_post_action_steps()

        self._append_agent_stats(action)

        observation = self._get_observation()
        self._update_rewards(action)

        step_limit_reached = self.get_check_if_done()
        self.support.save_and_print_info(step_limit_reached)

        self.step_count += 1

        return observation, self.total_reward * 0.001, False, step_limit_reached, {}


    def _run_pre_action_steps(self):
        self.map.save_pre_action_pos()
        self.battle.save_pre_action_battle()

    def _run_post_action_steps(self):
        self.map.save_post_action_pos()
        self.battle.save_post_action_battle()

    def get_check_if_done(self):
        return self.support.check_if_done()

    def _append_agent_stats(self, action):
        badges = self.game.player.get_badges()

        self.agent_stats.append({
            'reward': self.total_reward,
            # 'last_action': action,
            'discovered': self.map.steps_discovered,
            'collisions': self.map.collisions,
            #'badges' : badges[0],
            'wild_mon_killed': self.battle.wild_pokemon_killed,
            'trainers_killed': self.battle.trainer_pokemon_killed,
            #'trainer_mon_killed': self.battle.trainer_pokemon_killed,
            #'gym_mon_killed': self.battle.gym_pokemon_killed,
            #'died': self.battle.died,
            'battle_action_avg': self.battle.get_avg_battle_action_avg(),
            'battle_turn_avg': self.battle.get_avg_battle_turn_avg(),
            'k/d': self.battle.get_kill_to_death(),
            'dmg_ratio': self.battle.get_damage_done_vs_taken(),
            'heal': len(self.map.pokecenter_history) - 1,
        })

    def _get_observation(self):
        self.map.update_map_obs()

        observation = {
            # Game View:
            "screen": self.map.screen,
            "visited": self.map.visited,
            "walkable": self.map.walkable,
            "action": self.gameboy.action_history,
            "coordinates": self.map.coordinates,

            # Game:
            "game_state": self.game.get_game_state(),

            # Player:
            "pokemon_lineup": self.game.player.get_player_lineup_arr(),
            "pokemon_health": self.game.player.get_player_health_arr(),
            "pokemon_stats": self.game.player.get_player_stats_arr(),
            #"badges": self.game.player.get_badges(),

            # Battle
            "battle_type": np.array([self.game.battle.get_battle_type()], dtype=np.uint8),
            "enemies_left": np.array([self.game.battle.get_battles_pokemon_left()], dtype=np.uint8),
            "player_stats": self.game.battle.get_player_fighting_pokemon_arr(),
            "enemy_stats": self.game.battle.get_enemy_fighting_pokemon_arr(),
            "turn_info": self.game.battle.get_battle_turn_info_arr(),
            "type_hint": np.array([self.game.battle.get_battle_type_hint()], dtype=np.uint8),
        }

        return observation
    
    '''            

            # Items
            "bag_ids": self.game.items.get_bag_item_ids(),
            "bag_quan": self.game.items.get_bag_item_quantities(),
            "pc_item_ids": self.game.items.get_pc_item_ids(),
            "pc_item_quan": self.game.items.get_pc_item_quantities(),
            "pc_pokemon": self.game.items.get_pc_pokemon_stored(),
            "item_selection_quan": self.game.items.get_item_quantity(),
            
            # World
            "milestones": self.game.world.get_game_milestones(),
            "audio": np.array([self.game.world.get_playing_audio_track()], dtype=np.uint8),
            "pokemart_items": self.game.world.get_pokemart_options(),

'''

    def _update_rewards(self, action):
        state_scores = {
            # 'pallet_town_explorer': self.support.map.tester.pallet_town_explorer_reward(),
            # 'pallet_town_point_nav': self.support.map.tester.pallet_town_point_nav(),
            'explore': self.map.get_exploration_reward(),
            'battle': self.battle.get_battle_win_reward(),
            'battle_turn': self.battle.get_battle_action_reward(),
            #'badges': self.player.get_badge_reward(),
            'pokecenter' : self.map.get_pokecenter_reward(),
        }

        # TODO: If pass in some test flag run just a single test reward
        self.total_reward = sum(val for _, val in state_scores.items())
