# Assuming these constants are defined in red_env_constants
from cmath import exp
from math import floor
import math

import numpy as np
from red_env_constants import *
from ram_reader.red_memory_battle import *
from ram_reader.red_memory_menus import BATTLE_MENU_STATES, RedRamMenuValues



class BattleTurn:
    def __init__(self):
        self.menus_visited = {}  # Reward for visiting a menu for the first time each battle turn, dec static reward

class BattleMemory:
    def __init__(self):
        # Start of turn values
        self.player_pokemon = 0
        self.enemy_pokemon = 0
        self.player_modifiers_sum = 0
        self.enemy_modifiers_sum = 0
        self.player_hp_cur = 0
        self.player_hp_total = 0
        self.enemy_hp_cur = 0
        self.enemy_hp_total = 0
        self.player_status = 0
        self.enemy_status = 0
        self.type_hint = 0
        self.battle_turn = BattleTurn()


class RedGymBattle:
    def __init__(self, env):
        if env.debug:
            print('**** RedGymBattle ****')

        self.env = env
        self.wild_pokemon_killed = 0
        self.trainer_pokemon_killed = 0
        self.gym_pokemon_killed = 0
        self.total_battle_action_cnt = 0
        self.total_battle_turns = 0 
        self.total_battles = 0
        self.total_party_hp_lost = 0
        self.total_enemy_hp_lost = 0
        self.entering_battle = False
        self.after_battle = 0

        self._reset_battle_stats()


    def _reset_battle_stats(self):
        self.current_battle_action_cnt = 0
        self.battle_has_started = False
        self.battle_won = False
        self.last_party_head_hp = 0
        self.last_enemy_head_hp = 0
        self.player_pokemon_switch = False
        self.enemy_pokemon_switch = False

        self.pre_turn = None  # Don't use the space unless in battle
        self.post_turn = None  # Don't use the space unless in battle


    def _calc_battle_type_stats(self):
        battle_type = self.env.game.battle.get_battle_type()
        if battle_type == BattleTypes.WILD_BATTLE:
            self.wild_pokemon_killed += 1
        elif battle_type == BattleTypes.TRAINER_BATTLE:
            self.trainer_pokemon_killed += 1
        # TODO: Need to ID Gym Battle
        #elif battle_type == int(BattleTypes.GYM_BATTLE):
        #    self.gym_pokemon_killed += 1
        elif battle_type == BattleTypes.DIED:
            pass
        else:
            print(f'Unknown battle type: {battle_type}')

    def _inc_move_count(self):
        if not self.env.game.battle.battle_done:
            self.current_battle_action_cnt += 1
            self.total_battle_action_cnt += 1

    def _inc_battle_counter(self):
        if not self.battle_has_started:
            self.total_battles += 1
            self.battle_has_started = True

    def _inc_hp_lost_vs_taken(self):
        if not self.env.game.battle.in_battle:
            return

        if self.last_party_head_hp == 0:
            self.last_party_head_hp = self.post_turn.player_hp_cur
        if self.last_enemy_head_hp == 0:
            self.last_enemy_head_hp = self.post_turn.enemy_hp_cur

        if self.post_turn.player_hp_cur < self.last_party_head_hp:
            self.total_party_hp_lost += (self.last_party_head_hp - self.post_turn.player_hp_cur)
            self.last_party_head_hp = self.post_turn.player_hp_cur

        if self.post_turn.enemy_hp_cur < self.last_enemy_head_hp:
            self.total_enemy_hp_lost += (self.last_enemy_head_hp - self.post_turn.enemy_hp_cur)
            self.last_enemy_head_hp = self.post_turn.enemy_hp_cur

    def _calc_level_decay(self, avg_enemy_level, avg_player_lvl):
        POKEMON_BATTLE_LEVEL_FLOOR = 1
        level_delta = avg_player_lvl - avg_enemy_level
        if level_delta < POKEMON_BATTLE_LEVEL_FLOOR:
            return 1

        # view reward roll off: www.desmos.com/calculator
        result = math.exp(level_delta * -0.56) + 0.05

        if result > 1.0:
            return 1
        
        return result
        
    def _calc_avg_pokemon_level(self, pokemon):        
        avg_level, size = 0, 0
        for i, level in enumerate(pokemon):
            if level == 0:
                break

            avg_level += level
            size = i
        
        return avg_level / (size + 1)
    
    def _get_battle_turn_stats(self, turn):
        player_mods = sum(self.env.game.battle.get_player_party_head_modifiers())
        enemy_mods = sum(self.env.game.battle.get_enemy_party_head_modifiers())

        player_hp_total, player_hp_cur = self.env.game.battle.get_player_party_head_hp()
        enemy_hp_total, enemy_hp_cur = self.env.game.battle.get_enemy_party_head_hp()

        turn.player_pokemon = self.env.game.battle.get_player_head_index()
        turn.enemy_pokemon = self.env.game.battle.get_enemy_party_head_pokemon()
        turn.player_modifiers_sum = player_mods
        turn.enemy_modifiers_sum = enemy_mods
        turn.player_hp_total = player_hp_total
        turn.player_hp_cur = player_hp_cur
        turn.enemy_hp_total = enemy_hp_total
        turn.enemy_hp_cur = enemy_hp_cur
        turn.player_status = self.env.game.battle.get_player_party_head_status()
        turn.enemy_status = self.env.game.battle.get_enemy_party_head_status()
        turn.type_hint = self.env.game.battle.get_battle_type_hint()


    def _update_pre_turn_memory(self):
        if self.pre_turn == None:
            self.pre_turn = BattleMemory()

        self._get_battle_turn_stats(self.pre_turn)


    def _update_post_turn_memory(self):
        if self.post_turn == None:
            self.post_turn = BattleMemory()

        # post turn we may have just enter'd battle so we need pre_turn object
        if self.pre_turn == None:
            self.pre_turn = BattleMemory()

        self._get_battle_turn_stats(self.post_turn)


    def _update_menu_selected(self):
        if ((self.env.gameboy.a_button_selected() and self.env.game.game_state == self.env.game.GameState.BATTLE_TEXT) or
             self.env.game.game_state == self.env.game.GameState.BATTLE_ANIMATION):
            return
        
        selection_count = self.pre_turn.battle_turn.menus_visited.get(self.env.game.game_state.value, 0)
        self.pre_turn.battle_turn.menus_visited[self.env.game.game_state.value] = selection_count + 1
    
    def get_battle_decay(self):
        avg_enemy_level = self._calc_avg_pokemon_level(self.env.game.battle.get_enemy_lineup_levels())
        avg_player_lvl = self._calc_avg_pokemon_level(self.env.game.player.get_player_lineup_levels())
        return self._calc_level_decay(avg_enemy_level, avg_player_lvl)
    
    def save_pre_action_battle(self):
        if not self.env.game.battle.in_battle:
            return
        
        self.after_battle = 2  # 1 tick after battle, some items don't update until tick after battle
        
        self._update_pre_turn_memory()

    def _update_battle_pokemon_swaps(self):
        self.player_pokemon_switch = (self.post_turn.player_pokemon != self.pre_turn.player_pokemon)
        self.enemy_pokemon_switch = (self.post_turn.enemy_pokemon != self.pre_turn.enemy_pokemon)

    def save_post_action_battle(self):
        if not self.env.game.battle.in_battle and self.after_battle != 0:
            self._reset_battle_stats()
            self.after_battle -= 1
            return
        elif not self.env.game.battle.in_battle:
            return
        
        # IN BATTLE: Falls through

        self.battle_won = self.env.game.battle.win_battle()  # allows single occurrence won flag per battle, when enemy mon's hp all -> 0
        if self.battle_won:
            self.env.game.battle.battle_done = True   # TODO: The API handles setting this, back this out

        self._update_post_turn_memory()  # Order first in post_actions for dependency calc's

        self._inc_move_count()
        self._inc_battle_counter()
        self._inc_hp_lost_vs_taken()
        self._update_menu_selected()
        self._update_battle_pokemon_swaps()

        # cal this way instead of w/ inc_move_count() b/c of long post battle text, which can count as still in battle
        if not self.battle_won:
            return
        
        # Won Battle falls though, to update total battle's stat's. This calc can only happen once per battle b/c of battle_won flag's design
        self._calc_battle_type_stats()
        self.total_battle_turns += self.env.game.battle.turns_in_current_battle

    def get_battle_win_reward(self):
        if not self.env.game.battle.in_battle:
            return 0
        elif not self.battle_won:
            return 0.15
        
        # Won Battle falls though
        BATTLE_MOVE_CEILING = 500
        battle_type = self.env.game.battle.get_battle_type()
        if battle_type == BattleTypes.WILD_BATTLE:
            return 0
        elif battle_type == BattleTypes.TRAINER_BATTLE:
            pokemon_fought = self.env.game.battle.get_enemy_party_count()
            return max(350 * pokemon_fought, (BATTLE_MOVE_CEILING * pokemon_fought) - self.current_battle_action_cnt * 5) * (self.get_battle_decay() * 2)
        # TODO: Need to ID Gym Battle
        #elif battle_type == BattleTypes.GYM_BATTLE):
        #    return 600
        elif battle_type == BattleTypes.DIED:
            return 0
        
        self.env.support.save_and_print_info(False, True, True)
        assert(False), "Unknown battle type"
    
    def _pp_select_reward(self):
        pp_1, pp_2, pp_3, pp_4 = self.env.game.battle.get_player_party_head_pp()
        match self.env.game.game_state:
            case RedRamMenuValues.BATTLE_MOVE_1:
                return int(pp_1 == 0)
            case RedRamMenuValues.BATTLE_MOVE_2:
                return int(pp_2 == 0)
            case RedRamMenuValues.BATTLE_MOVE_3:
                return int(pp_3 == 0)
            case RedRamMenuValues.BATTLE_MOVE_4:
                return int(pp_4 == 0)

        return 0
    
    def _menu_selection_punish(self):
        selection_count = self.pre_turn.battle_turn.menus_visited.get(self.env.game.game_state.value, 0)
        if selection_count <= 1:
            return 0  # Don't reward new menu discovery or AI will farm menu hovering
        
        # TODO: Run in trainer battle not working, need to fix, no neg
        return max(-0.001 * pow(selection_count, 2), -0.15)
            
    def _get_battle_action_reward(self):
        if not self.env.gameboy.a_button_selected():
            return 0
        
        action_reward = 0
        action_reward += self._pp_select_reward() * -0.1

        return action_reward
    
    def _get_battle_hint_reward(self):
        type_hint_delta = self.post_turn.type_hint - self.pre_turn.type_hint  # pos good, neg bad

        if self.player_pokemon_switch or self.enemy_pokemon_switch:
            if type_hint_delta > 0:
                return 4
            elif type_hint_delta < 0:  # Discourage bad switches and switch cycling for point farming
                return -0.1
            
        return 0
        
    def _get_battle_stats_reward(self):
        if self.player_pokemon_switch or self.enemy_pokemon_switch:
            return 0
        
        multiplier = 1
        battle_type = self.env.game.battle.get_battle_type()
        if battle_type == BattleTypes.WILD_BATTLE:
            multiplier = 1

        player_modifiers_delta = self.post_turn.player_modifiers_sum - self.pre_turn.player_modifiers_sum  # pos good, neg bad
        enemy_modifiers_delta = self.post_turn.enemy_modifiers_sum - self.pre_turn.enemy_modifiers_sum  # pos bad, neg good
        player_hp_delta = self.post_turn.player_hp_cur - self.pre_turn.player_hp_cur  # pos good, neg bad
        enemy_hp_delta = self.post_turn.enemy_hp_cur - self.pre_turn.enemy_hp_cur # pos bad, neg good

        reward = 0

        if player_modifiers_delta > 0:
            reward += 15
        if enemy_modifiers_delta < 0:
            reward += 15
        if player_hp_delta > 0:
            reward += 250 * max((player_hp_delta / self.post_turn.player_hp_total), 0.375)
        if enemy_hp_delta < 0:
            reward += 75 * max((abs(enemy_hp_delta) / self.post_turn.enemy_hp_total), 0.375) * self.post_turn.type_hint * multiplier
        if self.post_turn.player_status == 0 and self.pre_turn.player_status != 0:
            reward += 200
        if self.post_turn.enemy_status != 0 and self.pre_turn.enemy_status == 0:
            reward += 200

        return reward
    
    def get_battle_action_reward(self):
        if not self.env.game.battle.in_battle or self.current_battle_action_cnt < 20 or self.battle_won:
            return 0
        
        selection_reward = self._menu_selection_punish()
        #print(f'\n\nMenu Selection Reward: {selection_reward} \n\n')
        #if reward < 0:
        #    return reward  # No decay for bad menu selections

        #reward += self._get_battle_action_reward()
        #print(f'Action Reward: {self._get_battle_action_reward()}')
        #hit_reward = self._get_battle_hint_reward()
        #print(f'Hint Reward: {hit_reward}')

        stats_reward = self._get_battle_stats_reward()
        #print(f'Stats Reward: {stats_reward}')

        return selection_reward + (stats_reward * self.get_battle_decay())

    def get_avg_battle_action_avg(self):
        if self.total_battles == 0:
            return 0
        return self.total_battle_action_cnt / self.total_battles

    def get_avg_battle_turn_avg(self):
        if self.total_battles == 0:
            return 0
        return self.total_battle_turns / self.total_battles

    def get_kill_to_death(self):
        died = self.env.player.died + 1

        return (self.wild_pokemon_killed + self.trainer_pokemon_killed + self.gym_pokemon_killed) / died
    
    def get_damage_done_vs_taken(self):
        if self.total_party_hp_lost == 0:
            return 0
        return self.total_enemy_hp_lost / self.total_party_hp_lost
    
    def obs_in_battle(self):
        return np.array([self.env.game.battle.in_battle], dtype=np.uint8)
    
    def obs_battle_type(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((4, ), dtype=np.uint8)
        
        battle_type = np.array(self.env.game.battle.get_battle_type(), dtype=np.uint8)
        binary_status = np.unpackbits(battle_type)[4:]

        return binary_status.astype(np.uint8)

    def obs_enemies_left(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((1, ), dtype=np.float32)

        return np.array([self.env.game.battle.get_battles_pokemon_left()], dtype=np.float32)

    def obs_player_head_index(self):
        if not self.env.game.battle.in_battle:  # TODO: What if mon fainted? Should show next avail mon in party
            return np.zeros((1, ), dtype=np.uint8)

        return np.array([self.env.game.battle.get_player_head_index()], dtype=np.uint8)
    
    def obs_player_head_pokemon(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((1, ), dtype=np.uint8)

        return np.array([self.env.game.battle.get_player_head_pokemon()], dtype=np.uint8)

    def obs_player_modifiers(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((6, ), dtype=np.float32)

        return self.env.support.normalize_np_array(np.array(self.env.game.battle.get_player_party_head_modifiers(), dtype=np.float32))

    def obs_enemy_head(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((1, ), dtype=np.uint8)

        return np.array([self.env.game.battle.get_enemy_party_head_pokemon()], dtype=np.uint8)

    def obs_enemy_level(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((1, ), dtype=np.float32)

        return self.env.support.normalize_np_array(np.array([self.env.game.battle.get_enemy_party_head_level()], dtype=np.float32) * 2)

    def obs_enemy_hp(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((2, ), dtype=np.float32)

        return self.env.support.normalize_np_array(np.array(self.env.game.battle.get_enemy_party_head_hp(), dtype=np.float32), False, 705.0)

    def obs_enemy_types(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((2, ), dtype=np.uint8)

        return np.array(self.env.game.battle.get_enemy_party_head_types(), dtype=np.uint8)

    def obs_enemy_modifiers(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((6, ), dtype=np.float32)

        return self.env.support.normalize_np_array(np.array(self.env.game.battle.get_enemy_party_head_modifiers(), dtype=np.float32))

    def obs_enemy_status(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((5, ), dtype=np.uint8)

        # https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_data_structure_(Generation_I)#Status_conditions
        # First 3 bits unused
        status = self.env.game.battle.get_enemy_party_head_status()
        status_array = np.array(status, dtype=np.uint8)
        binary_status = np.unpackbits(status_array)[3:8]
        return binary_status.astype(np.uint8)
    
    def obs_battle_moves_selected(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((2, ), dtype=np.uint8)

        return np.array(self.env.game.battle.get_battle_turn_moves(), dtype=np.uint8)

    def obs_type_hint(self):
        if not self.env.game.battle.in_battle:
            return np.zeros((4, ), dtype=np.uint8)
                
        hint = np.array(self.env.game.battle.get_battle_type_hint(), dtype=np.uint8)
        binary_status = np.unpackbits(hint)[4:]

        return binary_status.astype(np.uint8)
