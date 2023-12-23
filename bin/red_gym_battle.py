# Assuming these constants are defined in red_env_constants
from red_env_constants import *
from ram_reader.red_memory_battle import *


class RedGymBattle:
    def __init__(self, env):
        if env.debug:
            print('**** RedGymBattle ****')

        self.env = env
        self.wild_pokemon_killed = 0
        self.trainer_pokemon_killed = 0
        self.gym_pokemon_killed = 0
        self.died = 0
        self.current_battle_action_cnt = 0
        self.total_battle_action_cnt = 0
        self.total_battle_turns = 0 
        self.total_battles = 0
        self.prev_move_in_battle = False
        self.battle_won = False
        self.battle_won_sticky = False
        self.total_party_hp_lost = 0
        self.total_enemy_hp_lost = 0
        self.last_party_head_hp = 0
        self.last_enemy_head_hp = 0


    def _clear_battle_stats(self):
            self.last_party_head_hp = 0
            self.last_enemy_head_hp = 0
            self.current_battle_action_cnt = 0
            self.prev_move_in_battle = False
            self.battle_won_sticky = False

    def _calc_battle_type_stats(self):
        battle_type = self.env.game.battle.get_battle_type()
        if battle_type == int(BattleTypes.WILD_BATTLE):
            self.wild_pokemon_killed += 1
        elif battle_type == int(BattleTypes.TRAINER_BATTLE):
            self.trainer_pokemon_killed += 1
        # TODO: Need to ID Gym Battle
        #elif battle_type == int(BattleTypes.GYM_BATTLE):
        #    self.gym_pokemon_killed += 1
        elif battle_type == int(BattleTypes.DIED):
            self.died += 1
        else:
            print(f'Unknown battle type: {battle_type}')

    def _inc_move_count(self):
        if not self.battle_won_sticky:
            self.current_battle_action_cnt += 1
            self.total_battle_action_cnt += 1

    def _inc_battle_counter(self):
        if not self.prev_move_in_battle:
            self.total_battles += 1
            self.prev_move_in_battle = True

    def _inc_hp_lost_vs_taken(self):
        if not self.env.game.battle.in_battle:
            return
        
        party_head_hp = self.env.game.battle.get_player_party_head_hp()
        enemy_head_hp = self.env.game.battle.get_enemy_head_pokemon_hp()

        if self.last_party_head_hp == 0:
            self.last_party_head_hp = party_head_hp
        if self.last_enemy_head_hp == 0:
            self.last_enemy_head_hp = enemy_head_hp

        if party_head_hp < self.last_party_head_hp:
            self.total_party_hp_lost += (self.last_party_head_hp - party_head_hp)
            self.last_party_head_hp = party_head_hp

        if enemy_head_hp < self.last_enemy_head_hp:
            self.total_enemy_hp_lost += (self.last_enemy_head_hp - enemy_head_hp)
            self.last_enemy_head_hp = enemy_head_hp

    def save_post_action_battle(self):
        if not self.env.game.battle.in_battle:
            self._clear_battle_stats()
            return

        self.battle_won = self.env.game.battle.win_battle()  # allows single occurrence won flag per battle, when enemy mon's hp all -> 0
        if self.battle_won:
            self.battle_won_sticky = True

        self._inc_move_count()
        self._inc_battle_counter()
        self._inc_hp_lost_vs_taken()

        # cal this way instead of w/ inc_move_count() b/c of long post battle text, which can count as still in battle
        if not self.battle_won:
            return
        
        # Won Battle falls though, to update total battle's stat's. This calc can only happen once per battle b/c of battle_won flag's design
        self._calc_battle_type_stats()
        self.total_battle_turns += self.env.game.battle.turns_in_current_battle

    def get_battle_reward(self):
        if not self.env.game.battle.in_battle:
            return 0
        elif not self.battle_won:
            return 0.1  # Being in a battle is 10x better than exploring prev explored tiles but 1/50th as good as finding a fresh tile
        
        # Won Battle falls though

        battle_type = self.env.game.battle.get_battle_type()
        if battle_type == int(BattleTypes.WILD_BATTLE):
            return 10 + (max(0, (150 - self.current_battle_action_cnt)) * 0.25)
        elif battle_type == int(BattleTypes.TRAINER_BATTLE):
            return 30 + (max(0, (150 - self.current_battle_action_cnt)) * 0.25)
        # TODO: Need to ID Gym Battle
        #elif battle_type == int(BattleTypes.GYM_BATTLE):
        #    return 600
        elif battle_type == int(BattleTypes.DIED):
            return -.5
        
        # TODO: Find what's getting through here
        self.env.support.save_and_print_info(False, True)
        return 0
        
    def get_avg_battle_action_avg(self):
        if self.total_battles == 0:
            return 0
        return self.total_battle_action_cnt / self.total_battles

    def get_avg_battle_turn_avg(self):
        if self.total_battles == 0:
            return 0
        return self.total_battle_turns / self.total_battles

    def get_kill_to_death(self):
        if self.died == 0:
            return 0
        return (self.wild_pokemon_killed + self.trainer_pokemon_killed + self.gym_pokemon_killed) / self.died
    
    def get_damage_done_vs_taken(self):
        if self.total_party_hp_lost == 0:
            return 0
        return self.total_enemy_hp_lost / self.total_party_hp_lost
