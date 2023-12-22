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
        self.battle_lengths = 0  # rolling avg
        self.battle_won = False


    def save_post_action_battle(self):
        self.battle_won = self.env.game.battle.win_battle()

        if self.battle_won:
            self.battle_lengths += self.env.game.battle.turns_in_current_battle

    def inc_move_count(self):
        if self.env.game.battle.in_battle:
            self.move_count += 1
        else:
            self.move_count = 0

    def get_battle_reward(self):
        if not self.env.game.battle.in_battle:
            return 0
        elif not self.battle_won:
            return 0.1  # Being in a battle is 2x better than exploring prev explored tiles but 1/50th as good as finding a fresh tile
        
        # Won Battle falls though

        battle_type = self.env.game.battle.get_battle_type()

        if battle_type == int(BattleTypes.WILD_BATTLE):
            self.wild_pokemon_killed += 1
            return 20 + (max(0, (105 - self.move_count)) * 0.1)
        elif battle_type == int(BattleTypes.TRAINER_BATTLE):
            self.trainer_pokemon_killed += 1
            return 300
        #elif battle_type == int(BattleTypes.GYM_BATTLE):
        #    self.gym_pokemon_killed += 1
        #    return 600
        elif battle_type == int(BattleTypes.DIED):
            self.died += 1
            return -.5
        else:
            print(f'Unknown battle type: {battle_type}')

    def get_avg_battle_length(self):
        total_fights = self.wild_pokemon_killed + self.trainer_pokemon_killed + self.gym_pokemon_killed

        return self.battle_lengths / total_fights

    def get_kill_to_death(self):
        return (self.wild_pokemon_killed + self.trainer_pokemon_killed + self.gym_pokemon_killed) / self.died
