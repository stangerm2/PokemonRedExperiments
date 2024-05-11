import numpy as np
from red_env_constants import *
from ram_reader.red_memory_items import *
from ram_reader.red_memory_battle import BattleTypes


class RedGymPlayer:
    def __init__(self, env):
        self.env = env
        if env.debug:
            print('**** RedGymPlayer ****')
        
        self.current_badges = 0
        self.bag_items = {}
        self.bank_items = {}
        self.money = 0
        self.died = 0
        self.dead = False
        self.item_decay = 0
        self.items_found = 0
        self.items_bought = 0
        self.items_sold = 0
        self.mart_reward = False


    def _inc_died(self):
        if self.env.game.battle.get_battle_type() == BattleTypes.DIED or self.env.game.player.is_player_dead():
            if not self.dead:
                self.died += 1
                self.dead = True
                return
        else:
            self.dead = False

    def _lookup_player_items(self, item_ids, item_counts):
        items = {}
        for i in range(len(item_ids)):
            if item_counts[i] == 0:
                break

            items[item_ids[i]] = item_counts[i]

        return items
    
    def _dec_item_decay(self):
        if self.item_decay == 0:
            return
        
        self.item_decay -= 1

    def _do_item_decay(self):
        # A linear decay from x = 0 to 50, where y is bound between 1 to 0
        decay = max(1 - 0.02 * self.item_decay, 0)
        self.item_decay += 25

        return decay
    
    def _get_player_money(self):
        return self.env.game.player.get_player_money()
    
    def _find_item_in_bag(self, bag1, bag2):
        for key, value in bag1.items():
            if value != bag2.get(key, 0):
                return key
    
    def save_post_action_player(self):
        self._inc_died()
    
    def get_item_reward(self):
        if self.env.game.battle.in_battle or self.env.battle.after_battle != 0:
            return 0

        bag_item_counts = self.env.game.items.get_bag_item_quantities()
        pc_item_counts = self.env.game.items.get_pc_item_quantities()

        # Prevent reward by shuffling items around in bank & bag
        cur_sum_bag_items, prev_sum_bag_items = sum(bag_item_counts), sum(self.bag_items.values())
        cur_total_items = cur_sum_bag_items + sum(pc_item_counts)
        prev_total_items = prev_sum_bag_items + sum(self.bank_items.values())

        # No items in bags changed
        if cur_total_items == prev_total_items:
            return 0
        
        # Reward for gaining items, ignore using/selling items here
        item_changed = None
        bag_item_delta = (cur_sum_bag_items - prev_total_items) + 1
        bag_item_ids = self.env.game.items.get_bag_item_ids()
        bag_contents = dict(zip(bag_item_ids, bag_item_counts))
        if bag_item_delta > 0:
            item_changed = self._find_item_in_bag(bag_contents, self.bag_items)
        else:
            item_changed = self._find_item_in_bag(self.bag_items, bag_contents)


        #print("\n\nbag ids now:", bag_item_ids)
        #print("bag counts now:", bag_item_counts)
        #print("bag prev:", self.bag_items)
        #print("item changed: ", ITEM_LOOKUP[item_changed])
        #print("item delta, ", bag_item_delta)

        item_norm = (bag_item_delta * ITEM_COSTS.get(item_changed, 0)) / 100

        # Handle using items in battle
        # if self.env.game.battle.in_battle or self.env.battle.after_battle != 0:
        #    battle_type = self.env.game.battle.get_battle_type()
        #    if battle_type == BattleTypes.TRAINER_BATTLE and item_changed != ITEMS_HEX.Pokeball:
        #        return 100
        #    elif battle_type == BattleTypes.WILD_BATTLE:
        ##        return 10
        #    else:
        #        return 0

        # Don't reward selling items, money is selling reward
        if bag_item_delta < 0:
            self.items_sold -= bag_item_delta
            item_norm /= 2
            return 0
        
        # TODO: WIP testing in mart, build tracking into API w/ keys off nurse and sound
        audio_track = self.env.game.world.get_playing_audio_track()
        if audio_track != 0xBD:
            self.items_found += 1
            return 70
        #elif self.mart_reward == False:
        #    return 0
        
        self.mart_reward = False
        
        #print('\nitem_norm ', item_norm)
        #print('item_decay cnt ', self.item_decay)

        # item_norm could be pos bought item or neg sold item but both are good rewards, using should always win over buy/sell loops b/c selling halves money value
        self.items_bought += bag_item_delta
        item_decay = self._do_item_decay()
        if item_norm < 0:
            item_decay = 1 - item_decay

        #print('item_decay ', item_decay)

        return item_norm * item_decay * 15
    

    def get_badge_reward(self):
        badges = self.env.game.player.get_badges()
        if badges > self.current_badges:
            self.mart_reward = True
            self.current_badges = badges
            return 2000
            
        return 0
    
    def get_money_reward(self):
        cur_money = self._get_player_money()
        money_delta = cur_money - self.money

        return money_delta / 100  # was 10
    
    def save_pre_action_player(self):
        self.bag_items = self._lookup_player_items(self.env.game.items.get_bag_item_ids(), self.env.game.items.get_bag_item_quantities())
        self.bank_items = self._lookup_player_items(self.env.game.items.get_pc_item_ids(), self.env.game.items.get_pc_item_quantities())
        self.money = self._get_player_money()
        self._dec_item_decay()

    def obs_player_pokemon(self):
        pokemon_array = np.array(self.env.game.player.get_player_lineup_pokemon(), dtype=np.uint8)
        return np.pad(pokemon_array, ((0, 6 - len(pokemon_array))), mode='constant')
    
    def obs_player_levels(self):
        levels_array = np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_levels()), dtype=np.float32)
        padded_levels = np.pad(levels_array, (0, 6 - len(levels_array)), mode='constant')
        return padded_levels
    
    def obs_player_types(self):
        types_array = np.array(self.env.game.player.get_player_lineup_types(), dtype=np.uint8).flatten()
        padded_types = np.pad(types_array, (0, 12 - len(types_array)), constant_values=0)
        return padded_types
    
    def obs_player_health(self):
        health_array = np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_health(), False, 705), dtype=np.float32).flatten()
        padded_health = np.pad(health_array, (0, 12 - len(health_array)), mode='constant')
        return padded_health
    
    def obs_player_moves(self):
        moves_array = np.array(self.env.game.player.get_player_lineup_moves(), dtype=np.uint8).flatten()
        padded_moves = np.pad(moves_array, (0, 24 - len(moves_array)), constant_values=0)
        return padded_moves
    
    def obs_player_xp(self):
        xp_array = np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_xp(), False, 250000), dtype=np.float32)
        padded_xp = np.pad(xp_array, (0, 6 - len(xp_array)), mode='constant')
        return padded_xp
    
    def obs_player_pp(self):
        pp_array = np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_pp()), dtype=np.float32).flatten()
        padded_pp = np.pad(pp_array, (0, 24 - len(pp_array)), mode='constant')
        return padded_pp
    
    def obs_player_stats(self):
        stats_array = np.array(self.env.support.normalize_np_array(self.env.game.player.get_player_lineup_stats()), dtype=np.float32).flatten()
        padded_stats = np.pad(stats_array, (0, 24 - len(stats_array)), mode='constant')
        return padded_stats
    
    def obs_player_status(self):
        # https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_data_structure_(Generation_I)#Status_conditions
        # First 3 bits unused
        status = self.env.game.player.get_player_lineup_status()
        status_array = np.array(status, dtype=np.uint8)

        binary_status = np.zeros(30, dtype=np.uint8)  # 6 pokemon * 5 status bits
        for i, status in enumerate(status_array):
            binary_status[i*5:(i+1)*5] = np.unpackbits(status)[3:8]

        return binary_status

    def obs_total_badges(self):
        badges = self.env.game.player.get_badges()
        badges_array = np.array(badges, dtype=np.uint8)
        binary_badges = np.unpackbits(badges_array)[0:8]
        return binary_badges.astype(np.uint8)
    
    def obs_bag_ids(self):
        bag_item_ids = self.env.game.items.get_bag_item_ids()
        padded_ids = np.pad(bag_item_ids, (0, 20 - len(bag_item_ids)), constant_values=0)
        return np.array(padded_ids, dtype=np.uint8)
    
    def obs_bag_quantities(self):
        return self.env.support.normalize_np_array(self.env.game.items.get_bag_item_quantities())
    
    def obs_total_money(self):
        return self.env.support.normalize_np_array(np.array([self.money], dtype=np.float32), False, 200000)
