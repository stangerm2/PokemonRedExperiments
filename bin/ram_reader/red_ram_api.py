from red_memory_battle import *
from red_memory_env import *
from red_memory_items import *
from red_memory_map import *
from red_memory_menus import *
from red_memory_player import *

from enum import Enum
from pyboy import WindowEvent

'''
GPL-3.0 License:
Author: Matthew Stanger
Email: stangerm2@gmail.com
'''


class PyBoyRAMInterface:
    def __init__(self, pyboy):
        self.pyboy = pyboy

    def read_memory(self, address):
        return self.pyboy.get_memory_value(address)

    def write_memory(self, address, value):
        return self.pyboy.set_memory_value(address, value)
    

class Game:
    def __init__(self, pyboy):
        self.ram_interface = PyBoyRAMInterface(pyboy)

        self.world = World(self)
        self.battle = Battle(self)
        self.items = Items(self)
        self.map = Map(self)
        self.menus = Menus(self)
        self.player = Player(self)

        self.game_state = self.GameState.GAME_STATE_UNKNOWN

    class GameState(Enum):
        FILTERED_INPUT = 0x00
        IN_BATTLE = 0x01
        BATTLE_ANIMATION = 0x02
        # catch mon
        TALKING = 0x03
        EXPLORING = 0x04
        ON_PC = 0x05
        POKE_CENTER = 0x06
        MART = 0x07
        GYM = 0x08
        START_MENU = 0x09
        GAME_MENU = 0x0A
        BATTLE_TEXT = 0x0B
        FOLLOWING_NPC = 0x0C
        GAME_STATE_UNKNOWN = 0xFF

        SUB_MENU = RedRamMenuValues

    
    # Order of precedence is important here, we want to check for battle first, then menus
    def get_game_states(self):
        ORDERED_GAME_STATES = [
            self.battle.get_battle_state,
            self.player.is_following_npc,
            self.menus.get_menu_state,
            # TODO: Locations (mart, gym, pokecenter, etc.)
        ]

        for game_state in ORDERED_GAME_STATES:
            self.game_state = game_state()
            if self.game_state != self.GameState.GAME_STATE_UNKNOWN:
                return self.game_state
        
        return self.GameState.EXPLORING
        

    def allow_menu_selection(self, input):
        FILTERED_INPUTS = {
            RedRamMenuValues.START_MENU_POKEDEX: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.START_MENU_SELF: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.START_MENU_SAVE: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.START_MENU_OPTION: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.START_MENU_QUIT: {WindowEvent.PRESS_BUTTON_A},
            RedRamMenuValues.PC_OAK: {WindowEvent.PRESS_BUTTON_A},
            RedRamSubMenuValues.PC_SOMEONE_CONFIRM_STATS: {WindowEvent.PRESS_BUTTON_A},
            RedRamSubMenuValues.PC_SOMEONE_CHANGE_BOX: {WindowEvent.PRESS_BUTTON_A},
        }

        filtered_keys = FILTERED_INPUTS.get(self.game_state, None)
        if filtered_keys is None or input not in filtered_keys:
            return True

        return False


class World:
    def __init__(self, env):
        self.env = env
    
    def get_game_milestones(self):
        return [self.env.ram_interface.read_memory(item) for item in GAME_MILESTONES]
    
    def get_playing_audio_track(self):
        return self.env.ram_interface.read_memory(AUDIO_CURRENT_TRACK_NO_DELAY)
    
    def get_pokemart_options(self):
        mart = [0] * POKEMART_AVAIL_SIZE
        for i in range(POKEMART_AVAIL_SIZE):
            item = self.env.ram_interface.read_memory(POKEMART_ITEMS + i)
            if item == 0xFF:
                break

            mart[i] = item

        return mart


class Battle:
    def __init__(self, env):
        self.env = env
        self.in_battle = False
        self.turns_in_current_battle = 0
        self.last_turn_count = 0

    def _in_battle_state(self):
        if self.env.game_state in BATTLE_MENU_STATES or self.env.game_state == self.env.GameState.BATTLE_TEXT:
            return True
        return False

    def get_battle_state(self):
        self.in_battle = self.is_in_battle()
        in_pre_battle = self.is_in_pre_battle()

        if not (self.in_battle or in_pre_battle):
            self.turns_in_current_battle = 0
            self.last_turn_count = 0
            return self.env.GameState.GAME_STATE_UNKNOWN

        cursor_location, state = self.env.menus.get_item_menu_context()
        game_state = TEXT_MENU_CURSOR_LOCATIONS.get(cursor_location, RedRamMenuValues.UNKNOWN_MENU)

        # when text is on screen but menu reg's are clear, we can't be in a menu
        if cursor_location == RedRamMenuKeys.MENU_CLEAR or not self.in_battle:
            return self.env.GameState.BATTLE_ANIMATION
        elif game_state == RedRamMenuValues.MENU_YES or game_state == RedRamMenuValues.MENU_NO:
            return game_state
        elif self.env.ram_interface.read_memory(BATTLE_TEXT_PAUSE_FLAG) == 0x00:
            return self.env.GameState.BATTLE_TEXT

        if state != RedRamMenuValues.UNKNOWN_MENU:
            if self.env.menus.get_menu_item_state(cursor_location) != RedRamSubMenuValues.UNKNOWN_MENU:
                item_number = self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_COUNTER_1) + self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_COUNTER_2) + 1
                state = TEXT_MENU_ITEM_LOCATIONS.get(item_number, RedRamMenuValues.ITEM_RANGE_ERROR)

            return state


        return self.env.GameState.GAME_STATE_UNKNOWN
    
    def is_in_battle(self):
        return self.env.ram_interface.read_memory(IN_BATTLE)
    
    def is_in_pre_battle(self):
        return self.env.ram_interface.read_memory(CURRENT_OPPONENT)
    
    def get_battle_type(self):
        return self.env.ram_interface.read_memory(BATTLE_TYPE)
    
    def get_player_fighting_pokemon_arr(self):
        if not self.is_in_battle():
            return [0x00] * 7

        pokemon = self.env.ram_interface.read_memory(PLAYER_LOADED_POKEMON)
        # Redundant to Pokemon Party info
        # level = self.env.ram_interface.read_memory(PLAYERS_POKEMON_LEVEL)
        # hp = (self.env.ram_interface.read_memory(PLAYERS_POKEMON_HP[0]) << 8) + self.env.ram_interface.read_memory(PLAYERS_POKEMON_HP[1])
        # type = [self.env.ram_interface.read_memory(val) for val in PLAYERS_POKEMON_TYPES]
        # status = self.env.ram_interface.read_memory(PLAYERS_POKEMON_STATUS)
        attack_mod = self.env.ram_interface.read_memory(PLAYERS_POKEMON_ATTACK_MODIFIER)
        defense_mod = self.env.ram_interface.read_memory(PLAYERS_POKEMON_DEFENSE_MODIFIER)
        speed_mod =  self.env.ram_interface.read_memory(PLAYERS_POKEMON_SPEED_MODIFIER)
        accuracy_mod =  self.env.ram_interface.read_memory(PLAYERS_POKEMON_ACCURACY_MODIFIER)
        special_mod =  self.env.ram_interface.read_memory(PLAYERS_POKEMON_SPECIAL_MODIFIER)
        evasion_mod = self.env.ram_interface.read_memory(PLAYERS_POKEMON_SPECIAL_MODIFIER)

        return [pokemon, attack_mod, defense_mod, speed_mod, accuracy_mod, special_mod, evasion_mod]
    
    def get_player_fighting_pokemon_dict(self):
        player_fighting_pokemon = self.get_player_fighting_pokemon_arr()

        return {
            'pokemon': player_fighting_pokemon[0],
            'attack_mod': player_fighting_pokemon[1],
            'defense_mod': player_fighting_pokemon[2],
            'speed_mod': player_fighting_pokemon[3],
            'accuracy_mod': player_fighting_pokemon[4],
            'special_mod': player_fighting_pokemon[5],
            'evasion_mod': player_fighting_pokemon[6]
        }
    
    def get_enemy_fighting_pokemon_arr(self):
        if not self.is_in_battle():
            return [0x00] * 12
        
        party_count = self.env.ram_interface.read_memory(ENEMY_PARTY_COUNT)
        pokemon = self.env.ram_interface.read_memory(ENEMYS_POKEMON)
        level = self.env.ram_interface.read_memory(ENEMYS_POKEMON_LEVEL)
        hp = (self.env.ram_interface.read_memory(ENEMYS_POKEMON_HP[0]) << 8) + self.env.ram_interface.read_memory(ENEMYS_POKEMON_HP[1])
        type = [self.env.ram_interface.read_memory(val) for val in ENEMYS_POKEMON_TYPES]
        status = self.env.ram_interface.read_memory(ENEMYS_POKEMON_STATUS)
        attack_mod = self.env.ram_interface.read_memory(ENEMYS_POKEMON_ATTACK_MODIFIER)
        defense_mod = self.env.ram_interface.read_memory(ENEMYS_POKEMON_DEFENSE_MODIFIER)
        speed_mod =  self.env.ram_interface.read_memory(ENEMYS_POKEMON_SPEED_MODIFIER)
        accuracy_mod = self.env.ram_interface.read_memory(ENEMYS_POKEMON_ACCURACY_MODIFIER)
        special_mod =  self.env.ram_interface.read_memory(ENEMYS_POKEMON_SPECIAL_MODIFIER)
        evasion_mod = self.env.ram_interface.read_memory(ENEMYS_POKEMON_SPECIAL_MODIFIER)

        return [party_count, pokemon, level, hp, type, status, attack_mod, defense_mod, speed_mod, accuracy_mod, special_mod, evasion_mod]

    def get_enemy_fighting_pokemon_dict(self):
        enemy_fighting_pokemon = self.get_enemy_fighting_pokemon_arr()

        return {
            'party_count': enemy_fighting_pokemon[0],
            'pokemon': enemy_fighting_pokemon[1],
            'level': enemy_fighting_pokemon[2],
            'hp': enemy_fighting_pokemon[3],
            'type': enemy_fighting_pokemon[4],
            'status': enemy_fighting_pokemon[5],
            'attack_mod': enemy_fighting_pokemon[6],
            'defense_mod': enemy_fighting_pokemon[7],
            'speed_mod': enemy_fighting_pokemon[8],
            'accuracy_mod': enemy_fighting_pokemon[9],
            'special_mod': enemy_fighting_pokemon[10],
            'evasion_mod': enemy_fighting_pokemon[11]
        }


    def get_battle_turn_info_arr(self):
        if not self.is_in_battle():
            return [0x00] * 3

        turns_in_current_battle = self.env.ram_interface.read_memory(TURNS_IN_CURRENT_BATTLE)
        if turns_in_current_battle != self.last_turn_count:
            self.turns_in_current_battle += 1
            self.last_turn_count = turns_in_current_battle

        # turns_in_current_battle = self.env.ram_interface.read_memory(TURNS_IN_CURRENT_BATTLE)
        player_selected_move = self.env.ram_interface.read_memory(PLAYER_SELECTED_MOVE)
        enemy_selected_move = self.env.ram_interface.read_memory(ENEMY_SELECTED_MOVE)

        return [self.turns_in_current_battle, player_selected_move, enemy_selected_move]                                                                
    
    def get_battle_turn_info_dict(self):
        battle_turn_info = self.get_battle_turn_info_arr()

        return {
            'turns_in_current_battle': battle_turn_info[0],
            'player_selected_move': battle_turn_info[1],
            'enemy_selected_move': battle_turn_info[2]
        }


class Items:
    def __init__(self, env):
        self.env = env

    def _get_items_in_range(self, size, index, offset):
        items = [None] * size
        for i in range(size):
            item_val = self.env.ram_interface.read_memory(index + i * offset)
            items[i] = 0 if item_val == 0xFF else item_val  # Modern parsing, we don't need termination byte so strip it out
        return items
    
    def get_bag_item_count(self):
        return self.env.ram_interface.read_memory(BAG_TOTAL_ITEMS)

    def get_bag_item_ids(self):
        return self._get_items_in_range(BAG_SIZE, BAG_ITEMS_INDEX, ITEMS_OFFSET)

    def get_bag_item_quantities(self):
        return [self.env.ram_interface.read_memory(BAG_ITEM_QUANTITY_INDEX + i * ITEMS_OFFSET) for i in range(BAG_SIZE)]

    def get_pc_item_ids(self):
        return self._get_items_in_range(STORAGE_SIZE, PC_ITEMS_INDEX, ITEMS_OFFSET)
    
    def get_pc_item_quantities(self):
        return [self.env.ram_interface.read_memory(PC_ITEM_QUANTITY_INDEX + i * ITEMS_OFFSET) for i in range(STORAGE_SIZE)]
    
    def get_pc_pokemon_count(self):
        return self.env.ram_interface.read_memory(BOX_POKEMON_COUNT)
    
    def get_pc_pokemon_stored(self):
        return [(self.env.ram_interface.read_memory(BOX_POKEMON_1 + i * BOX_OFFSET),
                  self.env.ram_interface.read_memory(BOX_POKEMON_1_LEVEL + i * BOX_OFFSET)) for i in range(BOX_SIZE)]

    def get_item_quantity(self):
        # TODO: need to map sub menu state for buy/sell count
        if self.env.game_state != RedRamMenuValues.ITEM_QUANTITY:
            return 0
        
        return self.env.ram_interface.read_memory(ITEM_SELECTION_QUANTITY)
            

class Map:
    def __init__(self, env):
        self.env = env

    def get_location_pos(self):
        return (self.env.ram_interface.read_memory(PLAYER_LOCATION_X),
                self.env.ram_interface.read_memory(PLAYER_LOCATION_Y))

    def get_location_map(self):
        return self.env.ram_interface.read_memory(PLAYER_MAP)


class Menus:
    def __init__(self, env):
        self.env = env

    def get_item_menu_context(self):
        cursor_location = (self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_LOCATION[0]),
                    self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_LOCATION[1]))
        return cursor_location, TEXT_MENU_CURSOR_LOCATIONS.get(cursor_location, RedRamMenuValues.UNKNOWN_MENU)

    def get_menu_state(self):
        text_on_screen = self.env.ram_interface.read_memory(TEXT_FONT_ON_LOADED)
        if text_on_screen:
            cursor_location, state = self.get_item_menu_context()

            # when text is on screen but menu reg's are clear, we can't be in a menu
            if cursor_location == RedRamMenuKeys.MENU_CLEAR:
                return self.env.GameState.TALKING
            
            # In a sub-box that requires fetching count of menu pos, such as mart items
            sub_state = self.get_menu_item_state(cursor_location)
            if sub_state != RedRamSubMenuValues.UNKNOWN_MENU:
                return sub_state

            # check the bigger of the two submenu's, they have the same val's, to see if we are in a submenu
            sub_state = self._get_sub_menu_state(cursor_location)
            if sub_state != RedRamSubMenuValues.UNKNOWN_MENU:
                return sub_state

            return state
        else:
            self.env.ram_interface.write_memory(TEXT_MENU_CURSOR_LOCATION[0], 0x00)
            self.env.ram_interface.write_memory(TEXT_MENU_CURSOR_LOCATION[1], 0x00)
            for i in range(POKEMART_AVAIL_SIZE):
                self.env.ram_interface.write_memory(POKEMART_ITEMS + i, 0x00)

        return self.env.GameState.GAME_STATE_UNKNOWN
    
    def _get_sub_menu_state(self, cursor_location):
        if PC_POKE_MENU_CURSOR_LOCATIONS.get(cursor_location, RedRamSubMenuValues.UNKNOWN_MENU) == RedRamSubMenuValues.UNKNOWN_MENU:
            return RedRamSubMenuValues.UNKNOWN_MENU

        # Peek at screen memory to detect submenu's which have hard coded menu renderings w/ diff's between them. Reverse engineered.
        pc_menu_screen_peek = self.env.ram_interface.read_memory(PC_SUB_MENU_SCREEN_PEEK)

        # pokemon pc sub menu
        if pc_menu_screen_peek == 0x91:
            if cursor_location != RedRamSubMenuKeys.SUB_MENU_6:  # menu 6 is the same for deposit and withdraw so we have to normalize it
                return PC_POKE_MENU_CURSOR_LOCATIONS.get(cursor_location, RedRamSubMenuValues.UNKNOWN_MENU)
            else:
                pc_menu_screen_peek = self.env.ram_interface.read_memory(PC_SUB_MENU_DEPO_WITH_SCREEN_PEEK)
                return RedRamSubMenuValues.PC_SOMEONE_CONFIRM_WITHDRAW if pc_menu_screen_peek == 0x91 else RedRamSubMenuValues.PC_SOMEONE_CONFIRM_DEPOSIT
            
        # item pc sub menu
        elif pc_menu_screen_peek == 0x93:
            return PC_ITEM_MENU_CURSOR_LOCATIONS.get(cursor_location, RedRamSubMenuValues.UNKNOWN_MENU)
        
        return RedRamSubMenuValues.UNKNOWN_MENU

    def get_menu_item_state(self, cursor_location):
        if cursor_location == RedRamMenuKeys.BATTLE_MART_PC_ITEM_1 or cursor_location == RedRamMenuKeys.BATTLE_MART_PC_ITEM_2 or cursor_location == RedRamMenuKeys.BATTLE_MART_PC_ITEM_N:
            item_number = self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_COUNTER_1) + self.env.ram_interface.read_memory(TEXT_MENU_CURSOR_COUNTER_2) + 1
            if self.env.ram_interface.read_memory(ITEM_COUNT_SCREEN_PEAK) == 0x7E:  # 0x7E is the middle pokeball icon on screen, unique to the 3 sub menu pop out
                return RedRamMenuValues.ITEM_QUANTITY
            
            # self.env.ram_interface.write_memory(ITEM_SELECTION_QUANTITY, 0x00)
            
            return TEXT_MENU_ITEM_LOCATIONS.get(item_number, RedRamMenuValues.ITEM_RANGE_ERROR)
        
        return RedRamSubMenuValues.UNKNOWN_MENU
    

class Pokemon:
    def __init__(self, ram_interface, party_index=0):
        offset = party_index * PARTY_OFFSET

        self.pokemon = ram_interface.read_memory(POKEMON_1 + offset)
        self.level = ram_interface.read_memory(POKEMON_1_LEVEL_ACTUAL + offset)
        self.type = [ram_interface.read_memory(val + offset) for val in POKEMON_1_TYPES] 
        self.hp_total = (ram_interface.read_memory(POKEMON_1_MAX_HP[0] + offset) << 8) + ram_interface.read_memory(POKEMON_1_MAX_HP[1] + offset)
        self.hp_avail = (ram_interface.read_memory(POKEMON_1_CURRENT_HP[0] + offset) << 8) + ram_interface.read_memory(POKEMON_1_CURRENT_HP[1] + offset)
        self.xp = (ram_interface.read_memory(POKEMON_1_EXPERIENCE[0] + offset) << 16) + (ram_interface.read_memory(POKEMON_1_EXPERIENCE[1] + offset) << 8) + ram_interface.read_memory(POKEMON_1_EXPERIENCE[2] + offset)
        self.moves = [ram_interface.read_memory(val + offset) for val in POKEMON_1_MOVES]
        self.pp = [ram_interface.read_memory(val + offset) for val in POKEMON_1_PP_MOVES] 
        self.attack = (ram_interface.read_memory(POKEMON_1_ATTACK[0] + offset) << 8) + ram_interface.read_memory(POKEMON_1_ATTACK[1] + offset)
        self.defense = (ram_interface.read_memory(POKEMON_1_DEFENSE[0] + offset) << 8) + ram_interface.read_memory(POKEMON_1_DEFENSE[1] + offset)
        self.speed = (ram_interface.read_memory(POKEMON_1_SPEED[0] + offset) << 8) + ram_interface.read_memory(POKEMON_1_SPEED[1] + offset)
        self.special = (ram_interface.read_memory(POKEMON_1_SPECIAL[0] + offset) << 8) + ram_interface.read_memory(POKEMON_1_SPECIAL[1] + offset)
        self.health_status = ram_interface.read_memory(POKEMON_1_STATUS + offset)

    def get_pokemon_data_dict(self):
        return {
            'pokemon': self.pokemon,
            'level': self.level,
            'type': self.type,
            'hp_total': self.hp_total,
            'hp_avail': self.hp_avail,
            'xp': self.xp,
            'moves': self.moves,
            'pp': self.pp,
            'attack': self.attack,
            'defense': self.defense,
            'speed': self.speed,
            'special': self.special,
            'health_status': self.health_status
        }
        
    def get_pokemon_data_arr(self):
        return [self.id, self.level, self.type, self.hp_total, self.hp_avail, self.xp, self.moves, self.pp, self.attack, self.defense, self.speed, self.special, self.health_status]


class Player:
    def __init__(self, env):
        self.env = env

    def _get_player_lineup(self):
        lineup = [None] * 6
        for i in range(len(POKEMON_PARTY)):
            lineup[i] = Pokemon(self.env.ram_interface, i)

        return lineup
    
    def _pokedex_bit_count(self, pokedex_address):
        bit_count = 0
        for i in range(POKEDEX_ADDR_LENGTH):
            binary_value = bin(self.env.ram_interface.read_memory(pokedex_address + i))
            bit_count += binary_value.count('1')

        return bit_count

    def get_player_lineup_dict(self):
        return [pokemon.get_pokemon_data_dict() for pokemon in self._get_player_lineup()]

    def get_player_lineup_arr(self):
        return [pokemon.get_pokemon_data_arr() for pokemon in self._get_player_lineup()]
    
    def is_following_npc(self):
        if self.env.ram_interface.read_memory(FOLLOWING_NPC_FLAG) != 0x00:
            return self.env.GameState.FOLLOWING_NPC
        
        return self.env.GameState.GAME_STATE_UNKNOWN
    
    def get_badges(self):
        return self.env.ram_interface.read_memory(OBTAINED_BADGES)
    
    def get_pokedex_seen(self):
        return self._pokedex_bit_count(POKEDEX_SEEN)
    
    def get_pokedex_owned(self):
        return self._pokedex_bit_count(POKEDEX_OWNED)
    
    def get_player_money(self):
        # Trigger warning, money is a base16 literal as base 10 numbers
        money_bytes = [self.env.ram_interface.read_memory(addr) for addr in PLAYER_MONEY]
        money_hex = ''.join([f'{byte:02x}' for byte in money_bytes])
        money_decimal = int(money_hex, 10)
        return money_decimal
    
    

    
