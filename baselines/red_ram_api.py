import numpy as np
from red_memory_battle import *
from red_memory_env import *
from red_memory_items import *
from red_memory_map import *
from red_memory_menus import *
from red_memory_player import *

from enum import Enum

# Assuming PyBoy is initialized elsewhere and imported here
from pyboy import PyBoy

class GameState(Enum):
    IN_BATTLE = 2
    TALKING = 4
    EXPLORING = 5
    FOLLOWING = 5
    ON_PC = 10
    IN_START_MENU = 11
    GAME_STATE_UNKNOWN = 99


class PyBoyRAMInterface:
    def __init__(self, pyboy):
        self.pyboy = pyboy

    def read_memory(self, address):
        return self.pyboy.get_memory_value(address)


class Battle:
    def __init__(self, pyboy):
        self.memory_interface = PyBoyRAMInterface(pyboy)

    def ram_battle_data(self):
        return np.array(np.zeros(1))

    def get_battle_data(self, game_state):
        if game_state.IN_BATTLE:
            return self.ram_battle_data()

        return np.array(np.zeros(1))

    def get_battle_state(self):
        # Trainer fight's IN_BATTLE lag's, CURRENT_OPPONENT instant. Poke/wild fights opp = 0 & in_btl is instant
        if self.memory_interface.read_memory(IN_BATTLE) or self.memory_interface.read_memory(CURRENT_OPPONENT):
            return GameState.IN_BATTLE

        return GameState.GAME_STATE_UNKNOWN


class Environment:
    def __init__(self, pyboy):
        self.memory_interface = PyBoyRAMInterface(pyboy)

    def get_text_box_state(self):
        # Text box's can be a substate of other things, like battles so usage order matters
        if self.memory_interface.read_memory(TEXT_ON_SCREEN):
            return GameState.TALKING

        return GameState.GAME_STATE_UNKNOWN

    def get_play_time_hours(self):
        return self.memory_interface.read_memory(0xDA41)


class Items:
    def __init__(self, pyboy):
        self.memory_interface = PyBoyRAMInterface(pyboy)


class Map:
    def __init__(self, pyboy):
        self.memory_interface = PyBoyRAMInterface(pyboy)

    def get_location_pos(self):
        return (self.memory_interface.read_memory(PLAYER_LOCATION_X),
                self.memory_interface.read_memory(PLAYER_LOCATION_Y))

    def get_location_map(self):
        return self.memory_interface.read_memory(PLAYER_MAP)


class Menus:
    def __init__(self, pyboy):
        self.memory_interface = PyBoyRAMInterface(pyboy)

    def get_party_count(self):
        return self.memory_interface.read_memory(0xD89C)


class Player:
    def __init__(self, pyboy):
        self.memory_interface = PyBoyRAMInterface(pyboy)

    def get_bag_items(self):
        # Assuming 0xC235 is the start of bag items in memory and you want to fetch a range of items
        return [self.memory_interface.read_memory(0xC235 + i) for i in range(number_of_items)]

