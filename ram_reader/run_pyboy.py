from os.path import exists
from pyboy import PyBoy
from pyboy import WindowEvent

from red_api import *

pyboy = PyBoy('PokemonRed.gb')
red_api = PokemonRedAPI(pyboy)

while not pyboy.tick():
    game_state = red_api.get_game_states()
    print(game_state)

pyboy.stop()
