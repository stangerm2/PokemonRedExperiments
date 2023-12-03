from os.path import exists
from pyboy import PyBoy
from pyboy import WindowEvent
from red_memory_map import *

from red_api import *

pyboy = PyBoy('PokemonRed.gb')
red_api = PokemonRedAPI(pyboy)
frame = 0

while not pyboy.tick():
    frame += 1

    if frame == 24:
        cur_x = pyboy.get_memory_value(PLAYER_LOCATION_X)
        cur_y = pyboy.get_memory_value(PLAYER_LOCATION_Y)
        map = pyboy.get_memory_value(PLAYER_MAP)

        wNorthConnectionStripSrc = pyboy.get_memory_value(0xD372)
        wNorthConnectionStripDest = pyboy.get_memory_value(0xD374)
        wNorthConnectionStripLength = pyboy.get_memory_value(0xD376)
        wNorthConnectedMapWidth = pyboy.get_memory_value(0xD377)
        wNorthConnectedMapYAlignment = pyboy.get_memory_value(0xD378)
        wNorthConnectedMapXAlignment = pyboy.get_memory_value(0xD379)

        wSouthhConnectionStripSrc = pyboy.get_memory_value(0xD37D)
        wSouthConnectionStripDest = pyboy.get_memory_value(0xD37F)
        wSouthConnectionStripLength = pyboy.get_memory_value(0xD381)
        wSouthConnectedMapWidth = pyboy.get_memory_value(0xD382)
        wSouthConnectedMapYAlignment = pyboy.get_memory_value(0xD383)
        wSouthConnectedMapXAlignment = pyboy.get_memory_value(0xD384)

        str = f'cur_x: {cur_x}, cur_y: {cur_y}, map: {map}'
        str1 = (f' wNorthConnectionStripSrc: {wNorthConnectionStripSrc},'
                f' wNorthConnectionStripDest: {wNorthConnectionStripDest},'
                f' wNorthConnectionStripLength: {wNorthConnectionStripLength},'
                f' wNorthConnectedMapWidth: {wNorthConnectedMapWidth},'
                f' wNorthConnectedMapYAlignment: {wNorthConnectedMapYAlignment},'
                f' wNorthConnectedMapXAlignment: {wNorthConnectedMapXAlignment}')
        str2 = (f' wSouthhConnectionStripSrc: {wSouthhConnectionStripSrc},'
                f' wSouthConnectionStripDest: {wSouthConnectionStripDest},'
                f' wSouthConnectionStripLength: {wSouthConnectionStripLength},'
                f' wSouthConnectedMapWidth: {wSouthConnectedMapWidth},'
                f' wSouthConnectedMapYAlignment: {wSouthConnectedMapYAlignment},'
                f' wSouthConnectedMapXAlignment: {wSouthConnectedMapXAlignment}')
        print(str)
        print(str1)
        print(str2)

        game_state = red_api.get_game_states()
        print(game_state)
        frame = 0

pyboy.stop()
