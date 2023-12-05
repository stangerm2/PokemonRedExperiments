import os
from os.path import exists
import sys
from pyboy import PyBoy, WindowEvent

from red_ram_api import *

pyboy = PyBoy('../../PokemonRed.gb')
game = Game(pyboy)
frame = 0

pyboy.set_emulation_speed(5)  # Configurable emulation speed


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
#print(str)
##print(str1)
#print(str2)

def _pokemon_dict_pretty_str(lineup):
        pokemons = ""
        for pokemon in lineup:
                for key, val in pokemon.items():
                        if key == 'pokemon':
                                pokemons += f'{key}: {POKEMON_LOOKUP.get(val, "None")}, '
                        else:
                                pokemons += f'{key}: {val}, '
                pokemons += '\n'
        return pokemons

def get_player_str():
        following = game.player.is_following_npc()
        print(f'following: {following}')

        pokemons = "Pokemon:\n" + _pokemon_dict_pretty_str(game.player.get_player_lineup_dict())

        money = game.player.get_player_money()
        pokedex_seen = game.player.get_pokedex_seen()
        pokedex_owned = game.player.get_pokedex_owned()
        badges = game.player.get_badges()

        return f'{pokemons}\nmoney: {money}, pokedex_seen: {pokedex_seen}, pokedex_owned: {pokedex_owned}, badges: {badges}'


def get_items_str():
        bag_ids = game.items.get_bag_item_ids()
        bag_quan = game.items.get_bag_item_quantities()
        pc_item_ids = game.items.get_pc_item_ids()
        pc_item_quan = game.items.get_pc_item_quantities()
        pc_pokemon_count = game.items.get_pc_pokemon_count()
        pc_pokemon_data = game.items.get_pc_pokemon_stored()
        item_quantity = game.items.get_item_quantity()

        return f'\n\nbag_ids: {bag_ids} \nbag_quan: {bag_quan} \npc_item_ids: {pc_item_ids} \npc_item_quan: {pc_item_quan} \npc_pokemon_count: {pc_pokemon_count} \npc_pokemon_data: {pc_pokemon_data}, \nitem_selection_quantity: {item_quantity}'

def get_world_str():
        milestones = game.world.get_game_milestones()
        audio = game.world.get_playing_audio_track()
        pokemart = game.world.get_pokemart_options()

        return f'\n\nmilestones: {milestones}, audio: {audio}, pokemart: {pokemart}'

def get_battle_str():
        in_battle = game.battle.is_in_battle()
        battle_type = game.battle.get_battle_type()
        player_stats = game.battle.get_player_fighting_pokemon_dict()
        enemy_stats = _pokemon_dict_pretty_str([game.battle.get_enemy_fighting_pokemon_dict()])
        turns = game.battle.get_battle_turn_info_dict()

        return f'\n\nin_battle: {in_battle}, battle_type: {battle_type} \n\nplayer_stats: \n{player_stats} \n\nenemy_stats: \n{enemy_stats} \n\nturns: {turns}'

def clear_screen():
        if os.name == 'nt':
                os.system('cls')
        else:
                os.system('clear')

count = 0
while not pyboy.tick():
        frame += 1

        if frame < 24:
                continue

        game_state = f'{game.get_game_states()}\n'
        game_state += f'Menu Allowed: {game.allow_menu_selection(WindowEvent.PRESS_BUTTON_A)}\n'

        game_state += get_player_str()
        game_state += get_items_str()
        game_state += get_world_str()
        game_state += get_battle_str()

        frame = 0

        clear_screen()
        sys.stdout.write(f'\r{game_state}')
        sys.stdout.flush()

pyboy.stop()
