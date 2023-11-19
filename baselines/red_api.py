from red_ram_api import *

class PokemonRedAPI:
    def __init__(self, pyboy):
        self.battle = Battle(pyboy)
        self.environment = Environment(pyboy)
        self.items = Items(pyboy)
        self.map = Map(pyboy)
        self.menus = Menus(pyboy)
        self.player = Player(pyboy)


    def get_game_states(self):
        game_state = GameState.GAME_STATE_UNKNOWN

        # Battle: menu loc, poke stats
        game_state = self.battle.get_battle_state()
        if game_state != GameState.GAME_STATE_UNKNOWN:
            return game_state

        # Text: menu loc, poke stats
        game_state = self.environment.get_text_box_state()
        if game_state != GameState.GAME_STATE_UNKNOWN:
            return game_state