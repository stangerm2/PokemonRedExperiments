import numpy as np
from collections import deque

# Assuming these constants are defined in red_env_constants
from red_env_constants import *
from red_gym_obs_tester import RedGymObsTester


class RedGymMap:
    def __init__(self, env):
        if env.debug:
            print('**** RedGymMap ****')

        self.env = env
        self.x_pos_org, self.y_pos_org, self.n_map_org = 0, 0, 0
        self.map_ptr_org = 0x00
        self.visited_pos = {}
        self.visited_pos_order = deque()
        self.map_loading = False
        self.new_map = False
        self.discovered_map = False
        self.moved_location = False  # indicates if the player moved 1 or more spot
        self.location_history = deque()
        self.steps_discovered = 0
        self.collisions = 0
        self.collisions_lookup = {}
        self.visited_maps = set()

        self.visited = np.zeros((1, SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.simple_screen = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.simple_screen_channels = np.zeros((11, SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.coordinates = np.zeros((3, BITS_PER_BYTE), dtype=np.uint8)  # x,y,map stacked
        self.connections = np.zeros((4, ), dtype=np.float32)  #  distance to each map connection north, south, east, west
        #self.tester = RedGymObsTester(self)

        self.coords =  np.zeros((3, ), dtype=np.uint8)



    def _clear_map_obs(self):
        self.visited = np.zeros((1, SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.simple_screen = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.simple_screen_channels = np.zeros((11, SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.coordinates = np.zeros((3, BITS_PER_BYTE), dtype=np.uint8)

        self.coords = np.zeros((3, ), dtype=np.uint8)



    def _update_collision_lookup(self, collision_ptr):
        if collision_ptr in self.collisions_lookup:
            return
        
        collection_tiles = self.env.game.map.get_collision_tiles()
        self.collisions_lookup[collision_ptr] = collection_tiles


    def _update_simple_screen_obs(self, x_pos_new, y_pos_new, n_map_new):
        collision_ptr = self.env.game.map.get_collision_pointer()
        self._update_collision_lookup(collision_ptr)

        # Extract the 7x7 matrix from the center of the bottom_left_screen_tiles
        top_left_tiles, bottom_left_tiles = self.env.game.map.get_screen_tilemaps()
        bottom_left_tiles_7x7 = bottom_left_tiles[1:1+SCREEN_VIEW_SIZE, 1:1+SCREEN_VIEW_SIZE]
        top_left_tiles_7x7 = top_left_tiles[1:1+SCREEN_VIEW_SIZE, 1:1+SCREEN_VIEW_SIZE]

        tileset_index = self.env.game.map.get_tileset_index()
        sprites = self.env.game.map.get_npc_location_dict()
        warps = self.env.game.map.get_warp_tile_positions()

        callback = lambda x, y, pos: self._walk_simple_screen(x, y, pos, collision_ptr, tileset_index, sprites, warps, bottom_left_tiles_7x7, top_left_tiles_7x7)
        self._walk_screen(x_pos_new, y_pos_new, n_map_new, callback)


    def _update_visited_obs(self, x_pos_new, y_pos_new, n_map_new):
        callback = lambda x, y, pos: self._walk_visited_screen(x, y, pos)
        self._walk_screen(x_pos_new, y_pos_new, n_map_new, callback)

        # DO NOT set cur pos as visited on the obs until the next turn, it REALLY helps the AI
        # ie.. self.visited[3][3] = 0 (this is intentional)


    def _update_pos_obs(self, x_pos_new, y_pos_new, n_map_new):
        try:
            x_pos_binary = format(x_pos_new, f'0{BITS_PER_BYTE}b')
            y_pos_binary = format(y_pos_new, f'0{BITS_PER_BYTE}b')
            m_pos_binary = format(n_map_new, f'0{BITS_PER_BYTE}b')

            self.coords = [x_pos_new, y_pos_new, n_map_new]
        
            # appends the x,y, pos binary form to the bottom of the screen and visited matrix's
            for i, bit in enumerate(x_pos_binary):
                self.coordinates[0][i] = bit

            for i, bit in enumerate(y_pos_binary):
                self.coordinates[1][i] = bit

            for i, bit in enumerate(m_pos_binary):
                self.coordinates[2][i] = bit

        except Exception as e:
            print(f"An error occurred: {e}")
            self.env.support.save_and_print_info(False, True, True)
            self.env.support.save_debug_string("An error occurred: {e}")
            assert(True)


    def _walk_screen(self, x_pos_new, y_pos_new, n_map_new, callback):
        center_x = center_y = SCREEN_VIEW_SIZE // 2

        for y in range(SCREEN_VIEW_SIZE):
            for x in range(SCREEN_VIEW_SIZE):
                center_x = center_y = SCREEN_VIEW_SIZE // 2
                x_offset = x - center_x
                y_offset = y - center_y
                current_pos = x_pos_new + x_offset, y_pos_new + y_offset, n_map_new

                callback(x, y, current_pos)


    def _walk_visited_screen(self, x, y, pos):
        if pos in self.visited_pos:
            self.visited[0][y][x] = 0
        else:
            self.visited[0][y][x] = 1


    def _update_tileset_openworld(self, bottom_left_tiles_7x7, x, y):
        if bottom_left_tiles_7x7[y][x] == 0x36 or bottom_left_tiles_7x7[y][x] == 0x37:  # Jump Down Ledge
            self.simple_screen[y][x] = 6
        elif bottom_left_tiles_7x7[y][x] == 0x27:  # Jump Left Ledge
            self.simple_screen[y][x] = 7
        elif bottom_left_tiles_7x7[y][x] == 0x1D:  # Jump Right Ledge
            self.simple_screen[y][x] = 8
        elif bottom_left_tiles_7x7[y][x] == 0x52:  # Grass
            self.simple_screen[y][x] = 2
        elif bottom_left_tiles_7x7[y][x] == 0x14:  # Water
            self.simple_screen[y][x] = 3
        elif bottom_left_tiles_7x7[y][x] == 0x3D:  # Tree
            self.simple_screen[y][x] = 10


    def _update_tileset_cave(self, x, y, bottom_left_tiles_7x7, tiles_top_left):
        if tiles_top_left[y][x] == 0x29:  # One Pixel Wall Tile (NOTE: Top Left tile contains the tile identifier)
            self.simple_screen[y][x] = 5
        elif bottom_left_tiles_7x7[y][x] == 0x14:  # Water
            self.simple_screen[y][x] = 3
        elif bottom_left_tiles_7x7[y][x] == 0x20 or bottom_left_tiles_7x7[y][x] == 0x05 or bottom_left_tiles_7x7[y][x] == 0x15:  # Cave Ledge, Floor or Stairs
            self.simple_screen[y][x] = 2


    def _update_tileset_cemetery(self, x, y, bottom_left_tiles_7x7):
        if bottom_left_tiles_7x7[y][x] == 0x01:  # Cemetery Floor
            self.simple_screen[y][x] = 2


    def _update_tileset_forest(self, x, y, bottom_left_tiles_7x7):
        if bottom_left_tiles_7x7[y][x] == 0x20:  # Grass
            self.simple_screen[y][x] = 2
    

    def _update_matrix_with_npcs(self, x, y, pos, sprites):
        if pos in sprites:
            self.simple_screen[y][x] = 9


    def _update_matrix_with_warps(self, x, y, pos, warps):
        location = (pos[0], pos[1])
        if self.simple_screen[y][x] != 0 and location in warps:
            self.simple_screen[y][x] = 4


    def _walk_simple_screen(self, x, y, pos, collision_ptr, tileset_index, sprites, warps, bottom_left_tiles_7x7, top_left_tiles_7x7):
        if bottom_left_tiles_7x7[y][x] in self.collisions_lookup[collision_ptr]:
            self.simple_screen[y][x] = 1  # Walkable
        else:
            self.simple_screen[y][x] = 0  # Wall

        if tileset_index == 0x00:
            self._update_tileset_openworld(bottom_left_tiles_7x7, x, y)
        elif tileset_index == 0x11:
            self._update_tileset_cave(x, y, bottom_left_tiles_7x7, top_left_tiles_7x7)
        elif tileset_index == 0x0F:
            self._update_tileset_cemetery(x, y, bottom_left_tiles_7x7)
        elif tileset_index == 0x03:
            self._update_tileset_forest(x, y, bottom_left_tiles_7x7)

        self._update_matrix_with_npcs(x, y, pos, sprites)
        self._update_matrix_with_warps(x, y, pos, warps)

    def _update_simple_screen_channel_obs(self):
        self.simple_screen_channels = np.zeros((11, SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        for y in range(SCREEN_VIEW_SIZE):
            for x in range(SCREEN_VIEW_SIZE):
                self.simple_screen_channels[self.simple_screen[y][x]][y][x] = 1


    def _update_connections(self):
        height, width = self.env.game.map.get_map_size()
        connection_mask = self.env.game.map.get_map_connection_mask()

        self.connections[0] = 1 - ((height - self.y_pos_org) / height) if connection_mask & 0b1000 else 0  # north
        self.connections[1] = 1 - ((self.y_pos_org + 1) / height) if connection_mask & 0b0100 else 0       # south
        self.connections[2] = 1 - ((self.x_pos_org + 1) / width)  if connection_mask & 0b0001 else 0       # east
        self.connections[3] = 1 - ((width - self.x_pos_org) / width)  if connection_mask & 0b0010 else 0   # west


    def _is_main_world_map(self):
        # Main outdoor maps (so far):
        # 0x00 - Global, 0x11 - Cave, 0x03 - Forest
        map_tileset = self.env.game.map.get_tileset_index()
        return (map_tileset == 0x00 or map_tileset == 0x11 or map_tileset == 0x03)


    def _update_map_coordinates(self):
        self.new_map = False
        x_pos_new, y_pos_new, n_map_new = self.env.game.map.get_current_location()
        self.moved_location = not (self.x_pos_org == x_pos_new and
                                   self.y_pos_org == y_pos_new and
                                   self.n_map_org == n_map_new)

        new_map = (n_map_new != self.n_map_org)
        if new_map or self.map_loading:
            self.map_loading = True
            self.x_pos_org, self.y_pos_org, self.n_map_org = x_pos_new, y_pos_new, n_map_new

            if self.map_ptr_org == self.env.game.map.get_current_map_ptr():
                return

            self.new_map = True
            self.map_loading = False

            # Don't count house's or they'll add to the visited reward scaler
            if n_map_new not in self.visited_maps and self._is_main_world_map():
                self.visited_maps.add(n_map_new)
                self.discovered_map = True


    def save_post_action_pos(self):
        self._update_map_coordinates()


    def save_pre_action_pos(self):
        self.x_pos_org, self.y_pos_org, self.n_map_org = self.env.game.map.get_current_location()
        self.discovered_map = False
        self.map_ptr_org = self.env.game.map.get_current_map_ptr()

        if len(self.visited_pos_order) > MAX_STEP_MEMORY:
            del_key = self.visited_pos_order.popleft()
            del self.visited_pos[del_key]

        current_pos = (self.x_pos_org, self.y_pos_org, self.n_map_org)
        if current_pos not in self.visited_pos:
            self.visited_pos[current_pos] = self.env.step_count
            self.visited_pos_order.append(current_pos)


    def update_map_stats(self):
        new_x_pos, new_y_pos, new_map_n = self.env.game.map.get_current_location()

        debug_str = f"Moved: {self.moved_location} \n"
        if self.new_map:
            debug_str = f"\nNew Map!\n"
        debug_str += f"Start location: {self.x_pos_org, self.y_pos_org, self.n_map_org} \n"
        debug_str += f"New location: {new_x_pos, new_y_pos, new_map_n} \n"
        debug_str += f"\n"
        debug_str += f"{self.simple_screen}"
        debug_str += f"\n"
        debug_str += f"{self.visited}"

        if len(self.location_history) > 10:
            self.location_history.popleft()
        self.location_history.append(debug_str)


    def get_exploration_reward(self):
        if self.env.game.battle.in_battle:
            return 0

        x_pos, y_pos, map_n = self.env.game.map.get_current_location()
        if not self.moved_location:
            if (not (self.env.gameboy.action_history[0] == 5 or self.env.gameboy.action_history[0] == 6) and 
                self.env.game.get_game_state() == self.env.game.GameState.EXPLORING and not self.map_loading):
                self.collisions += 1
                return -0.1
        
            return 0
        elif (x_pos, y_pos, map_n) in self.visited_pos:
            return 0.01
        
        # FALL THROUGH: In new location

        self.steps_discovered += 1
        # Bonus for exploring Gym, encourage discovery of gym boss fights (0xC0)
        # Bonus for exploring pokecenter/mart(0xBD) before talking to first nurse, ie. encourage early learning of pokecenter healing
        # Note, that pokecenter will be one on entering the first pokecenter building and 2 when talking to the first nurse
        audio_track = self.env.game.world.get_playing_audio_track()
        if audio_track == 0xC0 or (self.env.world.pokecenter_history <= 3 and audio_track == 0xBD):
            return 10
        else:
            return 1


    def get_map_reward(self):
        if self.discovered_map and self._is_main_world_map():
            return min(30 * len(self.visited_maps), 1000)
        
        return 0


    def update_map_obs(self):
        if self.env.game.battle.in_battle:
            self._clear_map_obs()  # Don't show the map while in battle b/c human can't see map when in battle
        else:
            x_pos_new, y_pos_new, n_map_new = self.env.game.map.get_current_location()

            self._update_visited_obs(x_pos_new, y_pos_new, n_map_new)
            self._update_simple_screen_obs(x_pos_new, y_pos_new, n_map_new)
            self._update_pos_obs(x_pos_new, y_pos_new, n_map_new)
            self._update_simple_screen_channel_obs()
            self._update_connections()
            
        self.update_map_stats()
