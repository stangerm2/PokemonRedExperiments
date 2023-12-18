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
        self.x_pos_org, self.y_pos_org, self.n_map_org = None, None, None
        self.visited_pos = {}
        self.visited_pos_order = deque()
        self.new_map = 0  # TODO: Inc/dec to 6
        self.moved_location = False  # indicates if the player moved 1 or more spot
        self.discovered_location = False # indicates if the player is in previously unvisited location
        self.location_history = deque()
        self.steps_discovered = 0
        self.pokecenter_history = {0 : True}

        self.screen = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.float32)
        self.visited = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.walkable = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.coordinates = np.zeros((3, 7), dtype=np.uint8)  # x,y,map stacked, 7 bits as all val's are < 128

        self.tester = RedGymObsTester(self)

        np.set_printoptions(linewidth=np.inf)


    def _update_tile_obs(self):
        # The screen tiles don't have sprites/npc's with them
        self.screen[0:7, 0:7] = self.env.game.map.get_centered_7x7_tiles()
        self.walkable[0:7, 0:7] = self.env.game.map.get_centered_step_count_7x7_screen()


    def _update_visited_obs(self, x_pos_new, y_pos_new, n_map_new):
        callback = lambda x, y, pos: self._update_matrix_visited(x, y, pos)
        self._traverse_matrix(x_pos_new, y_pos_new, n_map_new, callback)

        # DO NOT set cur pos as visited on the obs until the next turn, it REALLY helps the AI
        # ie.. self.visited[3][3] = 0 (this is intentional)

    def _update_npc_and_norm_obs(self, x_pos_new, y_pos_new, n_map_new):
        sprites = self.env.game.map.get_npc_location_dict(n_map_new)

        callback = lambda x, y, pos: self._update_matrix_npc_and_normalize(x, y, pos, sprites)
        self._traverse_matrix(x_pos_new, y_pos_new, n_map_new, callback)


    def _update_pos_obs(self, x_pos_new, y_pos_new, n_map_new):
        x_pos_binary = format(x_pos_new, f'0{SCREEN_VIEW_SIZE}b')
        y_pos_binary = format(y_pos_new, f'0{SCREEN_VIEW_SIZE}b')
        m_pos_binary = format(n_map_new, f'0{SCREEN_VIEW_SIZE}b')
    
        # appends the x,y, pos binary form to the bottom of the screen and visited matrix's
        for i, bit in enumerate(x_pos_binary):
            self.coordinates[0][i] = bit

        for i, bit in enumerate(y_pos_binary):
            self.coordinates[1][i] = bit

        for i, bit in enumerate(m_pos_binary):
            self.coordinates[2][i] = bit


    def _traverse_matrix(self, x_pos_new, y_pos_new, n_map_new, callback):
        center_x = center_y = SCREEN_VIEW_SIZE // 2

        for y in range(SCREEN_VIEW_SIZE):
            for x in range(SCREEN_VIEW_SIZE):
                center_x = center_y = SCREEN_VIEW_SIZE // 2
                x_offset = x - center_x
                y_offset = y - center_y
                current_pos = x_pos_new + x_offset, y_pos_new + y_offset, n_map_new

                callback(x, y, current_pos)


    def _update_matrix_visited(self, x, y, pos):
        if pos in self.visited_pos:
            self.visited[y][x] = 0
        else:
            self.visited[y][x] = 1


    def _update_matrix_npc_and_normalize(self, x, y, pos, sprites):
        if pos in sprites:
            self.screen[y][x] = self.env.memory.byte_to_float_norm[sprites[pos]] + 0.1
        else:
            self.screen[y][x] = self.env.memory.byte_to_float_norm[int(self.screen[y][x])]

    def _clear_map_obs(self):
        self.screen = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.float32)
        self.visited = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.walkable = np.zeros((SCREEN_VIEW_SIZE, SCREEN_VIEW_SIZE), dtype=np.uint8)
        self.coordinates = np.zeros((3, 7), dtype=np.uint8)

    def save_post_action_pos(self):
        x_pos_new, y_pos_new, n_map_new = self.env.game.map.get_current_location()
        self.moved_location = not (self.x_pos_org == x_pos_new and
                                   self.y_pos_org == y_pos_new and
                                   self.n_map_org == n_map_new)

        if self.moved_location:
            # Bug check: AI is only allowed to move 0 or 1 spots per turn, new maps change x,y ref pos so don't count.
            # When the game goes to a new map, it changes m first, then y,x will update on the next turn, still some corner cases like fly, blackout, bike
            if self.new_map:
                self.x_pos_org, self.y_pos_org, self.n_map_org = x_pos_new, y_pos_new, n_map_new
                self.new_map -= 1
            elif n_map_new == self.n_map_org:
                if not (abs(self.x_pos_org - x_pos_new) + abs(self.y_pos_org - y_pos_new) <= 1):
                    self.update_map_stats()

                    debug_str = ""
                    while len(self.location_history):
                        debug_str += self.location_history.popleft()
                    self.env.support.save_debug_string(debug_str)
                    # assert False
            else:
                self.new_map = 1

            if (x_pos_new, y_pos_new, n_map_new) in self.visited_pos:
                self.discovered_location = True


    def save_pre_action_pos(self):
        self.x_pos_org, self.y_pos_org, self.n_map_org = self.env.game.map.get_current_location()
        self.discovered_location = False

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
        debug_str += f"{self.screen}"
        debug_str += f"\n"
        debug_str += f"{self.visited}"
        debug_str += f"\n"
        debug_str += f"{self.walkable}"

        if len(self.location_history) > 10:
            self.location_history.popleft()
        self.location_history.append(debug_str)

    def get_pokecenter_reward(self):
        audio_id = self.env.game.world.get_playing_audio_track()
        if audio_id != 0xBD:
            return 0  # we aren't in a mart or pokecenter

        pokecenter_id = self.env.game.world.get_pokecenter_id()
        if pokecenter_id != 1 and pokecenter_id in self.pokecenter_history:
            return 0.2 # 2x better than battle steady state, encourage being in pokecenter
                
        self.pokecenter_history[pokecenter_id] = True
        return 100


    def get_exploration_reward(self):
        x_pos, y_pos, map_n = self.env.game.map.get_current_location()
        if not self.moved_location:
            return 0
        elif (x_pos, y_pos, map_n) in self.visited_pos:
            return 0.01
        else:
            self.steps_discovered += 1
            return 1        

    def update_map_obs(self):
        if self.env.game.battle.in_battle:
            self._clear_map_obs()  # Don't show the map while in battle b/c human can't see map when in battle
        else:
            x_pos_new, y_pos_new, n_map_new = self.env.game.map.get_current_location()

            # Order matters here
            self._update_tile_obs()
            self._update_visited_obs(x_pos_new, y_pos_new, n_map_new)
            self._update_npc_and_norm_obs(x_pos_new, y_pos_new, n_map_new)  # Overwrites screen with npc's locations
            self._update_pos_obs(x_pos_new, y_pos_new, n_map_new)
            #self.tilemap_matrix()

        self.update_map_stats()



    def get_tile_obs(self):
        # return a matrix size 9,10 that represents the world map tiles
        
        screen_tilemap = self.tilemap_matrix()

        walkable_tiles_indices = []
        tile_obs = np.zeros((9,10), np.uint8)
        collision_ptr = TilesetHeaderAddress.COLLISION_DATA_POINTER.read(env.pyboy)
        tileset_type = MapMiscAddress.TILESET_TYPE.read(env.pyboy)
        tileset = TilesetID.find_value(PlayerPositionAddress.CURRENT_TILESET.read(env.pyboy))
        grass_tile_index = TilesetHeaderAddress.GRASS_TILE.read(env.pyboy)
        talking_over_tiles = TilesetHeaderAddress.TALKING_OVER_TILES.read_list(env.pyboy)
        talking_over_tiles = list(filter(lambda x: x != 0xFF, talking_over_tiles))
        for i in range(len(talking_over_tiles)):
            talking_over_tiles[i] = talking_over_tiles[i] + 0x100

        if tileset_type > 0:
            if grass_tile_index != 0xFF:
                grass_tiles = [grass_tile_index + 0x100]
            else:
                grass_tiles = []
        else:
            grass_tiles = []


        for i in range(0x180): # Check all tiles in tilemap
            tile_index = read_m(env.pyboy, collision_ptr + i)
            if tile_index == 0xFF:
                break
            else:
                walkable_tiles_indices.append(tile_index + 0x100) #Adding 256 for some reason?

        
        #  0 - WALL
        #  2 - WALKABLE

        tile_obs = np.multiply(np.isin(screen_tilemap, walkable_tiles_indices).astype(np.uint8),2)
        
        #  3 - GRASS
        grass_matrix = np.multiply(np.isin(screen_tilemap, grass_tiles).astype(np.uint8),1) #Grass is walkable so its add 1
        tile_obs = np.add(tile_obs, grass_matrix)

        #  5 - CUT

        if tileset in CUT_TREE:
            jump_matrix = np.multiply(np.isin(screen_tilemap, CUT_TREE[tileset]).astype(np.uint8),5)
            tile_obs = np.add(tile_obs, jump_matrix)


        #  6 - SEA
        if tileset in SURFABLE:
            jump_matrix = np.multiply(np.isin(screen_tilemap, SURFABLE[tileset]).astype(np.uint8),6)
            tile_obs = np.add(tile_obs, jump_matrix)


        #  7 - JUMP
        if tileset in JUMPABLE:
            jump_matrix = np.multiply(np.isin(screen_tilemap, JUMPABLE[tileset]).astype(np.uint8),7) # Jump is a Collision so Zero
            tile_obs = np.add(tile_obs, jump_matrix)


        #  4 - WARP
        player_y = PlayerPositionAddress.CURRENT_Y_POS.read(env.pyboy)
        player_x = PlayerPositionAddress.CURRENT_X_POS.read(env.pyboy)
        warp_matrix = np.zeros(screen_tilemap.shape, dtype=np.uint8)
        obj_data_ptr = MapHeaderAddress.OBJ_DATA_POINTER.read(env.pyboy)
        num_warps = read_m(env.pyboy, obj_data_ptr + 0x1) # First Byte of object Data is border block ID

        # 4 bytes per Warp
        for i in range(0,num_warps):
            y, x, _ , _ = read_multi_m_list(env.pyboy,obj_data_ptr + 0x2 + (i * 0x4),4)
            ## Format for Warps is
            # Y, X, Destination warp-to ID
            y_offset = player_y - y
            x_offset = player_x - x

            matrix_y = 4 - y_offset
            matrix_x = 4 - x_offset

            if matrix_y >= 9 or matrix_y < 0:
                continue

            if matrix_x >= 10 or matrix_x < 0:
                continue

            if tile_obs[matrix_y][matrix_x] == 0:
                continue

            warp_matrix[matrix_y][matrix_x] = 4

        tile_obs = np.add(tile_obs,warp_matrix)
        
        #  8 - TALK

        talkover_area_matrix = np.multiply(np.isin(screen_tilemap, talking_over_tiles).astype(np.uint8),8)
        tile_obs = np.add(tile_obs, talkover_area_matrix)

        #  9 - Sprite

        for i in range(0x10, 0x100, 0x010):

            base_sprite_address = SpriteBaseAddress.PLAYER_SPRITE.value + i
            sprite_pic_address = base_sprite_address + SpriteMemoryOffset.PICTURE_ID.value[0]
            sprite_x_addr = base_sprite_address + SpriteMemoryOffset.X_SCREEN_POS.value[0]
            sprite_y_addr = base_sprite_address + SpriteMemoryOffset.Y_SCREEN_POS.value[0]
            sprite_offscreen = base_sprite_address + SpriteMemoryOffset.IMAGE_INDEX.value[0]
            sprite_offscreen = read_m(env.pyboy,sprite_offscreen)
            if sprite_offscreen == 0xFF:
                continue
            x,y = read_m(env.pyboy,sprite_x_addr), read_m(env.pyboy, sprite_y_addr)
            sprite_pic_id = read_multi_m(env.pyboy,sprite_pic_address)

            # player is at 64,60
            x_offset, y_offset = x - 64, y - 6
            x_offset, y_offset = int(x_offset/16) , int(y_offset /16)
            matrix_y = y_offset + 1
            matrix_x = x_offset + 4

            if matrix_y >= 9 or matrix_y < 0:
                continue

            if matrix_x >= 10 or matrix_x < 0:
                continue

            tile_obs[matrix_y][matrix_x] = 9


        #  1 - PLAYER
        tile_obs[4][4] = 1


        #tile_obs = np.multiply(np.divide(tile_obs,9,dtype=np.float16),255, dtype=np.uint8, casting='unsafe')
        return tile_obs
