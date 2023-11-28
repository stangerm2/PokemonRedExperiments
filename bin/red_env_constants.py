# constants.py

# Constants for RedGymEnv
BYTE_SIZE = 256
VEC_DIM = 4320
NUM_ELEMENTS = 20000
MAX_STEP_MEMORY = 350
FRAME_STACKS = 3
OUTPUT_SHAPE = (36, 40, 3)
MEM_PADDING = 2
MEMORY_HEIGHT = 8
OUTPUT_FULL = (
    OUTPUT_SHAPE[0] * FRAME_STACKS + 2 * (MEM_PADDING + MEMORY_HEIGHT),
    OUTPUT_SHAPE[1],
    OUTPUT_SHAPE[2]
)
POS_HISTORY_SIZE = 14
POS_BYTES = 9
XYM_BYTES = 3
POS_MAP_DETAIL_BYTES = 6
NEXT_STEP_VISITED = 13  # Num of pos's that are within two moves from cur pos + cur pos
PYBOY_RUN_SPEED = 6
MAP_VALUE_PALLET_TOWN = 12

GLOBAL_SEED = 0


# Player addresses
PLAYER_LOCATION_X = 0xD362
PLAYER_LOCATION_Y = 0xD361
PLAYER_MAP = 0xD35E
PLAYER_ANIM_FRAME_COUNTER = 0xC108
PLAYER_FACING_DIR = 0xC109
PLAYER_COLLISION = 0xC10C # Running into NPC, doesn't count map boundary collisions
PLAYER_IN_GRASS = 0xC207 # 0x80 in poke grass, else 00

# Player's surroundings (tiles above, below, left and right of player sprite)[N/A when in chat/menu screen]
TILE_ABOVE_PLAYER = 0xC434
TILE_BELOW_PLAYER = 0xC484
TILE_LEFT_OF_PLAYER = 0xC45A
TILE_RIGHT_OF_PLAYER = 0xC45E
TILE_CURRENT_AND_FRONT_BUMP_PLAYER = 0xCFC6 # Tile ID of player until sprite bump obstacle, then it's obstacle tile ID
