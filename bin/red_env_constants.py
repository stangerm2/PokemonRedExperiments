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
POS_BYTES = 3
POS_MAP_DETAIL_BYTES = 6
PYBOY_RUN_SPEED = 0
