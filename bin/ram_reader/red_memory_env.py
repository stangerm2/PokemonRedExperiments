# Game Tiem Played
PLAY_TIME_HOURS = 0xDA41
PLAY_TIME_MINUTES = 0xDA43
PLAY_TIME_SECONDS = 0xDA44

# Pokedex
POKEDEX_ADDR_LENGTH = 19
POKEDEX_OWNED = 0xD2F7
POKEDEX_SEEN = 0xD30A

# Gym Status
OBTAINED_BADGES = 0xD356
FOUGHT_GIOVANNI_FLAG = 0xD751
FOUGHT_BROCK_FLAG = 0xD755
FOUGHT_MISTY_FLAG = 0xD75E
FOUGHT_LT_SURGE_FLAG = 0xD773
FOUGHT_ERIKA_FLAG = 0xD77C
FOUGHT_ARTICUNO_FLAG = 0xD782
FOUGHT_KOGA_FLAG = 0xD792
FOUGHT_BLAINE_FLAG = 0xD79A
FOUGHT_SABRINA_FLAG = 0xD7B3

# Flags for (dis)appearing sprites, like the guard in Cerulean City or the Pokéballs in Oak's Lab
MOVABLE_OBJECTS_FLAGS = (0xD5A6, 0xD5C5)

TOWN_MAP_FLAG = 0xD5F3
OAKS_PARCEL_FLAG = 0xD60D
FOSSILIZED_POKEMON_FLAG = 0xD710
LAPRAS_RECEIVED_FLAG = 0xD72E
FOUGHT_ZAPDOS_FLAG = 0xD7D4
FOUGHT_SNORLAX_VERMILION_FLAG = 0xD7D8
FOUGHT_SNORLAX_CELADON_FLAG = 0xD7E0
FOUGHT_MOLTRES_FLAG = 0xD7EE
SS_ANNE_FLAG = 0xD803
MEWTWO_APPEAR_FLAG = 0xD5C0  # bit 1
MEWTWO_CATCH_FLAG = 0xD85F  # Needs D5C0 bit 1 clear, bit 2 clear

# Game Milestones
GAME_MILESTONES = [
    TOWN_MAP_FLAG,
    OAKS_PARCEL_FLAG,
    FOSSILIZED_POKEMON_FLAG,
    LAPRAS_RECEIVED_FLAG,
    FOUGHT_ZAPDOS_FLAG,
    FOUGHT_SNORLAX_VERMILION_FLAG,
    FOUGHT_SNORLAX_CELADON_FLAG,
    FOUGHT_MOLTRES_FLAG,
    SS_ANNE_FLAG,
]

# Safari States
SAFARI_ZONE_TIME = (0xD70D, 0xD70E)
SAFARI_GAME_OVER_FLAG = 0xD790  # bit 7

# Travel States
BIKE_SPEED = 0xD700
FLY_ANYWHERE_FLAG = (0xD70B, 0xD70C)
POSITION_IN_AIR = 0xD714
FOLLOWING_NPC_FLAG = 0xCD38

# Audio Cue's
AUDIO_CURRENT_TRACK = 0xC026  # The music that's playing in the background
AUDIO_OVERLAY_SOUND = 0xC001  # Audio sounds that overlay the music, like bumping into a wall or entering a house
AUDIO_CURRENT_TRACK_NO_DELAY = 0xD35B  # The music that's playing in the background (0xC026 has delay trans)
AUDIO_FADE_OUT = 0x0CFC7
