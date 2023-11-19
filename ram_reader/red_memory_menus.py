# Text Interactions
TEXT_ON_SCREEN = 0x8800

# Text Interaction Animation (NASTY HACK)
# There is absolutely no RAM value for when text animation is in progress, but by viewing the games working
# registers and pairing that info with 'TEXT_ON_SCREEN' it's possible to make a pretty good assumption but there
# are many combo's to account for!
WORKING_REG_A = 0xCC51
WORKING_REG_B = 0xCC52
WORKING_REG_C = 0xCC53

# Core menu navigation identification
TEXT_MENU_CURSOR_X_POS = 0xCC25 # Stale after interaction
TEXT_MENU_CURSOR_Y_POS = 0xCC26 # Stale after interaction
TEXT_MENU_CURSOR_LOCATION = (0xCC30, 0xCC31)
FIRST_DISPLAYED_MENU_ITEM_ID = 0xCC36

# Alt menu navigation identification
TEXT_MENU_TILE_BEHIND_CURSOR = 0xCC27 # Stale after interaction
TEXT_MENU_MAX_MENU_ITEM = 0xCC28 # Stale after interaction
TEXT_MENU_MENU_WATCHED_KEYS = 0xCC29 # Stale after interaction
TEXT_MENU_LAST_MENU_ITEM = 0xCC2A # Stale after interaction
LAST_CURSOR_POSITION_PARTY_BILLS_PC = 0xCC2B # Stale after interaction
LAST_CURSOR_POSITION_ITEM_SCREEN = 0xCC2C # Stale after interaction
LAST_CURSOR_POSITION_START_MENU = 0xCC2D # Stale after interaction

# Using PC
USING_PC_FLAG = 0xCF0C

# Text menu long
# 7f 07 cb start menu


# TEXT_MENU_CURSOR_LOCATIONS
# CC30: D3  CC31: C3 - start menu pokedex
# CC30: FB  CC31: C3 - start menu pokemon
# CC30: 23  CC31: C4 - start menu item
# CC30: B4  CC31: C4 - start menu self
# CC30: 73  CC31: C4 - start menu save
# CC30: 9B  CC31: C4 - start menu option
# CC30: C3  CC31: C4 - start menu quit

# CC30: 4C  CC31: C4 - pokecenter heal
# CC30: 74  CC31: C4 - pokecenter cancel

# CC30: B5  CC31: C3 - pokemart buy
# CC30: DD  CC31: C3 - pokemart sell
# CC30: 05  CC31: C4 - pokemart quit

# CC30: C9  CC31: C3 - PC someone
# CC30: C9  CC31: C3 - PC someone withdraw
# CC30: F1  CC31: C3 - PC someone deposit
# CC30: 19  CC31: C4 - PC someone release
# CC30: 41  CC31: C4 - PC someone change box
# CC30: 69  CC31: C4 - PC someone exit
# CC30: F1  CC31: C3 - PC self
# CC30: C9  CC31: C3 - PC self withdraw item
# CC30: F1  CC31: C3 - PC self deposit item
# CC30: 19  CC31: C3 - PC self toss item
# CC30: 41  CC31: C4 - PC self exit
# CC30: 19  CC31: C4 - PC oak
# CC30: 41  CC31: C4 - PC logoff

# CC30: C1  CC31: C4 - Battle/wild menu, fight
# CC30: A9  CC31: C4 - Battle/wild mv1
# CC30: BD  CC31: C4 - Battle/wild mv2
# CC30: D1  CC31: C4 - Battle/wild mv3
# CC30: E5  CC31: C4 - Battle/wild mv4
# CC30: C7  CC31: C4 - Battle menu, pkmn
# CC30: B4  CC31: C3 - Battle/roster pkmn 1
# CC30: DC  CC31: C3 - Battle/roster pkmn 2
# CC30: 04  CC31: C4 - Battle/roster pkmn 3
# CC30: 2C  CC31: C4 - Battle/roster pkmn 4
# CC30: 54  CC31: C4 - Battle/roster pkmn 5
# CC30: 7C  CC31: C4 - Battle/roster pkmn 6
# CC30: E9  CC31: C4 - Battle menu, item
# CC30: 8A  CC31: C4 - Battle menu, item X - use
# CC30: B2  CC31: C4 - Battle menu, item X - toss
# CC30: F5  CC31: C3 - Battle item 1
# CC30: 1D  CC31: C4 - Battle item 2
# CC30: 45  CC31: C4 - Battle item 3
# CC30: 45  CC31: C4 - Battle item 4 (items have weird scroll offset and also need 0xCC26 & CC36)
# CC30: EF  CC31: C4 - Battle menu, run

# CC30: 4F  CC31: C4 - Wild catch name yes
# CC30: 77  CC31: C4 - Wild catch name no
