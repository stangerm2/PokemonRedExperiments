# Players Fighter
PLAYER_LOADED_POKEMON = 0xCC2F # Index of fighting mon, stale out battle.

# Constants
BATTLE_TOTAL_PLAYER_ATTRIBUTES = 7
BATTLE_TOTAL_ENEMIES_ATTRIBUTES = 13
BATTLE_TOTAL_TURN_ATTRIBUTES = 3

# General Battle Flags and Status
BATTLE_TYPE = 0xD057  # Battle type (normal, safari, etc.)
PARTY_XP_COUNT = 0xD058  # 0x0 during animation, Stale out of battle
CURRENT_OPPONENT = 0xD059 # Set the moment battle anim begins, and clears on battle end (not used for wild mon)
SPECIAL_BATTLE_TYPE = 0xD05A # (Normal battle, Safari Zone, Old Man battle...)
CRITICAL_HIT_OHKO_FLAG = 0xD05E  # Critical hit or OHKO flag
MOVE_MISSED = 0xD05F
POKEMON_MAX_COUNT = 6

# Battle Turn Info
TURNS_IN_CURRENT_BATTLE = 0xCCD5 # Player + Enemy Move = 1 Turn (Resets only on next battle)
PLAYER_SELECTED_MOVE = 0xCCDC # Stale out of battle
ENEMY_SELECTED_MOVE = 0xCCDD # Stale out of battle
BATTLE_TEXT_PAUSE_FLAG = 0xCC52

# Player's Attack Move Information (Refreshed while in battle 'Fight' menu, else stale even in battle)
PLAYERS_MOVE_NUM = 0xCFD2
PLAYERS_MOVE_EFFECT = 0xCFD3  # Player's Move Effect
PLAYERS_MOVE_POWER = 0xCFD4  # Player's Move Power
PLAYERS_MOVE_TYPE = 0xCFD5  # Player's Move Type
PLAYERS_MOVE_ACCURACY = 0xCFD6  # Player's Move Accuracy
PLAYERS_MOVE_MAX_PP = 0xCFD7  # Player's Move Max PP
PLAYER_BATTLE_STATUS = (0xD062, 0xD063, 0xD064)
# 0xD062: 0-Bide, 1-Thrash/Petal Dance, 2-MultiHit, 3-Flinch, 4-Charging, 5-MultiTurn, 6-Invuln, 7-Confuse
# 0xD063: 0-X Accuracy, 1-Mist, 2-Focus Energy, 4-Substitute, 5-Recharge, 6-Rage, 7-Leech Seed
# 0xD064: 0-Toxic, 1-Light Screen, 2-Reflect, 3-Transformed
PLAYERS_MULTI_HIT_MOVE_COUNTER = 0xD06A # Count left of multi-turn moves
PLAYERS_CONFUSION_COUNTER = 0xD06B
PLAYERS_TOXIC_COUNTER = 0xD06C
PLAYERS_DISABLE_COUNTER = (0xD06D, 0xD06E)

# Player's mon's modified battle stat's (from stat modifying moves)
PLAYERS_POKEMON_ATTACK_MODIFIER = 0xCD1A
PLAYERS_POKEMON_DEFENSE_MODIFIER = 0xCD1B
PLAYERS_POKEMON_SPEED_MODIFIER = 0xCD1C
PLAYERS_POKEMON_SPECIAL_MODIFIER = 0xCD1D
PLAYERS_POKEMON_ACCURACY_MODIFIER = 0xCD1E
PLAYERS_POKEMON_EVASION_MODIFIER = 0xCD1F



# Enemy's Pokémon Stats (In-Battle)
ENEMY_PARTY_COUNT = 0xD89C # N/A for wild mon, Stale out of battle
ENEMY_PARTY_SPECIES = (0xD89D, 0xD89E, 0xD89F, 0xD8A0, 0xD8A1, 0xD8A2)  # N/A wild mon, Stale out of battle, 0xFF term
ENEMYS_POKEMON = 0xCFE5 # Enemy/wild current Pokemon, Stale out of battle
ENEMYS_POKEMON_LEVEL = 0xCFF3  # Enemy's level, Stale out of battle
ENEMYS_POKEMON_HP = (0xCFE6, 0xCFE7)  # Enemy's current HP

ENEMYS_POKEMON_STATUS = 0xCFE9  # Enemy's status effects, Stale out of battle
ENEMYS_POKEMON_TYPES = (0xCFEA, 0xCFEB)  # Enemy's type, Stale out of battle
ENEMYS_POKEMON_MOVES = (0xCFED, 0xCFEE, 0xCFEF, 0xCFF0)  # Enemy's moves, Stale out of battle

ENEMY_TRAINER_POKEMON_HP = (0xD8A5, 0xD8A6)  # Only valid for trainers/gyms not wild mons. HP doesn't dec until mon is dead, then it's 0
ENEMY_TRAINER_POKEMON_HP_OFFSET = 0x2C

# Enemy's Battle Information
ENEMYS_MOVE_ID = 0xCFCC  # Enemy's Move ID
ENEMYS_MOVE_EFFECT = 0xCFCD  # Enemy's Move Effect
ENEMYS_MOVE_POWER = 0xCFCE  # Enemy's Move Power
ENEMYS_MOVE_TYPE = 0xCFCF  # Enemy's Move Type
ENEMYS_MOVE_ACCURACY = 0xCFD0  # Enemy's Move Accuracy
ENEMYS_MOVE_MAX_PP = 0xCFD1  # Enemy's Move Max PP

# More detailed Enemy's Pokémon Stats
ENEMYS_POKEMON_MAX_HP = 0xCFF4  # Enemy's max HP
ENEMYS_POKEMON_ATTACK = 0xCFF6  # Enemy's attack
ENEMYS_POKEMON_DEFENSE = 0xCFF8  # Enemy's defense
ENEMYS_POKEMON_SPEED = 0xCFFA  # Enemy's speed
ENEMYS_POKEMON_SPECIAL = 0xCFFC  # Enemy's special
ENEMYS_POKEMON_MAX_PP_FIRST_SLOT = 0xCFFE  # Enemy's PP for the first move slot
ENEMYS_POKEMON_MAX_PP_SECOND_SLOT = 0xCFFF  # Enemy's PP for the second move slot
ENEMYS_POKEMON_MAX_PP_THIRD_SLOT = 0xD000  # Enemy's PP for the third move slot
ENEMYS_POKEMON_MAX_PP_FOURTH_SLOT = 0xD001  # Enemy's PP for the fourth move slot
ENEMYS_POKEMON_ATTACK_MODIFIER = 0xCD2E  # TODO: Needs to be verified
ENEMYS_POKEMON_DEFENSE_MODIFIER = 0xCD2F
ENEMYS_POKEMON_SPEED_MODIFIER = 0xCD30
ENEMYS_POKEMON_SPECIAL_MODIFIER = 0xCD31
ENEMYS_POKEMON_ACCURACY_MODIFIER = 0xCD32
ENEMYS_POKEMON_EVASION_MODIFIER = 0xCD33

ENEMYS_MULTI_HIT_MOVE_COUNTER = 0xD06F
ENEMYS_CONFUSION_COUNTER = 0xD070
ENEMYS_TOXIC_COUNTER = 0xD071
ENEMYS_DISABLE_COUNTER = (0xD072, 0xD073)

