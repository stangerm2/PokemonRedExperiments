from os.path import exists
from pyboy import PyBoy
from pyboy import WindowEvent

'''pyboy = PyBoy('PokemonRed.gb')
while not pyboy.tick():
    print(
        f'c: {pyboy.get_memory_value(0xFF8C)},'
        f'1: {pyboy.get_memory_value(0xCC51)},'
        f' 2: {pyboy.get_memory_value(0xCC52)},'
        f' 3: {pyboy.get_memory_value(0xCC53)}')
pyboy.stop()'''
for i in range(500, 0, -1):
    print(i)
