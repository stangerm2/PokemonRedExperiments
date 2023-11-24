import sys

from pyboy import PyBoy, WindowEvent
from pyboy.logger import log_level
from red_env_constants import *


def pyboy_init_actions(extra_buttons):
    valid_actions = [
        WindowEvent.PRESS_ARROW_DOWN,
        WindowEvent.PRESS_ARROW_LEFT,
        WindowEvent.PRESS_ARROW_RIGHT,
        WindowEvent.PRESS_ARROW_UP,
        WindowEvent.PRESS_BUTTON_A,
        WindowEvent.PRESS_BUTTON_B,
    ]

    if extra_buttons:
        valid_actions.extend([
            WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PASS
        ])

    return valid_actions


def pyboy_term_actions(action):
    match action:
        case WindowEvent.PRESS_ARROW_DOWN:
            return WindowEvent.RELEASE_ARROW_DOWN
        case WindowEvent.PRESS_ARROW_UP:
            return WindowEvent.RELEASE_ARROW_UP
        case WindowEvent.PRESS_ARROW_LEFT:
            return WindowEvent.RELEASE_ARROW_LEFT
        case WindowEvent.PRESS_ARROW_RIGHT:
            return WindowEvent.RELEASE_ARROW_RIGHT
        case WindowEvent.PRESS_BUTTON_A:
            return WindowEvent.RELEASE_BUTTON_A
        case WindowEvent.PRESS_BUTTON_B:
            return WindowEvent.RELEASE_BUTTON_B
        case _:
            return WindowEvent.PASS


class PyBoyManager:
    def __init__(self, env):
        self.env = env
        self.pyboy = None
        self.valid_actions = pyboy_init_actions(self.env.extra_buttons)
        self.setup_pyboy()

    def setup_pyboy(self):
        log_level("ERROR")
        window_type = 'headless' if self.env.headless else 'SDL2'
        self.pyboy = PyBoy(
            self.env.rom_location,
            debugging=False,
            disable_input=False,
            window_type=window_type,
            hide_window='--quiet' in sys.argv,
        )

        if not self.env.headless:
            self.pyboy.set_emulation_speed(PYBOY_RUN_SPEED)  # Configurable emulation speed

        self.reload_game()

    def reload_game(self):
        self._load_save_file(self.env.init_state)

    def _load_save_file(self, save_file):
        if self.pyboy and save_file:
            with open(save_file, "rb") as f:
                self.pyboy.load_state(f)

    def run_action_on_emulator(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        if not self.env.save_video and self.env.headless:
            self.pyboy._rendering(False)

        for i in range(self.env.act_freq):
            if i == 8:
                termination_action = pyboy_term_actions(action)
                self.pyboy.send_input(termination_action)

            if self.env.save_video and not self.env.fast_video:
                self.env.screen.add_video_frame()

            if i == self.env.act_freq - 1:
                self.pyboy._rendering(True)

            self.pyboy.tick()

        if self.env.save_video and self.env.fast_video:
            self.env.screen.add_video_frame()

    def get_memory_value(self, addr):
        return self.pyboy.get_memory_value(addr)

    def _read_bit(self, addr, bit: int) -> bool:
        return bin(256 + self.get_memory_value(addr))[-bit - 1] == '1'
