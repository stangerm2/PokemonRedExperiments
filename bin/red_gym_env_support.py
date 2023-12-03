import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from red_gym_map import RedGymMap
from red_env_constants import *


def calc_byte_float_norm():
    bytes_norm = []
    for i in range(BYTE_SIZE):
        bytes_norm.append(math.floor((i / 256.0) * 10 ** 4) / 10 ** 4) # normalize lookup for 0-255

    return bytes_norm


class RedGymGlobalMemory:
    def __init__(self):
        self.byte_to_float_norm = calc_byte_float_norm()


class RedGymEnvSupport:
    def __init__(self, env):
        if env.debug:
            print('**** RedGymEnvSupport ****')

        self.env = env
        self.map = RedGymMap(self.env)

    def save_screenshot(self, image=None):
        x_pos, y_pos, map_n = self.map.get_current_location()

        if image is None:
            image = self.env.screen.render(reduce_res=False)

        ss_dir = self.env.s_path / Path(f'screenshots/{self.env.instance_id}')
        ss_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'{self.env.step_count}_{map_n}_{x_pos}_{y_pos}.jpeg'),
            image)

    def check_if_done(self):
        return self.env.step_count >= self.env.max_steps

    def save_and_print_info(self, done):
        if self.env.print_rewards:
            prog_string = self._construct_progress_string()
            print(f'\r{prog_string}', end='', flush=True)

        if self.env.print_rewards and done:
            self._print_final_rewards()

        if self.env.save_video and done:
            self._close_video_writers()

        if done:
            self._save_run_data()

    def save_debug_string(self, output_str):
        debug_path = self.env.s_path / 'debug'
        debug_path.mkdir(exist_ok=True)

        # Construct the full file path
        file_path = debug_path / f'debug_{self.env.step_count}.txt'

        # Write the output string to the file
        with open(file_path, 'w') as file:
            file.write(output_str)

    def _save_current_frame(self):
        plt.imsave(
            self.env.s_path / Path(f'curframe_{self.env.instance_id}.jpeg'),
            self.env.screen.render(reduce_res=False))

    def _close_video_writers(self):
        self.env.full_frame_writer.close()
        self.env.model_frame_writer.close()

    def _construct_progress_string(self):
        prog_string = f'step: {self.env.step_count:6d}'
        for key, val in self.env.agent_stats[-1].items():
            prog_string += f' {key}: {val:5.2f}'
        prog_string += f' sum: {self.env.total_reward:5.2f}'
        prog_string += f' reward: {self.env.total_reward:5.2f}'
        return prog_string

    def _print_final_rewards(self):
        print('', flush=True)

        if self.env.save_final_state:
            fs_path = self.env.s_path / 'final_states'
            fs_path.mkdir(exist_ok=True)
            plt.imsave(
                fs_path / Path(f'frame_r{self.env.total_reward:.4f}_{self.env.reset_count}_small.jpeg'),
                self.env.screen.render())
            plt.imsave(
                fs_path / Path(f'frame_r{self.env.total_reward:.4f}_{self.env.reset_count}_full.jpeg'),
                self.env.screen.render(reduce_res=False))

    def _save_run_data(self):
        stats_path = self.env.s_path / 'agent_stats'
        stats_path.mkdir(exist_ok=True)
        pd.DataFrame(self.env.agent_stats).to_csv(
            stats_path / Path(f'agent_stats_{self.env.instance_id}.csv.gz'), compression='gzip', mode='a')


