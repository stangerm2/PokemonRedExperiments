import json
import threading

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from red_env_constants import *



class RedGymEnvSupport:
    def __init__(self, env):
        self.env = env


    def update_coord_obs(self):
        new_x_pos, new_y_pos, new_map_n = self.get_current_location()
        self.env.pos = np.array([new_x_pos, new_y_pos, new_map_n])

        self.env.pos_memory = np.roll(self.env.pos_memory, POS_BYTES)
        self.env.pos_memory[:POS_BYTES] = self.env.pos[:POS_BYTES]

    def update_movement(self, x_pos_cur, y_pos_cur, n_map_cur):
        x_pos_new, y_pos_new, n_map_new = self.get_current_location()
        self.env.moved_location = not (x_pos_cur == x_pos_new and y_pos_cur == y_pos_new and n_map_cur == n_map_new)

        if self.env.moved_location:
            if self.env.new_map:
                self.env.new_map = False
            elif n_map_new == n_map_cur:
                pass  # Additional logic if needed
            else:
                self.env.new_map = True

    def save_screenshot(self, image=None):
        x_pos, y_pos, map_n = self.get_current_location()

        if image is None:
            image = self.env.screen.render(reduce_res=False)

        ss_dir = self.env.s_path / Path(f'screenshots/{map_n}_{x_pos}_{y_pos}')
        ss_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'{threading.get_ident()}_{self.env.step_count}.jpeg'),
            image)

    def get_current_location(self):
        x_pos = self.env.game.get_memory_value(0xD362)
        y_pos = self.env.game.get_memory_value(0xD361)
        map_n = self.env.game.get_memory_value(0xD35E)
        return x_pos, y_pos, map_n

    def get_and_save_pos(self):
        x_pos, y_pos, map_n = self.get_current_location()

        if len(self.env.visited_pos_order) > MAX_STEP_MEMORY:
            del_key = self.env.visited_pos_order.popleft()
            del self.env.visited_pos[del_key]

        if (x_pos, y_pos, map_n) in self.env.visited_pos:
            return x_pos, y_pos, map_n

        self.env.visited_pos[(x_pos, y_pos, map_n)] = self.env.step_count
        self.env.visited_pos_order.append((x_pos, y_pos, map_n))

        return x_pos, y_pos, map_n

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

    def _save_current_frame(self):
        plt.imsave(
            self.env.s_path / Path(f'curframe_{self.env.instance_id}.jpeg'),
            self.env.screen.render(reduce_res=False))

    def _close_video_writers(self):
        self.env.full_frame_writer.close()
        self.env.model_frame_writer.close()

    def _construct_progress_string(self):
        prog_string = f'step: {self.env.step_count:6d}'
        for key, val in self.env.progress_reward.items():
            prog_string += f' {key}: {val:5.2f}'
        prog_string += f' sum: {self.env.total_reward:5.2f}'
        prog_string += f' reward: {self.env.total_reward:5.2f}, seen_cord_len: {len(self.env.visited_pos)}'
        prog_string += f' steps_discovered: {self.env.steps_discovered}'
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
        self.env.all_runs.append(self.env.progress_reward)
        with open(self.env.s_path / Path(f'all_runs_{self.env.instance_id}.json'), 'w') as f:
            json.dump(self.env.all_runs, f)
        pd.DataFrame(self.env.agent_stats).to_csv(
            self.env.s_path / Path(f'agent_stats_{self.env.instance_id}.csv.gz'), compression='gzip', mode='a')


