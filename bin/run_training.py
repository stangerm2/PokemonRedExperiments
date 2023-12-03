import os
from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from red_env_constants import *

# Embedding sizes
embedding_size_x = 8
embedding_size_y = 8
embedding_size_map = 8
POS_MEMORY_SIZE = 10

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        # Define CNN architecture for spatial inputs
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully connected layer for the one-hot encoded vector
        self.fc_p2p = nn.Sequential(
            nn.Linear(37, 16),
            nn.ReLU()
        )

        # Output layer to combine features
        self.fc_combined = nn.Sequential(
            nn.Linear(16 * (7 + 3) * 7 * 2 + 16, features_dim),  # Adjust input size
            nn.ReLU()
        )

    def forward(self, observations):
        screen = observations["screen"].float() / 255  # Normalize and add channel dimension
        screen = screen.unsqueeze(1)  # Shape: [batch_size, 1, 10, 7]

        screen_visited = observations["screen_visited"].float().unsqueeze(1)  # Shape: [batch_size, 1, 10, 7]

        p2p = observations["p2p"].float()

        # Apply CNN to spatial inputs
        screen_features = self.cnn(screen)
        screen_visited_features = self.cnn(screen_visited)

        # Apply fully connected layer to p2p input
        p2p_features = self.fc_p2p(p2p)

        # Combine features
        combined = torch.cat([screen_features, screen_visited_features, p2p_features], dim=1)
        return self.fc_combined(combined)


def make_env(thread_id, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param id: (int) index of the subprocess
    """

    def _init():
        return RedGymEnv(thread_id, env_conf)

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    use_wandb_logging = True
    ep_length = 2048 * 1
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'../saved_runs/session_{sess_id}')

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': False, 'reward_scale': 1, 'extra_buttons': False,
        'explore_weight': 3  # 2.5
    }

    num_cpu = 1  # Also sets the number of episodes per training iteration

    if 0 < num_cpu < 31:
        env_config['debug'] = True
        env_config['headless'] = False
        use_wandb_logging = False

    print(env_config)

    env = SubprocVecEnv([make_env(i, env_config, GLOBAL_SEED) for i in range(num_cpu)])
    # env = DummyVecEnv([lambda: RedGymEnv(config=env_config)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length * 1, save_path=os.path.abspath(sess_path),
                                             name_prefix='poke')

    callbacks = [checkpoint_callback, TensorboardCallback()]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            dir=sess_path,
        )
        callbacks.append(WandbCallback())

    # put a checkpoint here you want to start from
    file_name = ''
    # file_name = '../saved_runs/session_b8e28b22/poke_125337600_steps'

    model = None
    checkpoint_exists = exists(file_name + '.zip')
    if len(file_name) != 0 and not checkpoint_exists:
        print('\nERROR: Checkpoint not found!')
    elif checkpoint_exists:
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        # policy_kwargs={"features_extractor_class": CustomFeatureExtractor, 
        #                   "features_extractor_kwargs": {"features_dim": 64}},
        model = PPO("MultiInputPolicy", env, 
                    verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998,
                    seed=GLOBAL_SEED, device="auto", tensorboard_log=sess_path)

    print(model.policy)

    model.learn(total_timesteps=ep_length * num_cpu * 1000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
