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

        # Embeddings for discrete values
        self.action_embedding = nn.Embedding(num_embeddings=256, embedding_dim=8)
        self.game_state_embedding = nn.Embedding(num_embeddings=256, embedding_dim=8)

        # Calculate the total features after concatenation
        cnn_output_dim = 16 * 10 * 7  # Adjust based on CNN output
        total_embedding_dim = 8 + 8  # Sum of embedding dimensions
        total_features_dim = 2 * cnn_output_dim + total_embedding_dim

        # Output layer to combine features
        self.fc_combined = nn.Sequential(
            nn.Linear(3232, features_dim),  # Adjust input size
            nn.ReLU()
        )

    def forward(self, observations):
        batch_size = observations["screen"].size(0)  # Get dynamic batch size from screen

        screen = observations["screen"].unsqueeze(1)  # Normalize and add channel dimension
        visited = observations["visited"].unsqueeze(1)

        # Apply CNN to spatial inputs
        screen_features = self.cnn(screen)
        visited_features = self.cnn(visited)

        # Embeddings for discrete values
        # Explicitly use batch_size for reshaping
        action_features = self.action_embedding(observations["action"].long()).view(batch_size, -1)
        game_state_features = self.game_state_embedding(observations["game_state"].long()).view(batch_size, -1)

        # Concatenate all features and ensure correct dimension
        combined_features = torch.cat([
            screen_features, 
            visited_features, 
            action_features, 
            game_state_features
        ], dim=1)

        # Ensure the input size to fc_combined matches the concatenated features size
        return self.fc_combined(combined_features)


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
        'action_freq': 24, 'init_state': '../pokemon_ai_squirt_poke_balls.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': False, 'reward_scale': 1, 'extra_buttons': False,
        'explore_weight': 3  # 2.5
    }

    num_cpu = 124  # Also sets the number of episodes per training iteration

    if 0 < num_cpu < 50:
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
    #file_name = '../saved_runs/session_2ea2edc2/poke_65519616_steps'

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
                    verbose=1, n_steps=2048 // 8, batch_size=128, n_epochs=3, gamma=0.998,  policy_kwargs={"features_extractor_class": CustomFeatureExtractor, "features_extractor_kwargs": {"features_dim": 64}},
                    seed=GLOBAL_SEED, device="auto", tensorboard_log=sess_path)

    print(model.policy)

    model.learn(total_timesteps=ep_length * num_cpu * 1000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
