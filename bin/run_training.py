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

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        self.x_embedding = nn.Embedding(256, embedding_size_x)
        self.y_embedding = nn.Embedding(256, embedding_size_y)
        self.map_embedding = nn.Embedding(256, embedding_size_map)

        # Assuming each sequence length is 10
        sequence_length = 6

        # RNN layer
        self.rnn = nn.GRU(input_size=embedding_size_x + embedding_size_y + embedding_size_map,
                          hidden_size=64,
                          num_layers=1,
                          batch_first=True)

        # Linear layer for processing RNN output
        self.fc = nn.Linear(64 * POS_MEMORY_SIZE, features_dim)

    def forward(self, observations):
        # Embed each x and y coordinate in the sequence
        x_embedded = self.x_embedding(observations["pos_x"].long()).view(-1, 10, embedding_size_x)
        y_embedded = self.y_embedding(observations["pos_y"].long()).view(-1, 10, embedding_size_y)
        map_embedded = self.map_embedding(observations["map"].long()).view(-1, 10, embedding_size_map)

        # Concatenate embeddings
        combined_embeddings = torch.cat((x_embedded, y_embedded, map_embedded), dim=2)

        # Process sequence with RNN
        rnn_output, _ = self.rnn(combined_embeddings)

        # Flatten RNN output for linear layer
        rnn_output_flattened = rnn_output.contiguous().view(rnn_output.size(0), -1)

        # Final processing with linear layer
        features = self.fc(rnn_output_flattened)

        return features

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

    num_cpu = 120  # Also sets the number of episodes per training iteration

    if 0 < num_cpu < 31:
        # env_config['debug'] = True
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
    file_name = ''#'../saved_runs/session_24763c7f/poke_10321920_steps'

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
        # policy_kwargs={'features_extractor_class': CustomFeatureExtractor},
        model = PPO("MultiInputPolicy", env,
                    verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998,
                    seed=GLOBAL_SEED, device="auto", tensorboard_log=sess_path)

    print(model.policy)

    model.learn(total_timesteps=ep_length * num_cpu * 1000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
