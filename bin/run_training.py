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



class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.coordinate_memory = None
        self.action_memory = None
        self.game_state_memory = None
        self.step_count = 0

        # Define CNN architecture for spatial inputs
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully connected layer for coordinates
        self.coordinates_fc = nn.Sequential(
            nn.Linear(3 * 7, features_dim),  # Flattened size of coordinates, repeated 3 times
            nn.ReLU()
        )

        self.action_embedding = nn.Embedding(num_embeddings=7, embedding_dim=8)
        self.game_state_embedding = nn.Embedding(num_embeddings=117, embedding_dim=8)

        # LSTM layers for action and game state embeddings
        self.coord_lstm = nn.LSTM(input_size=21, hidden_size=features_dim, batch_first=True)
        self.action_lstm = nn.LSTM(input_size=56, hidden_size=features_dim, batch_first=True)
        self.game_state_lstm = nn.LSTM(input_size=936, hidden_size=features_dim, batch_first=True)


        # Fully connected layers for output
        self.fc_layers = nn.Sequential(
            nn.Linear(976, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        batch_size = observations["visited"].size(0)  # Get dynamic batch size from screen
        device = observations["screen"].device  # Assuming 'screen' is part of your observations
        self._init_memory(device)

        combined_input = torch.cat([observations["screen"].unsqueeze(1),
                                    observations["visited"].unsqueeze(1),
                                    observations["walkable"].unsqueeze(1)], dim=1)

        # Apply CNN to spatial inputs
        screen_features = self.cnn(combined_input)

        # Process 'coordinates' and pass through fully connected layer
        coordinates_input = observations["coordinates"].view(batch_size, -1).unsqueeze(0).to(device)
        #coordinates_features = self.coordinates_fc(coordinates_input)

        # Explicitly use batch_size for reshaping
        action_input = self.action_embedding(observations["action"].int()).view(batch_size, -1).unsqueeze(0).to(device)
        game_state_input = self.game_state_embedding(observations["game_state"].int()).view(batch_size, -1).unsqueeze(0).to(device)

        # Process through LSTMs
        _, self.coordinate_memory = self.coord_lstm(coordinates_input, self.coordinate_memory)
        _, self.action_memory = self.action_lstm(action_input, self.action_memory)
        _, self.game_state_memory = self.game_state_lstm(game_state_input, self.game_state_memory)

        self._detach_hidden_states(device)

        # Extract LSTM final states
        coordinates_features = self.coordinate_memory[0].squeeze(0).squeeze(0)
        action_lstm_features = self.action_memory[0].squeeze(0).squeeze(0)
        game_state_lstm_features = self.game_state_memory[0].squeeze(0).squeeze(0)
        action_lstm_features = action_lstm_features.unsqueeze(0).repeat(batch_size, 1)
        game_state_lstm_features = game_state_lstm_features.unsqueeze(0).repeat(batch_size, 1)
        coordinates_features = coordinates_features.unsqueeze(0).repeat(batch_size, 1)

        combined_features = torch.cat([
            screen_features,
            coordinates_features,
            action_lstm_features, 
            game_state_lstm_features
        ], dim=1)

        # Ensure the input size to fc_combined matches the concatenated features size
        return self.fc_layers(combined_features)
    
    def _init_memory(self, device):
            # Initialize hidden states if None
        if self.coordinate_memory is not None:
            return
        
        self.coordinate_memory = (torch.zeros(1, 1, self.features_dim, device=device),
                                    torch.zeros(1, 1, self.features_dim, device=device))

        self.action_memory = (torch.zeros(1, 1, self.features_dim, device=device),
                                torch.zeros(1, 1, self.features_dim, device=device))

        self.game_state_memory = (torch.zeros(1, 1, self.features_dim, device=device),
                                    torch.zeros(1, 1, self.features_dim, device=device))

    def _detach_hidden_states(self, device):
        self.step_count += 1

        if self.step_count == 2048:
            self.action_memory = None
            self.game_state_memory = None
            self.coordinate_memory = None
            self.step_count = 0
            self._init_memory(device)
            return
        
        # Detach both hidden state and cell state
        hidden_state, cell_state = self.action_memory
        hidden_state = hidden_state.detach()
        cell_state = cell_state.detach()

        # Reconstruct the tuple
        self.action_memory = (hidden_state, cell_state)

        hidden_state, cell_state = self.game_state_memory
        hidden_state = hidden_state.detach()
        cell_state = cell_state.detach()

        # Reconstruct the tuple
        self.game_state_memory = (hidden_state, cell_state)

        hidden_state, cell_state = self.coordinate_memory
        hidden_state = hidden_state.detach()
        cell_state = cell_state.detach()

        # Reconstruct the tuple
        self.coordinate_memory = (hidden_state, cell_state)


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

    num_cpu = 1  # Also sets the number of episodes per training iteration

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
    # file_name = '../saved_runs/session_2700a5a5/poke_94470144_steps'

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
        model = PPO("MultiInputPolicy", env, ent_coef=0.01,
                    verbose=1, n_steps=512, batch_size=512, n_epochs=3, gamma=0.998,  policy_kwargs={"features_extractor_class": CustomFeatureExtractor, "features_extractor_kwargs": {"features_dim": 64}},
                    seed=GLOBAL_SEED, device="auto", tensorboard_log=sess_path)

    print(model.policy)

    model.learn(total_timesteps=ep_length * num_cpu * 1000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
