from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        # Define separate embeddings for different features
        embedding_size_map = 16
        embedding_size_surroundings = 16
        sin_freqs = 8
        self.map_embedding = nn.Embedding(256, embedding_size_map)
        self.surroundings_embedding = nn.Embedding(256, embedding_size_surroundings)

        # Separate processors for each feature type
        self.map_processor = nn.Sequential(
            nn.Linear(embedding_size_map, 32),
            nn.ReLU()
        )
        self.surroundings_processor = nn.Sequential(
            nn.Linear(embedding_size_surroundings * 4, 64),  # 4 surroundings
            nn.ReLU()
        )
        self.pos_processor = nn.Sequential(
            nn.Linear(sin_freqs * 2, 32),  # Assuming 2D positions
            nn.ReLU()
        )

        # Combine processed features with a final layer
        self.combined_fc = nn.Sequential(
            nn.Linear(32 + 64 + 32, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Process map feature
        map_obs = observations["map"].int()
        map_features = self.map_embedding(map_obs).squeeze(1)
        map_processed = self.map_processor(map_features)

        # Process surroundings feature
        surroundings_obs = observations["surroundings"].int()
        surroundings_features = [self.surroundings_embedding(surroundings_obs[:, i]).squeeze(1) for i in range(4)]
        surroundings_combined = torch.cat(surroundings_features, dim=1)
        surroundings_processed = self.surroundings_processor(surroundings_combined)

        # Process position feature
        pos_processed = self.pos_processor(observations["pos"])

        # Combine all processed features
        combined = torch.cat((map_processed, surroundings_processed, pos_processed), dim=1)
        return self.combined_fc(combined)

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    use_wandb_logging = True
    ep_length = 2048 * 1
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')

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
        env_config['headless'] = False
        use_wandb_logging = False

    print(env_config)

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    # env = DummyVecEnv([lambda: RedGymEnv(config=env_config)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
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
        )
        callbacks.append(WandbCallback())

    # env_checker.check_env(env)
    learn_steps = 40
    # put a checkpoint here you want to start from
    # file_name = 'baseline_session_ad2ee02f/poke_55296000_steps'
    file_name = '__session_2229ba62/poke_145981440_steps'

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO("MultiInputPolicy", env, policy_kwargs={'features_extractor_class': CustomFeatureExtractor},
                    verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998,
                    seed=0, device="auto", tensorboard_log=sess_path)

    print(model.policy)

    model.learn(total_timesteps=(ep_length) * num_cpu * 1000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
