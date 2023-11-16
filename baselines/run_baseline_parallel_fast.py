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


def precompute_fourier_table(freqs=8, max_value=255):
    """
    Precompute the sinusoidal Fourier table for all x,y coordinates from 0 to max_value.
    freqs: Number of frequencies/octaves to encode coordinates into.
    max_value: Maximum value for x and y coordinates.
    Returns: A tensor representing the precomputed Fourier table.
    """
    # Generate all combinations of x and y coordinates
    coords = torch.tensor([[x, y] for x in range(max_value + 1) for y in range(max_value + 1)], dtype=torch.float32, device="cpu")

    # Calculate the sinusoidal embeddings for all coordinates
    sin_embeddings = torch.hstack([
        torch.outer(coords[:, 0], 2 ** torch.arange(freqs, device="cpu")).sin(),
        torch.outer(coords[:, 1], 2 ** torch.arange(freqs, device="cpu")).sin()
    ])

    return sin_embeddings


def make_env(rank, fourier_table, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        # Instantiate the environment with the shared sin_table and config
        env = RedGymEnv(fourier_table, env_conf)
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

<<<<<<< Updated upstream
    try:
        # Copy precomputed Fourier table into shared memory
        fourier_table = precompute_fourier_table()
        env = SubprocVecEnv([make_env(i, fourier_table, env_config) for i in range(num_cpu)])
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
        file_name = '__sinusoidal_positional_encoding_session/poke_245760_steps'

        if exists(file_name + '.zip'):
            print('\nloading checkpoint')
            model = PPO.load(file_name, env=env)
            model.n_steps = ep_length
            model.n_envs = num_cpu
            model.rollout_buffer.buffer_size = ep_length
            model.rollout_buffer.n_envs = num_cpu
            model.rollout_buffer.reset()
        else:
            model = PPO("MlpPolicy", env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998,
                        seed=0, device="auto", tensorboard_log=sess_path)

        for i in range(learn_steps):
            model.learn(total_timesteps=(ep_length) * num_cpu * 1000, callback=CallbackList(callbacks))

        if use_wandb_logging:
            run.finish()

    finally:
        # Clean up the shared memory
        if shm is not None:
            shm.close()
            shm.unlink()
=======
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
        model = PPO("MlpPolicy", env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998,
                    seed=0, device="auto", tensorboard_log=sess_path)

    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length) * num_cpu * 1000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
>>>>>>> Stashed changes
