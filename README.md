
# Train RL agents to play Pokemon Red

## Watch the Video on Youtube!

  

<p  float="left">

<a  href="https://youtu.be/DcYLT37ImBY">

<img  src="/assets/youtube.jpg?raw=true"  height="192">

</a>

<a  href="https://youtu.be/DcYLT37ImBY">

<img  src="/assets/poke_map.gif?raw=true"  height="192">

</a>

</p>

  

## Join the discord server

[![Join the Discord server!](https://invidget.switchblade.xyz/RvadteZk4G)](http://discord.gg/RvadteZk4G)

## Getting Started 🎮

1. Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to `PokemonRed.gb` if it is not already. The sha1 sum should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which you can verify by running `shasum PokemonRed.gb`. The Pokemon.gb file MUST be in the main directory and your current directory MUST be the `baselines/` directory in order for this to work.
2. Move into the  `bin/`  directory:  
    `cd bin`
3.  Install Python dependencies:  
    `pip install -r ../helper_scripts/requirements.txt`  
    It may be necessary in some cases to separately install the SDL libraries.

## Running the Pretrained Model Interactively 🎮

🐍 Python 3.10 is recommended. Other versions may work but have not been tested.

Pretrained Models are currently not supported in this fork but likely will be added in the future.

  

## Training the Model 🏋️

  

<img  src="/assets/grid.png?raw=true"  height="156">

1. Run:

```python run_training.py```

  
## Pokemon Red API
This repo comes with a helper repo which tries to abstract reading the RAM interface with simpler API's. Checkout the `bin/ram_reader/red_ram_api.py` for accessing the API.

A simple usage example can be viewed and run by `python api_example.py`  from within the `bin` dir. 

## Tracking Training Progress 📈

The current state of each game is rendered to images in the session directory.

You can track the progress in tensorboard by moving into the session directory and running:

```tensorboard --logdir .```

You can then navigate to `localhost:6006` in your browser to view metrics.

To enable wandb integration, change `use_wandb_logging` in the training script to `True`.

  

## Supporting Libraries

Check out these awesome projects!

### [PyBoy](https://github.com/Baekalfen/PyBoy)

<a  href="https://github.com/Baekalfen/PyBoy">

<img  src="/assets/pyboy.svg"  height="64">

</a>

  

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)

<a  href="https://github.com/DLR-RM/stable-baselines3">

<img  src="/assets/sblogo.png"  height="64">

</a>