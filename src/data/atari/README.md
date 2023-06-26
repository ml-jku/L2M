# Atari data

## Overview 
For Atari, we make use of the [DQN Replay Dataset](https://research.google/tools/datasets/dqn-replay/).
For details on the dataset see the website. 

For loading the individual episodes, we rely on [d3rlpy](https://github.com/takuseno/d3rlpy) (a popular library for offline RL) and 
[d4rl-atari](https://github.com/takuseno/d4rl-atari).

The individual `.hdf5` episode files adhere to the following folder structure:
```
environment family (e.g. atari)
- environment name (e.g. breakout)
-- one .hdf5 file (numbered, zero padded to 9 digits) per episode with fields: states, actions, rewards, dones
```

## Installation
As `d3rlpy` and `d4rl-atari` have different dependencies than the regular codebase, we use a separate conda environment
to download the datasets.

Create conda environment:

```
conda create -n atari_data python=3.9
conda activate atari_data
```

Then install the requirements:
```
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

If problems are encountered with ROM licenses: 
```
pip install autorom
AutoROM
```

## Usage:
For each game, the Atari datasets are provided in a [GCP Bucket](https://console.cloud.google.com/storage/browser/atari-replay-datasets) and stored in 50 slices each containing 1M transitions (200M frames + frame-stack=4 --> 50M transistions).

To collect all 50 slices for Breakout, run: 
```
python download_atari_datasets.py --save_dir="SAVE_DIR" --num_slices=50
```

The episode quality can be specified using `--quality` ('random', or 'mixed', 'expert'). This is only used if `num_slices < 50`.
```
python download_atari_datasets.py --save_dir="SAVE_DIR" --quality="mixed" --num_slices=49
```

The maximum number of episodes to collect can be specified using `--max_episodes`.
```
python download_atari_datasets.py --save_dir="SAVE_DIR" --max_episodes=20000 --quality="mixed" --num_slices=49
```

This creates one `.hdf5` per episode in the specified `save_dir`. A `stats.json` file is written
to the same folder and contains information about the collected episodes (e.g., number of transitions, return/length of episodes).
In addition, `episode_lengths.json` is written to the same directory that contains the episode lengths for each episode.

There is an option for using differen file formats. Compressed files require less space, but data loading takes longer. 
We do not use compression by default. The default file format is `.hdf5`, as data loading is faster. 

Furthermore, return-to-gos can be pre-computed and added to the datasets. This is not done by default.

## Collecting datasets
We map actions to the full action space using `a_to_full_space` and add return-to-gos using `add_rtgs`. 

We collect 1M transitions for each of the 46 games used in [Multi-game Decision Transformer (MDGT)](https://arxiv.org/abs/2205.15241).
```
python download_atari_datasets.py --save_dir="DATA_DIR/atari_1M" --max_transitions=1000000 --envs pong asterix breakout qbert seaquest alien beam-rider freeway ms-pacman space-invaders amidar assault atlantis bank-heist battle-zone boxing carnival centipede chopper-command crazy-climber demon-attack double-dunk enduro fishing-derby frostbite gopher gravitar hero ice-hockey jamesbond kangaroo krull kung-fu-master name-this-game phoenix pooyan riverraid road-runner robotank star-gunner time-pilot up-n-down video-pinball wizard-of-wor yars-revenge zaxxon --quality="expert" --num_slices=49 --a_to_full_space --add_rtgs
```

Additional downloads for dataloading benchmark:
```
# npzc
python download_atari_datasets.py --save_dir="DATA_DIR/atari_1M_npzc" --max_transitions=1000000 --envs pong --quality="expert" --num_slices=49 --a_to_full_space --add_rtgs --save_format=npzc

# npz
python download_atari_datasets.py --save_dir="DATA_DIR/atari_1M_npz" --max_transitions=1000000 --envs pong --quality="expert" --num_slices=49 --a_to_full_space --add_rtgs --save_format=npz

# pickle
python download_atari_datasets.py --save_dir="DATA_DIR/atari_1M_pkl" --max_transitions=1000000 --envs pong --quality="expert" --num_slices=49 --a_to_full_space --add_rtgs --save_format=pkl
```

## Troubleshooting: 

In case the data download fails, this may be useful: 
- https://github.com/opencv/opencv-python/issues/370#issuecomment-996657018

