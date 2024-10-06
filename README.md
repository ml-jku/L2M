# Learning to Modulate pre-trained Models in RL
[![arXiv](https://img.shields.io/badge/arXiv-2306.14884-b31b1b.svg)](https://arxiv.org/abs/2306.14884)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Thomas Schmied<sup>**1**</sup>, Markus Hofmarcher<sup>**2**</sup>, Fabian Paischer<sup>**1**</sup>, Razvan Pacscanu<sup>**3,4**</sup>, Sepp Hochreiter<sup>**1,5**</sup> 

<sup>**1**</sup>ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria\
<sup>**2**</sup>JKU LIT SAL eSPML Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria\
<sup>**3**</sup>Google DeepMind\
<sup>**4**</sup>UCL\
<sup>**5**</sup>Institute of Advanced Research in Artificial Intelligence (IARAI), Vienna, Austria

This repository contains the source code for **"Learning to Modulate pre-trained Models in RL"** accepted at NeurIPS 2023.
The paper is available [here](https://arxiv.org/abs/2306.14884). 

![Multi-domain Decision Transformer (MDDT)](./figures/mddt.png) 

## Overview
This codebase supports training [Decision Transformer (DT)](https://arxiv.org/abs/2106.01345) models online or from offline datasets on the following domains: 
- [Meta-World](https://github.com/Farama-Foundation/Metaworld) / [Continual-World](https://github.com/awarelab/continual_world)
- [Atari](https://github.com/openai/gym)
- [Gym-MuJoCo](https://github.com/openai/gym)
- [ProcGen](https://github.com/openai/procgen)
- [DMControl](https://github.com/deepmind/dm_control)

This codebase relies on open-source frameworks, including: 
- [PyTorch](https://github.com/pytorch/pytorch)
- [Huggingface transformers](https://github.com/huggingface/transformers)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [wandb](https://github.com/wandb/wandb)
- [Hydra](https://github.com/facebookresearch/hydra)

What is in this repository?
```
.
├── configs                    # Contains all .yaml config files for Hydra to configure agents, envs, etc.
│   ├── agent_params            
│   ├── wandb_callback_params
│   ├── env_params
│   ├── eval_params
│   ├── run_params
│   └── config.yaml            # Main config file for Hydra - specifies log/data/model directories.
├── continual_world            # Submodule for Continual-World.
├── dmc2gym_custom             # Custom wrapper for DMControl.
├── figures             
├── scripts                    # Scrips for running experiments on Slurm/PBS in multi-gpu/node setups.
├── src                        # Main source directory.
│   ├── algos                  # Contains agent/model/prompt classes.
│   ├── augmentations          # Image augmentations.
│   ├── buffers                # Contains replay trajectory buffers.
│   ├── callbacks              # Contains callbacks for training (e.g., WandB, evaluation, etc.).
│   ├── data                   # Contains data utilities (e.g., for downloading Atari)
│   ├── envs                   # Contains functionality for creating environments.
│   ├── exploration            # Contains exploration strategies.
│   ├── optimizers             # Contains (custom) optimizers.
│   ├── schedulers             # Contains learning rate schedulers.
│   ├── tokenizers_custom      # Contains custom tokenizers for discretizing states/actions.
│   ├── utils                  
│   └── __init__.py
├── LICENSE
├── README.md
├── environment.yaml
├── requirements.txt
└── main.py                     # Main entry point for training/evaluating agents.
```
## Installation
Environment configuration and dependencies are available in `environment.yaml` and `requirements.txt`.

First, create the conda environment.
```
conda env create -f environment.yaml
conda activate mddt
```

Then install the remaining requirements (with MuJoCo already downloaded, if not see [here](#MuJoCo-installation)): 
```
pip install -r requirements.txt
```

<!-- It may be necessary to install PyTorch again, in case GPU is not detected: 
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
``` -->

Init the `continualworld` submodule and install: 
```
git submodule init
git submodule update
cd continualworld
pip install .
```
Install `meta-world`:
```
pip install git+https://github.com/rlworkgroup/metaworld.git@18118a28c06893da0f363786696cc792457b062b
```

Install custom version of [dmc2gym](https://github.com/denisyarats/dmc2gym). Our version makes `flatten_obs` optional, 
and, thus, allows us to construct the full observation space of all DMControl envs. 
```
cd dmc2gym_custom
pip install -e .
```

### MuJoCo installation
Download MuJoCo:
```
mkdir ~/.mujoco
cd ~/.mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200
wget https://www.roboti.us/file/mjkey.txt
```
Then add the following line to `.bashrc`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
```

#### Troubleshooting on cluster (without root access)
The following issues were helpful: 
- https://github.com/openai/mujoco-py/issues/96#issuecomment-678429159
- https://github.com/openai/mujoco-py/issues/627#issuecomment-1383054926
- https://github.com/openai/mujoco-py/issues/323#issuecomment-618365770

First, install the following packages: 
```
conda install -c conda-forge glew mesalib
conda install -c menpo glfw3 osmesa
pip install patchelf
```
Create the symlink manually: 
- https://github.com/openai/mujoco-py/issues/763#issuecomment-1519090452 
```
cp /usr/lib64/libGL.so.1 $CONDA_PREFIX/lib
ln -s $CONDA_PREFIX/lib/libGL.so.1 $CONDA_PREFIX/lib/libGL.so
```
Then do: 
```
mkdir ~/rpm
cd ~/rpm
curl -o libgcrypt11.rpm ftp://ftp.pbone.net/mirror/ftp5.gwdg.de/pub/opensuse/repositories/home:/bosconovic:/branches:/home:/elimat:/lsi/openSUSE_Leap_15.1/x86_64/libgcrypt11-1.5.4-lp151.23.29.x86_64.rpm
rpm2cpio libgcrypt11.rpm | cpio -id
```
Finally, export the path to `rpm` dir (add to `~/.bashrc`):
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/rpm/usr/lib64
export LDFLAGS="-L/~/rpm/usr/lib64"
```

## Setup

### Experiment configuration
This codebase relies on [Hydra](https://github.com/facebookresearch/hydra), which configures experiments via `.yaml` files. 
Hydra automatically creates the log folder structure for a given run, as specified in the respective `config.yaml` file.

The `config.yaml` is the main configuration entry point and contains the default parameters. The file references the respective default parameter files under the block
`defaults`. In addition, `config.yaml` contains 4 important constants that configure the directory paths: 
```
LOG_DIR: ../logs
DATA_DIR: ../data
SSD_DATA_DIR: ../data
MODELS_DIR: ../models
```

### Datasets
The genereated datasets are currently hosted via our web-server. Download Meta-World and DMControl datasets to the specified `DATA_DIR`: 
```
# Meta-World
wget --recursive --no-parent --no-host-directories --cut-dirs=2 -R "index.html*" https://ml.jku.at/research/l2m/metaworld
# DMControl
wget --recursive --no-parent --no-host-directories --cut-dirs=2 -R "index.html*" https://ml.jku.at/research/l2m/dm_control_1M
```
The datasets are also available on the Huggingface hub. Download using the `huggingface-cli`: 
```
# Meta-World
huggingface-cli download ml-jku/meta-world --local-dir=./meta-world --repo-type dataset
# DMControl
huggingface-cli download ml-jku/dm_control --local-dir=./dm_control --repo-type dataset
```
The framework also supports Atari, D4RL, and visual DMControl datasets. 
For [Atari](src/data/atari/README.md) and [visual DMControl](src/data/dm_control/README.md), we refer to the respective READMEs.

## Running experiments
In the following, we provide some illustrative examples of how to run the experiments in the paper. 

### Pre-training runs
To train a 40M multi-domain Decision Transformer (MDDT) model on MT40 + DMC10 with 3 seeds on a single GPU, run: 
```
python main.py -m experiment_name=pretrain seed=42,43,44 env_params=multi_domain_mtdmc run_params=pretrain eval_params=pretrain_disc agent_params=cdt_pretrain_disc agent_params.kind=MDDT agent_params/model_kwargs=multi_domain_mtdmc agent_params/data_paths=mt40v2_dmc10 +agent_params/replay_buffer_kwargs=multi_domain_mtdmc +agent_params.accumulation_steps=2
```

### Single-task fine-tuning
To fine-tune the pre-trained model using LoRA on a single CW10 task with 3 seeds, run: 
```
python main.py -m experiment_name=cw10_lora seed=42,43,44 env_params=mt50_pretrain run_params=finetune eval_params=finetune agent_params=cdt_mpdt_disc agent_params/model_kwargs=mdmpdt_mtdmc agent_params/data_paths=cw10_v2_cwnet_2M +agent_params/replay_buffer_kwargs=mtdmc_ft agent_params/model_kwargs/prompt_kwargs=lora env_params.envid=hammer-v2 agent_params.data_paths.names='${env_params.envid}.pkl' env_params.eval_env_names=
```

### Continual fine-tuning
To fine-tune the pre-trained model using L2M on all CW10 tasks in a sequential manner with 3 seeds, run: 
```
python main.py -m experiment_name=cw10_cl_l2m seed=42,43,44 env_params=multi_domain_ft env_params.eval_env_names=cw10_v2 run_params=finetune_coff eval_params=finetune_md_cl agent_params=cdt_mpdt_disc +agent_params.steps_per_task=100000 agent_params/model_kwargs=mdmpdt_mtdmc agent_params/data_paths=cw10_v2_cwnet_2M +agent_params/replay_buffer_kwargs=mtdmc_ft +agent_params.replay_buffer_kwargs.kind=continual agent_params/model_kwargs/prompt_kwargs=l2m_lora
```

### Multi-GPU training 
For multi-GPU training, we use `torchrun`. The tool conflicts with `hydra`. 
Therefore, a launcher plugin [hydra_torchrun_launcher](https://github.com/facebookresearch/hydra/tree/main/contrib/hydra_torchrun_launcher) was created.

To enable the plugin, clone the `hydra` repo, cd to `contrib/hydra_torchrun_launcher`, and pip install the plugin: 
```
git clone https://github.com/facebookresearch/hydra.git
cd hydra/contrib/hydra_torchrun_launcher
pip install -e .
```
The plugin can be used from the commandline: 
```
python main.py -m hydra/launcher=torchrun hydra.launcher.nproc_per_node=4 [...]
```
Running experiments on a local cluster on a single node can be done via `CUDA_VISIBLE_DEVICES` to specify the GPUs to use: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py -m hydra/launcher=torchrun hydra.launcher.nproc_per_node=4 [...]
```

On Slurm, executing `torchrun` on a single node works alike. E.g., to run on 2 GPUs on a single node: 
```
#!/bin/bash
#SBATCH --account=X
#SBATCH --qos=X
#SBATCH --partition=X
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32

source activate mddt
python main.py -m hydra/launcher=torchrun hydra.launcher.nproc_per_node=2 [...]
```
Example scripts for multi-gpu training on Slurm or PBS are available in `scripts`.

### Multi-node training
Running on Slurm/PBS in a multi-node setup requires a little more care. Example scripts are provided in `scripts`.

## Citation
If you find this useful, please consider citing our work: 
```
@article{schmied2024learning,
  title={Learning to Modulate pre-trained Models in RL},
  author={Schmied, Thomas and Hofmarcher, Markus and Paischer, Fabian and Pascanu, Razvan and Hochreiter, Sepp},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
