# DMControl data

## Overview
For state-based DMControl, we make use of the datasets provided by [RLUnplugged](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged).

**UPDATE**: It is not possible to reconstruct the original episodes from the dm_control data in RLUnplugged.
Thus, we genereated the data ourselves. 
- https://github.com/deepmind/deepmind-research/issues/110

For visual DMControl, we use the datasets provided by [vD4RL](https://github.com/conglu1997/v-d4rl), which are available at: 
- https://drive.google.com/drive/folders/15HpW6nlJexJP5A4ygGk-1plqt9XdcWGI

## Installation
The datasets provided in RLUnplugged are povided in tf dataset format. 
To avoid inconsistencies in our regular environment, we create a separate conda environment: 
```
conda create -n dmcontrol_data python=3.9
```

Then install [tensorflow](https://www.tensorflow.org/install/pip#linux) and [tfdata](https://github.com/tensorflow/datasets):
```
# install tensorflow
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#install tfdata
pip install tensorflow-datasets
```

## Data download 
For state-based DMControl, we make use of the datasets provided by [RLUnplugged](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged).
For details on the dataset see the website. 

Unfortunately, RLUnplugged does not provide a python-based way to download the datasets automatically. 
Therefore, we need to download them manually using `gsutil`. If not already installed, install `gsutil`, as described here:
- https://cloud.google.com/storage/docs/gsutil_install 

Then run: the shell script, to download the datasets:
```
bash download_rlunplugged.sh
```

For visual DMControl, we use the datasets provided by [vD4RL](https://github.com/conglu1997/v-d4rl), which are available at: 
- https://drive.google.com/drive/folders/15HpW6nlJexJP5A4ygGk-1plqt9XdcWGI

We manually downloaded the `main` folder for the `medium_expert` trajectories and use the `84x84` version.  


## Postprocessing the data

### DMControl
The downloaded datasets contain 100 (datatype?) files per task.
For faster dataloading, we create a single file npz file per task, which will be kept in memory by the dataloader for faster dataloading. 

### Visual DMControl:
The datasets are split into 5 `.hdf5` files per tasks. 
We create one `.hdf5` file for every episodes, for faster dataloading and partial trajectory loading. 

The individual `.hdf5` episode files adhere to the following folder structure:
```
environment family (i.e., visual_dm_control)
- environment name (e.g. cartpole_swingup)
-- one hdf5 file (numbered, zero padded to 9 digits) per episode with fields: states, actions, rewards, dones
```

To prepare the data, run: 
```
python prepare_visual_dmcontrol.py --data_dir=DATA_DIR --save_dir=SAVE_DIR --add_rtgs --grayscale
```
We automatically add the returns-togo using (`--rtg`) to the data and grayscale the image observations (`--grayscale`).
Grayscaling is required, as all Atari images are grayscale. 
