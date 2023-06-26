import h5py
import numpy as np
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
from pathlib import Path


def load_hdf5(path):
    returns_to_go = None
    with h5py.File(path, "r") as f:
        # fully trajectory
        observations = f['states'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        if "returns_to_go" in f:
            returns_to_go = f["returns_to_go"][:]
        dones = np.array([f['dones'][()]])
    return observations, actions, rewards, dones, returns_to_go


def save_image(obs, save_dir):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, obs)
    plt.close


if __name__ == "__main__":
    conf = OmegaConf.load("../../../configs/agent_params/data_paths/atari.yaml")
    conf.names = OmegaConf.load("../../../configs/agent_params/data_paths/names/atari46.yaml")
    paths = [Path(conf.base) / name / "000000000.hdf5" for name in conf.names]
    save_dir = Path("./figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for p in paths: 
        obs, _, _, _, _  = load_hdf5(p)
        
        # plot first 10 and last 10 images
        for i in [*list(range(0, 5)), *list(range(-5, 0))]:
            game = p.parent.name
            img_name = f"{game}_{i}.png"
            frame = obs[i][0]
            plt.imshow(frame)
            plt.title(img_name)
            save_path = Path(save_dir) / img_name
            print(f"Saving figure to {save_path}")
            plt.savefig(save_path, bbox_inches='tight')
