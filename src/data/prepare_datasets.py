import pickle
import argparse
import numpy as np
import dmc2gym_custom
from pathlib import Path
from tqdm import tqdm
from src.envs.dmcontrol_utils import map_flattened_obs_to_full_space
from src.envs.env_names import DM_CONTROL_ENVS


DM_CONTROL_MAPPING = {name.replace("-", "_"): name for name in DM_CONTROL_ENVS}


def prepare_datasets(dir_path, save_dir, to_full_obs_space=False, map_names=False):
    paths = [str(path) for path in Path(dir_path).glob("*.pkl")]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(paths):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        file_name = Path(path).stem
        save_path = save_dir / f"{file_name}.npz"
        write_to_npz(obj, save_path, to_full_obs_space, file_name=file_name, map_names=map_names)


def write_to_npz(obj, save_path, to_full_space=False, map_names=False, file_name=None):
    print(f"Writing datset to: {save_path}")
    observations, next_observations, actions, rewards, dones = obj.observations, obj.next_observations, obj.actions, \
        obj.rewards, obj.dones
    if to_full_space:
        # map observations to full obs space in DMControl
        assert file_name is not None
        file_name = DM_CONTROL_MAPPING[file_name] if map_names else file_name
        domain_name, task_name = file_name.split("-")
        env = dmc2gym_custom.make(domain_name=domain_name, task_name=task_name)
        observations = map_flattened_obs_to_full_space(observations, env.observation_spec())
        next_observations = map_flattened_obs_to_full_space(next_observations, env.observation_spec())
        print(f"New observations shape: {observations.shape}")
    np.savez_compressed(save_path, observations=observations, next_observations=next_observations,
                        actions=actions, rewards=rewards, dones=dones)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", default="/home/thomas/Projects-Linux/CRL_with_transformers/data/metaworld_v2_cwnet_2M", type=str)
    parser.add_argument("--save_dir", default="/home/thomas/Projects-Linux/CRL_with_transformers/data/mt50", type=str)
    parser.add_argument("--to_full_space", action="store_true", help="Should only be used with DMControl envs.")
    parser.add_argument("--map_names", action="store_true", 
                        help="In DMControl, some files are named without '-' between domain and task names.")
    args = parser.parse_args()
    prepare_datasets(args.load_dir, args.save_dir, args.to_full_space, args.map_names)
