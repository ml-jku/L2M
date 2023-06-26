import argparse
import hydra
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from stable_baselines3.common.buffers import ReplayBuffer


def load_dataset(path):
    if not path.exists():
        print("Missing path: ", path)
        return None
    path = str(path)
    if path.endswith(".pkl"):
        with open(path, 'rb') as f:
            ds = pickle.load(f)
    elif path.endswith(".npz"): 
        ds = np.load(path)
    return ds


def save_dataset(save_path, dataset):
    save_path = str(save_path)
    if save_path.endswith(".npz"):
        np.savez(save_path, **dataset)
    elif save_path.endswith(".pkl"):
        with open(str(save_path), 'wb') as f:
            pickle.dump(dataset, f)
    else: 
        raise ValueError(f"Unknown file extension: {save_path}")


def make_dummy_datasets(paths, save_dir, max_steps=10000):
    for path in tqdm(paths):
        print(path)
        save_path = save_dir / path.name
        dataset = load_dataset(path)
        if dataset is None:
            continue
        if isinstance(dataset, ReplayBuffer):
            dataset.actions = dataset.actions[:max_steps]
            dataset.observations = dataset.observations[:max_steps]
            dataset.next_observations = dataset.next_observations[:max_steps]
            dataset.rewards = dataset.rewards[:max_steps]
            dataset.dones = dataset.dones[:max_steps]
            dataset.timeouts = dataset.timeouts[:max_steps]
        else: 
            # is npz file
            ds = dict()
            ds["actions"] = dataset["actions"][:max_steps]
            ds["observations"] = dataset["observations"][:max_steps]
            ds["next_observations"] = dataset["next_observations"][:max_steps] 
            ds["rewards"] = dataset["rewards"][:max_steps]
            ds["dones"] = dataset["dones"][:max_steps]
            dataset = ds
        save_dataset(save_path, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/home/thomas/Projects-Linux/CRL_with_transformers/data/dummy")
    parser.add_argument("--data_paths", type=str, default="mt40_v2_cwnet_2M_local")
    parser.add_argument("--max_steps", type=int, default=10000)
    args = parser.parse_args()
    hydra.initialize(config_path="../../configs")
    conf = hydra.compose(config_name="config",
                         overrides=["env_params=mt50_pretrain",
                                    "agent_params=cdt_pretrain",
                                    f"agent_params/data_paths={args.data_paths}"])
    conf.env_params.eval_env_names = None
    print(conf)
    data_paths = [Path(conf.agent_params.data_paths.base) / name for name in conf.agent_params.data_paths.names]
    save_dir = Path(args.save_dir) / Path(conf.agent_params.data_paths.base).stem if args.save_dir else None
    save_dir.mkdir(parents=True, exist_ok=True)
    make_dummy_datasets(data_paths, save_dir, args.max_steps)
