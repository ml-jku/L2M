import pickle
import collections
import numpy as np
import argparse
import json
import hydra
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def extract_trajectories_from_buffer(buffer):
    """

    From trajectory_buffer.py
    Args:
        buffer: ReplayBuffer object.

    Returns: list of individual trajectories.

    """
    trajectories = []
    current_trj = collections.defaultdict(list)
    pos = buffer.pos if not buffer.full else len(buffer.observations)
    for s, s1, a, r, done in tqdm(zip(buffer.observations[:pos], buffer.next_observations[:pos],
                                      buffer.actions[:pos], buffer.rewards[:pos], buffer.dones[:pos]),
                                  total=pos, desc="Extracting trajectories"):
        nans = [np.isnan(s).any(), np.isnan(s1).any(), np.isnan(a).any(), np.isnan(r)]
        if any(nans):
            print("NaNs found:", nans)
        current_trj["observations"].append(s)
        current_trj["next_observations"].append(s1)
        current_trj["actions"].append(a)
        current_trj["rewards"].append(r)
        current_trj["terminals"].append(done)
        if done:
            trajectories.append(current_trj)
            current_trj = collections.defaultdict(list)
    return trajectories


def extract_returns(trajectories):
    return [np.array(trj["rewards"]).sum().item() for trj in trajectories]


def extract_stats(paths, target_multiplier=1, save_dir=None):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    all_r_stats, max_return_per_task, max_reward_per_task = {}, {}, {}
    all_a_stats, all_s_stats = {}, {}

    for path in paths:
        print(f"Loading trajectories from: {path}")
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        task_name = Path(path).stem
        trajectories = extract_trajectories_from_buffer(obj)

        # extract rewards, returns stats from trjs
        r_stats, max_return, max_reward = extract_return_stats(trajectories, target_multiplier)
        all_r_stats[task_name] = r_stats
        max_return_per_task[task_name] = max_return
        max_reward_per_task[task_name] = max_reward

        # extract state/action stats from trjs
        all_s_stats[task_name] = extract_array_stats(trajectories, "observations")
        all_a_stats[task_name] = extract_array_stats(trajectories, "actions")

    if save_dir is not None:
        dataset_name = Path(paths[0]).parts[-2]
        save_dir = Path(save_dir) / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_r_stats).round(4).T.to_csv(save_dir / "r_stats.csv")
        pd.DataFrame(all_a_stats).round(4).T.to_csv(save_dir / "a_stats.csv")
        pd.DataFrame(all_s_stats).round(4).T.to_csv(save_dir / "s_stats.csv")
        with open(save_dir / "max_returns.json", "w") as f:
            json.dump(max_return_per_task, f, indent=4, sort_keys=False)
        with open(save_dir / "max_rewards.json", "w") as f:
            json.dump(max_reward_per_task, f, indent=4, sort_keys=False)

    return max_return_per_task, max_reward_per_task


def extract_return_stats(trajectories, target_multiplier):
    rewards = np.array([trj["rewards"] for trj in trajectories]).reshape(-1)
    returns = extract_returns(trajectories)
    # compute stats
    reward_stats = pd.DataFrame({"rewards": rewards}).describe()
    return_stats = pd.DataFrame({"returns": returns}).describe()
    stats = pd.concat([reward_stats, return_stats], axis=1).to_dict()
    stats = pd.json_normalize(stats).to_dict(orient="records")[0]
    # compute max return/reward
    max_return, max_reward = max(returns), max(rewards)
    max_return = max_return * target_multiplier if max_return > 0 else max_return / target_multiplier
    max_reward = max_reward * target_multiplier if max_reward > 0 else max_reward / target_multiplier
    return stats, max_return, max_reward


def extract_array_stats(trajectories, kind="actions"):
    vals = np.array([trj[kind] for trj in trajectories])
    stats = {
        "min": np.min(vals),
        "max": np.max(vals),
        "mean": np.mean(vals),
        "std": np.std(vals),
        "q25": np.quantile(vals, 0.25),
        "q50": np.quantile(vals, 0.5),
        "q75": np.quantile(vals, 0.75),
        "q90": np.quantile(vals, 0.9),
        "q99": np.quantile(vals, 0.99),
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", default='mt40_v2_cwnet_2M_local.yaml')
    parser.add_argument("--save_dir", default='../../postprocessing/data_stats')
    parser.add_argument("--target_multiplier", default=1, type=float)
    args = parser.parse_args()
    hydra.initialize(config_path="../../configs")
    conf = hydra.compose(config_name="config",
                         overrides=["agent_params=cdt_clusterembeds",
                                    f"agent_params/data_paths={args.data_paths}"])
    base_path, names = conf.agent_params.data_paths["base"], conf.agent_params.data_paths["names"]
    paths = [str(Path(base_path) / name) for name in names]
    max_return_per_task, max_reward_per_task = extract_stats(
        paths, target_multiplier=args.target_multiplier, save_dir=args.save_dir
    )
    print(max_return_per_task)
    print(max_reward_per_task)


