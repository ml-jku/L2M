"""
We use datasets provided by RL Unplugged: 
 - https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged
 
Unfortunately, vD4RL does not provide a programatic approach for downloading the datsets.
We download them manually, as described in the README. This script provides funtionality for postprocessing the
downloaded data and prepare them in the right format for dataloading. 

"""
import pickle
import collections
import argparse
import json
import h5py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def read_hdf5_file(path): 
    with h5py.File(path, "r") as f:
        observations = f['observation'][:]
        actions = f['action'][:]
        rewards = f['reward'][:]
        step_types = f['step_type'][:]
        discount = f['discount'][:]
        assert np.all(discount == 1.0), "Discount is not 1.0 for all transitions." 
    return observations, actions, rewards, step_types


def discount_cumsum_np(x, gamma):
    new_x = np.zeros_like(x)
    rev_cumsum = np.cumsum(np.flip(x, 0)) 
    new_x = np.flip(rev_cumsum * gamma ** np.arange(0, len(x)), 0)
    new_x = np.ascontiguousarray(new_x).astype(np.float32)
    return new_x

    
def extract_episodes_from_slice(path, grayscale=False):
    # setup
    trajectories = []
    trj_id = 0
    current_trj = collections.defaultdict(list)
    
    # read hdf5
    observations, actions, rewards, step_types = read_hdf5_file(path)
    
    # extract trajectories
    for s, a, r, stype in tqdm(zip(observations, actions, rewards, step_types),
                                  total=len(observations), desc="Extracting trajectories"):
        nans = [np.isnan(s).any(), np.isnan(a).any(), np.isnan(r), np.isnan(stype)]
        if any(nans):
            print("NaNs found:", nans)
        stype = int(stype)
        if stype == 0:
            assert len(current_trj["observations"]) == 0, "Trajectory not empty. Can't be start of episode"
        done = stype == 2
        current_trj["actions"].append(a)
        current_trj["rewards"].append(r)
        current_trj["terminals"].append(done)
        if grayscale:
            # states are [channel, height, width] --> convert to [height, width, channel] to make cv2 happy
            # then convert to grayscale using cv2 (as used in sb3 atari wrappers)
            s_gray = cv2.cvtColor(s.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            current_trj["observations"].append(np.expand_dims(s_gray, axis=0))
        else: 
            current_trj["observations"].append(s)
        if done:
            current_trj["trj_id"] = trj_id
            trajectories.append(current_trj)
            current_trj = collections.defaultdict(list)
            trj_id += 1
    return trajectories    


def save_episodes(episodes, save_dir, start_idx=0, save_format="hdf5", add_rtgs=False,
                  max_episodes=None, max_transitions=None, num_collected_transitions=None):
    """
    Saves episodes to desired save_dir.
    Args:
        episodes: List of collections.defaultdicts containint the episodes.
        save_dir: pathlib.Path. Desired location to save individual episodes.
        save_format: Str.

    """
    print(f"Saving episodes to {str(save_dir)}.")
    ep_lengths = {}
    for i, episode in enumerate(tqdm(episodes, desc="Saving episodes")):
        ep_idx = start_idx + i
        if max_episodes is not None and ep_idx > max_episodes:
            break
        if max_transitions is not None and num_collected_transitions + len(episode) > max_transitions:
            break
        observations, actions, rewards, terminals = episode["observations"], episode["actions"], episode["rewards"], \
            episode["terminals"]
        to_save = {
            "states": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": terminals
        }
        if add_rtgs: 
            rtgs = discount_cumsum_np(rewards, 1)
            to_save["returns_to_go"] = rtgs
        
        file_name = str(start_idx + i).zfill(9)
        save_path = str(save_dir / file_name)
        ep_lengths[file_name] = len(observations)
        if save_format == "hdf5":
            with h5py.File(save_path + ".hdf5", "w") as f:
                f.create_dataset('states', data=observations)
                f.create_dataset('actions', data=actions)
                f.create_dataset('rewards', data=rewards)
                f.create_dataset('dones', data=terminals)
                if add_rtgs: 
                    f.create_dataset('returns_to_go', data=rtgs)
        elif save_format == "npzc": 
            np.savez_compressed(save_path, **to_save)
        elif save_format == "pkl": 
            with open(save_path + ".pkl", "wb") as f:
                pickle.dump(to_save, f)
        else: 
            np.savez(save_path, **to_save)
    return ep_lengths



def load_and_save_dmcontrol_episodes(data_dir, save_dir, save_format="hdf5", add_rtgs=False, grayscale=False,
                                     max_episodes=None, max_transitions=None):
    """
    Loads, saves and prepares the desired number of episodes/transitions for the specified DMControl tasks.
    We use datasets provided by vD4RL: 
        - https://github.com/conglu1997/v-d4rl

    After loading a slice, the episodes are saved to the given save_dir.
    To avoid running into memory issues, slices are loaded & saved sequentially.
    
    The downloaded .hdf5 files have the following attributes: 
    ```
    ['action', 'discount', 'observation', 'reward', 'step_type']

    ```
    The step type can be 0, 1, or 2. With 0=startofepisode, 1=mid, 2=endofepisode, according to: 
    - https://github.com/deepmind/dm_env/blob/91b46797fea731f80eab8cd2c8352a0674141d89/dm_env/_environment.py#L66
    
    Thus, we recover the episodes. 

    Files are saved as follows:
    ```
    environment family (e.g., dm_control)
    - environment name (e.g. cheetah_run)
    -- one hdf5 file (numbered, zero padded to 9 digits) per episode with fields: states, actions, rewards, dones

    ```

    Args:
        data_dir: Str. Path to data directory.
        save_dir: Str. Path to save directory.
        max_episodes: Int or None.
        max_transitions: Int or None.
        save_format: Str. File format to save episodes in.
        add_rtgs: Bool. Whether to add RTGs to the episode files.
        grayscale: Bool. Whether to convert the observations to grayscale.

    """
    num_collected_transitions, num_collected_episodes = 0, 0
    all_epname_to_len = {}
    ep_lens, ep_returns = [], []
    slice_paths = [p for p in data_dir.glob("*.hdf5")]
    print(f"Collecting episodes from {len(slice_paths)} slices in {data_dir}")

    # collect and save episodes
    for p in tqdm(slice_paths, desc="Loading slices"):
        # load slice + extract episodes in it    
        episodes = extract_episodes_from_slice(p, grayscale=grayscale)
        
        # save invidiual episodes 
        epname_to_len = save_episodes(
            episodes, save_dir, start_idx=num_collected_episodes, max_episodes=max_episodes,
            num_collected_transitions=num_collected_transitions, max_transitions=max_transitions,
            save_format=save_format, add_rtgs=add_rtgs
        )
        
        # store episode lengths 
        all_epname_to_len.update(epname_to_len)
        num_collected_episodes += len(epname_to_len.keys())
        # todo: num collected transistions, ep_returns is not actually correct since, we don't include all of the episodes
        num_collected_transitions += sum([len(ep) for ep in episodes])
        ep_lens += [v for v in epname_to_len.values()]
        ep_returns += [sum(ep["rewards"]) for ep in episodes]
        if max_episodes is not None and num_collected_episodes > max_episodes:
            print("Max number of episodes reached.")
            break
        if max_transitions is not None and num_collected_transitions > max_transitions:
            print("Max number of transitions reached.")
            break 

    # compute and dumpy episode stats
    stats = {
        "slices": len(slice_paths), "episodes": num_collected_episodes, "transitions": num_collected_transitions,
        "mean_episode_len": np.mean(ep_lens).round(decimals=4).item(),
        "min_episode_len": np.min(ep_lens).round(decimals=4).item(),
        "max_episode_len": np.max(ep_lens).round(decimals=4).item(),
        "mean_episode_return": np.mean(ep_returns).round(decimals=4).item(),
        "min_episode_return": np.min(ep_returns).round(decimals=4).item(),
        "max_episode_return": np.max(ep_returns).round(decimals=4).item()
    }
    print(" | ".join([f"{k}: {v}" for k, v in stats.items()]))
    with open(save_dir / "stats.json", "w") as f:
        json.dump(stats, f)
    with open(save_dir / "episode_lengths.json", "w") as f:
        json.dump(all_epname_to_len, f)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', nargs='+', default=['cheetah_run'])
    parser.add_argument("--data_dir", type=str,
                        default="/home/thomas/Projects-Linux/CRL_with_transformers/data/visual_dm_control_suite")
    parser.add_argument("--save_dir", type=str, 
                        default="/home/thomas/Projects-Linux/CRL_with_transformers/data/visual_dm_control_suite_proc")
    parser.add_argument('--save_format', type=str, default="hdf5", help="File format to save episodes in.")
    parser.add_argument('--quality', type=str, default="medium_expert",
                        help="Determines the data quality to use (as provided by authors).")
    parser.add_argument('--resolution', type=str, default="84px",
                        help="Determines the resolution of the image observations. Data is stored in 84px or 64px folders.")
    parser.add_argument('--max_episodes', type=int, help="Max episodes to use per task.")
    parser.add_argument('--max_transitions', type=int, help="Max transitions to use per task.")
    parser.add_argument('--add_rtgs', action="store_true", help="Wheter to precompute and add return-to-gos to files.")
    parser.add_argument('--grayscale', action="store_true", help="Wheter to convert image observations to grayscale.")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    for env in args.envs:
        load_dir = data_dir / env / args.quality / args.resolution
        save_dir = Path(args.save_dir)  / env 
        save_dir.mkdir(exist_ok=True, parents=True)
        load_and_save_dmcontrol_episodes(data_dir=load_dir,
                                         save_dir=save_dir,
                                         max_episodes=args.max_episodes,
                                         max_transitions=args.max_transitions,
                                         save_format=args.save_format, 
                                         add_rtgs=args.add_rtgs, 
                                         grayscale=args.grayscale)
