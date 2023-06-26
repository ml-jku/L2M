import h5py
import pickle
import numpy as np


def discount_cumsum(x, gamma):
    new_x = np.zeros_like(x)
    new_x[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        new_x[t] = x[t] + gamma * new_x[t + 1]
    return new_x

def discount_cumsum_np(x, gamma):
    # much faster version of the above
    new_x = np.zeros_like(x)
    rev_cumsum = np.cumsum(np.flip(x, 0)) 
    new_x = np.flip(rev_cumsum * gamma ** np.arange(0, x.shape[0]), 0)
    new_x = np.ascontiguousarray(new_x).astype(np.float32)
    return new_x


def compute_rtg_from_target(x, target_return):
    new_x = np.zeros_like(x)
    new_x[0] = target_return
    for i in range(1, x.shape[0]):
        new_x[i] = min(new_x[i - 1] - x[i - 1], target_return)
    return new_x


def filter_top_p_trajectories(trajectories, top_p=1):
    start = len(trajectories) - int(len(trajectories) * top_p)
    sorted_trajectories = sorted(trajectories, key=lambda x: np.array(x.get("rewards")).sum())
    return sorted_trajectories[start:]


def filter_trajectories_uniform(trajectories, p=1):
    idx = np.random.randint(0, len(trajectories), int(len(trajectories) * p))
    return [trajectories[i] for i in idx]


def load_npz(path, start_idx=None, end_idx=None): 
    returns_to_go = None
    # trj = np.load(path, mmap_mode="r" if start_idx and end_idx else None)
    with np.load(path, mmap_mode="r" if start_idx and end_idx else None) as trj: 
        if start_idx is not None and end_idx is not None:
            # subtrajectory only
            observations, actions, rewards = trj["states"][start_idx: end_idx].astype(np.float32), \
                trj["actions"][start_idx: end_idx].astype(np.float32), trj["rewards"][start_idx: end_idx].astype(np.float32)
            if "returns_to_go" in trj:
                returns_to_go = trj["returns_to_go"][start_idx: end_idx].astype(np.float32)
        else: 
            # fully trajectory
            observations, actions, rewards = trj["states"], trj["actions"], trj["rewards"], 
            if "returns_to_go" in trj:
                returns_to_go = trj["returns_to_go"].astype(np.float32)
        dones = np.array([trj["dones"]])
    return observations, actions, rewards, dones, returns_to_go


def load_hdf5(path, start_idx=None, end_idx=None):
    returns_to_go = None
    with h5py.File(path, "r") as f:
        if start_idx is not None and end_idx is not None:
            # subtrajectory only
            observations = f['states'][start_idx: end_idx]
            actions = f['actions'][start_idx: end_idx]
            rewards = f['rewards'][start_idx: end_idx]
            if "returns_to_go" in f:
                returns_to_go = f["returns_to_go"][start_idx: end_idx]
        else: 
            # fully trajectory
            observations = f['states'][:]
            actions = f['actions'][:]
            rewards = f['rewards'][:]
            if "returns_to_go" in f:
                returns_to_go = f["returns_to_go"][:]
        dones = np.array([f['dones'][()]])
    return observations, actions, rewards, dones, returns_to_go


def load_pkl(path, start_idx=None, end_idx=None): 
    returns_to_go = None
    with open(path, "rb") as f:
        trj = pickle.load(f)
    if start_idx is not None and end_idx is not None:
        # subtrajectory only
        observations, actions, rewards = trj["states"][start_idx: end_idx], \
            trj["actions"][start_idx: end_idx], trj["rewards"][start_idx: end_idx]
        if "returns_to_go" in trj:
            returns_to_go = trj["returns_to_go"][start_idx: end_idx]
    else: 
        # fully trajectory
        observations, actions, rewards = trj["states"], trj["actions"], trj["rewards"], 
        if "returns_to_go" in trj:
            returns_to_go = trj["returns_to_go"]
    dones = np.array([trj["dones"]])    
    return observations, actions, rewards, dones, returns_to_go
