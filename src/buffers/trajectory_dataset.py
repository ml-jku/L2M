import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .trajectory import Trajectory
from .buffer_utils import load_hdf5, load_npz, load_pkl
from ..algos.agent_utils import make_random_proj_matrix


class TrajectoryDataset(Dataset):

    def __init__(self, trajectories, env, context_len, action_pad,
                 to_rgb=False, trj_lengths=None, trj_sample_kwargs=None,
                 max_state_dim=None, max_act_dim=None, transforms=None):
        """
        Args:
            trajectories: List of Trajectory or Path objects.
            context_len: Int.
            action_pad: Int.
            to_rgb: Bool. Whether to convert gray-scale images to RGB format, by repeating the gray-scale channel.
            trj_lengths: Dict. Maps trj paths to respective lengths.
            max_state_dim: None or Int. For continuous observations (e.g., metaworld, dmcontrol) determines 
                if the obs sequence should be padded to max_state_dim. If None, no padding is done.
            max_act_dim: None or Int. For continous actions (e.g., metaworld, dmcontrol) determines if the
                action sequence should be padded to max_act_dim. If None, no padding is done.
            transforms: None or callable. If callable, it is applied to each state sample.
            
        """
        self.trajectories = trajectories
        self.context_len = context_len
        self.action_pad = action_pad
        self.env = env
        self.max_state_dim = max_state_dim
        self.max_act_dim = max_act_dim
        self.to_rgb = to_rgb
        self.transforms = transforms
        self.trj_lengths = trj_lengths if trj_lengths is not None else {}
        self.trj_sample_kwargs = trj_sample_kwargs if trj_sample_kwargs is not None else {}

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        if isinstance(trj, (str, Path)):
            # load from disk
            path = str(trj)
            s, s1, a, r, togo, t, done, task_id, trj_id = self.get_sample_from_disk(path, idx)
        else:
            # samples stored in memory, load from there
            s, s1, a, r, togo, t, done, task_id, trj_id = self.get_sample_from_memory(trj)
        
        # postprocess states, actions
        if len(s.shape) == 4 and self.to_rgb:
            # convert to "RGB" by repeating the gray-scale channel
            s = np.repeat(s, 3, axis=1)
            s1 = np.repeat(s1, 3, axis=1)
        if self.env is not None:
            s = self.env.normalize_obs(s)
            s1 = self.env.normalize_obs(s1)
        # pad if necessary + create mask
        padding = self.context_len - s.shape[0]
        mask = np.concatenate([np.zeros(padding), np.ones(s.shape[0])], axis=0)
        action_mask = np.ones_like(a, dtype=np.int32)
        if self.max_state_dim is not None and len(s.shape) == 2:
            s, s1 = self.pad_states(s, s1)
            # rand_proj_mat = make_random_proj_matrix(s.shape[-1], self.max_state_dim)
            # s, s1 = s @ rand_proj_mat.T, s1 @ rand_proj_mat.T           
        if self.max_act_dim is not None and a.dtype.kind == "f": 
            a, action_mask = self.pad_actions(a)
        if padding:
            s, s1, a, r, togo, t, done, action_mask = self.pad_sequences(s, s1, a, r, togo, t, 
                                                                         done, action_mask, padding)
        if len(s.shape) == 4 and self.transforms is not None:
            # perform image augmentations
            s = self.transforms(torch.from_numpy(s).float())
            s1 = self.transforms(torch.from_numpy(s1).float())
        
        return s, a, s1, np.expand_dims(r, axis=1), np.expand_dims(togo, axis=1), \
               t, mask, done, task_id, trj_id, action_mask
               
    def get_sample_from_memory(self, trj): 
        s, s1, a, r, togo, t, done, task_id, trj_id = trj.sample(self.context_len)
        return s, s1, a, r, togo, t, done, task_id, trj_id
    
    def get_sample_from_disk(self, path, idx): 
        # load trj file from disk
        if self.trj_lengths: 
            # (faster)
            # directly load subset of trajectory from disk making use of trj_lengths
            upper_bound = self.trj_lengths[path]
            start_idx = np.random.randint(0, upper_bound, size=1)[0]
            end_idx = min(start_idx + self.context_len, upper_bound)
            s, a, r, done_flag, togo = self.load_trj(path, start_idx=start_idx, end_idx=end_idx)
            assert togo is not None, "RTGs must be stored in trj file."
            r = r.astype(np.float32)
            if len(a.shape) == 1: 
                a = np.expand_dims(a, -1)
            if isinstance(done_flag, (list, tuple, np.ndarray)):
                done_flag = done_flag[..., -1]
            done = np.zeros(len(s))
            done[-1] = done_flag
            s1 = np.zeros_like(s)
            t = np.arange(start_idx, end_idx)
            task_id, trj_id = 0, idx
        else: 
            # (slow)
            # unknown trajectory length: load full trajectory and sample subsequence
            observations, actions, rewards, dones, returns_to_go = self.load_trj(path)
            obs_shape, act_dim = observations.shape[1:], actions.shape[-1]
            trajectory = Trajectory(obs_shape, act_dim, max_len=len(observations),
                                    init_trj_buffers=False, **self.trj_sample_kwargs)
            if len(actions.shape) == 1: 
                actions = np.expand_dims(actions, -1)
            trajectory.add_full_trj(obs=observations, next_obs=None, action=actions, reward=rewards,
                                    done=dones, task_id=0, trj_id=idx, returns_to_go=returns_to_go)
            trajectory.setup_final_trj(compute_stats=False)
            # sample subsequence
            s, s1, a, r, togo, t, done, task_id, trj_id = trajectory.sample(self.context_len)
            
        return s, s1, a, r, togo, t, done, task_id, trj_id
    
    def load_trj(self, path, start_idx=None, end_idx=None): 
        if path.endswith('.npz'):
            observations, actions, rewards, dones, returns_to_go = load_npz(path, start_idx=start_idx, end_idx=end_idx)
        elif path.endswith('.hdf5'):
            observations, actions, rewards, dones, returns_to_go = load_hdf5(path, start_idx=start_idx, end_idx=end_idx)
        elif path.endswith('.pkl'):
            observations, actions, rewards, dones, returns_to_go = load_pkl(path, start_idx=start_idx, end_idx=end_idx)
        else: 
            raise ValueError("Only .npz and .hdf5 files are supported.")
        return observations, actions, rewards, dones, returns_to_go
        
    def pad_sequences(self, s, s1, a, r, togo, t, done, action_mask, padding):
        # obs is either 4 dimensional (image) or 2 dimensional (state input) 
        # first dimension is the sequence length
        obs_shape, act_dim = s.shape[1:], a.shape[-1]
        s = np.concatenate([np.zeros((padding, *obs_shape), dtype=s.dtype), s], axis=0)
        s1 = np.concatenate([np.zeros((padding, *obs_shape), dtype=s1.dtype), s1], axis=0)
        a = np.concatenate([np.ones((padding, act_dim), dtype=a.dtype) * self.action_pad, a], axis=0)
        r = np.concatenate([np.zeros((padding), dtype=r.dtype), r], axis=0)
        togo = np.concatenate([np.zeros((padding), dtype=togo.dtype), togo], axis=0)
        t = np.concatenate([np.zeros((padding), dtype=t.dtype), t], axis=0)
        done = np.concatenate([np.zeros((padding), dtype=done.dtype), done], axis=0)
        action_mask = np.concatenate([np.zeros((padding, act_dim), dtype=action_mask.dtype), action_mask], axis=0)
        return s, s1, a, r, togo, t, done, action_mask
    
    def pad_states(self, s, s1):
        # pad state input to max_state_dim in case of continuous state only
        padding = self.max_state_dim - s.shape[-1]
        s = np.concatenate([s, np.zeros((s.shape[0], padding), dtype=s.dtype)], axis=-1)
        s1 = np.concatenate([s1, np.zeros((s.shape[0], padding), dtype=s1.dtype)], axis=-1)
        return s, s1

    def pad_actions(self, a):
        # pad action to max_act_dim in case of continuous actions only
        padding = self.max_act_dim - a.shape[-1]
        action_mask = np.concatenate([np.ones_like(a), np.zeros((a.shape[0], padding))], axis=-1)
        a = np.concatenate([a, np.ones((a.shape[0], padding), dtype=a.dtype) * self.action_pad], axis=-1)
        return a, action_mask
