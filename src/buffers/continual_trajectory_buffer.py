import json
import pickle
import collections
import numpy as np
from pathlib import Path
from .trajectory_buffer import TrajectoryReplayBuffer, ReplayBuffer


class ContinualTrajectoryReplayBuffer(TrajectoryReplayBuffer):

    def __init__(self, buffer_size, observation_space, action_space, num_tasks=10, **kwargs):
        """
        Designed to be used in offline-model only.
        Uses the trajectories-property to select trajectories for the current task.

        """
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        assert not self.as_heap, "ContinualReplayBuffer only works with as_heap=False"
        self.num_tasks = num_tasks
        self._trajectories = collections.defaultdict(
            lambda: collections.deque(maxlen=self.buffer_size // self.num_tasks)
        )
        self._trajectory_lengths = collections.defaultdict(dict)

    @property
    def trajectories(self):
        return self._trajectories[self.task_id]
    
    @property
    def trajectory_lengths(self):
        return self._trajectory_lengths[self.task_id]
    
    def load_trajectory_dataset(self, path):
        assert isinstance(path, Path), "Path must be a Path object."
        if path.suffix == ".pkl":
            with open(str(path), 'rb') as f:
                obj = pickle.load(f)
            if isinstance(obj, ReplayBuffer):
                trajectories = self.extract_trajectories_from_buffer(obj)
            else:
                trajectories = obj
        elif path.suffix == ".npz" or path.suffix == ".npy":
            obj = np.load(str(path))
            trajectories = self.extract_trajectories_from_npz(obj)
        elif path.is_dir():
            if self.from_disk: 
                trj_files = sorted([p for p in path.glob("*.npz")])
                trj_files += sorted([p for p in path.glob("*.hdf5")])
                trj_files += sorted([p for p in path.glob("*.pkl")])
                self._trajectories[self.task_id] += trj_files

                lengths_path = path / "episode_lengths.json"
                if lengths_path.exists():
                    with open(lengths_path, "r") as f:
                        name_to_len = json.load(f)
                    for p in trj_files: 
                        self._trajectory_lengths[self.task_id][str(p)] = name_to_len[p.stem]
                trajectories = None
            else: 
                trajectories = self.extract_trajectories_from_dir(path)
        else:
            raise NotImplementedError("Unsupported file type.")
        return trajectories
    
    
class CMDTrajectoryReplayBuffer(ContinualTrajectoryReplayBuffer):
    """
    Trajectory replay buffer that supports initing buffer from multi-domain dataset config. 
    
    """
    def __init__(self, buffer_size, observation_space, action_space, num_tasks=16, **kwargs):
        """
        Designed to be used in offline-model only.
        Uses the trajectories-property to select trajectories for the current task.

        """
        super().__init__(buffer_size, observation_space, action_space, num_tasks, **kwargs)
        assert not self.as_heap, "ContinualReplayBuffer only works with as_heap=False"
        # don't specify maxlen for multi-domain case
        self._trajectories = collections.defaultdict(
            lambda: collections.deque()
        )
        self._trajectory_lengths = collections.defaultdict(dict)
    
    def init_buffer_from_dataset(self, paths):
        assert isinstance(paths, (list, tuple, dict))
        if isinstance(paths, dict): 
            paths = list(paths.values())
        tasks_so_far = 0
        for p in paths: 
            super().init_buffer_from_dataset(p)
            tasks_so_far += len(p["names"])
            self.task_id = tasks_so_far
        self.set_task_id(0)
