import collections
import numpy as np
import heapq
import torch
from pathlib import Path
from tqdm import tqdm
from .trajectory import Trajectory
from .trajectory_buffer import TrajectoryReplayBuffer, TrajectoryReplayBufferSamples
from .buffer_utils import filter_top_p_trajectories, filter_trajectories_uniform


class PromptBuffer(TrajectoryReplayBuffer):

    def __init__(self, buffer_size, observation_space, action_space,
                 num_seq_per_prompt=1, num_trjs_per_task=5, multitask_batch=False,
                 sample_full_seqs_only=False, **kwargs):
        """

        Args:
            buffer_size: Int.
            observation_space: Gym observation space.
            action_space: Gym action space.
            num_seq_per_prompt: Int. Number of sequences to concatenate per sampled prompt.
            num_trjs_per_task: Int. Number of trajectories to store per task.
            sample_full_seqs_only: Bool. Whether to only sample full sequences from trajectories, i.e., no padding.
            **kwargs:

        """
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        self.num_seq_per_prompt = num_seq_per_prompt
        self.num_trjs_per_task = num_trjs_per_task
        self.multitask_batch = multitask_batch
        self.sample_full_seqs_only = sample_full_seqs_only
        if self.as_heap:
            # ensures that the best trajectories always only stay in the buffer.
            self._trajectories = collections.defaultdict(list)
            self.trajectories_heap = collections.defaultdict(list)
        else:
            self._trajectories = collections.defaultdict(lambda: collections.deque(maxlen=self.num_trjs_per_task))

    def add_trajectory(self, trajectory, task_id):
        """
        Takes one or multiple complete trajectories and appends to task buffer.

        Args:
            trajectory: Trajectory.

        """
        if not isinstance(trajectory, list):
            trajectory = [trajectory]
        for trj in trajectory:
            if self.as_heap:
                if len(self.trajectories[task_id]) >= self.num_trjs_per_task:
                    _ = heapq.heappushpop(self.trajectories_heap[task_id], (trj.total_return, trj))
                else:
                    heapq.heappush(self.trajectories_heap[task_id], (trj.total_return, trj))
                self.trajectories[task_id] = [t[1] for t in self.trajectories_heap[task_id]]
            else:
                self.trajectories[task_id].append(trj)

    def sample(self, batch_size=5, env=None, top_k=5, weight_by="len", task_id=None, trj_id=None):
        # deliberately sample batch_size * num_seq_per_prompt sequences, as a single prompt may contain multiple seqs
        prompt_samples = self.sample_batch(batch_size * self.num_seq_per_prompt, env, top_k,
                                         weight_by, task_id, trj_id)
        if self.num_seq_per_prompt > 1:
            stacked_prompt = []
            # exclude task_ids, trj_ids as only have one dimension
            for seq in prompt_samples:
                # ensure that all prompts have num_seq_per_prompt sequences
                if len(seq.shape) > 1:
                    stacked_prompt.append(seq.reshape(seq.shape[0] // self.num_seq_per_prompt,
                                                      seq.shape[1] * self.num_seq_per_prompt, *seq.shape[2:]))
                else:
                    stacked_prompt.append(seq)
            return TrajectoryReplayBufferSamples(*stacked_prompt)
        return prompt_samples

    def sample_batch(self, batch_size=5, env=None, top_k=5, weight_by="len", task_id=None, trj_id=None):
        if isinstance(task_id, torch.Tensor):
            # samples within batch come from different tasks.
            # need to iteratively make the trajectory set to sample subsequences from (slower)
            trajectories = []
            task_id, trj_id = task_id.cpu().numpy(), trj_id.cpu().numpy()
            for task, trj in zip(task_id, trj_id):
                task_trjs = self.trajectories[task]
                valid_trjs = np.setdiff1d(np.arange(0, len(task_trjs)), np.array([trj]))
                trj_inds = np.random.choice(valid_trjs, size=self.num_seq_per_prompt, replace=True)
                for i in trj_inds:
                    trajectories.append(task_trjs[i])
            trajectory_inds = np.arange(len(trajectories))
        else:
            # samples within batch come from same task.
            trajectories = self.trajectories[task_id]
            upper_bound = len(trajectories)
            trajectory_probs = None
            trajectory_inds = np.random.choice(np.arange(0, upper_bound), size=batch_size, p=trajectory_probs, replace=True)
        return self._get_samples(trajectories, trajectory_inds, env=env)
    
    def _get_samples(self, trajectories, trajectory_inds, env=None, augment=False, aug_std=0.01, p_aug=0.5):
        obs, next_obs, action, reward, reward_to_go = [], [], [], [], []
        timesteps, attn_mask, dones, task_ids, trj_ids = [], [], [], [], []
        for idx in trajectory_inds:
            s, s1, a, r, togo, t, done, task_id, trj_id = trajectories[idx].sample(self.context_len)
            if augment:
                if np.random.rand() < p_aug:
                    # i.e. same noise for each dimension
                    s = s + np.random.normal(0, aug_std, (s.shape[0], 1))
                    a = a + np.random.normal(0, aug_std, (a.shape[0], 1)).astype(a.dtype)

            # pad s,s1, a, r togo if necessary
            padding = self.context_len - s.shape[0]
            attn_mask.append(np.concatenate([np.zeros(padding), np.ones(s.shape[0])], axis=0))
            if padding:
                obs_shape, act_dim = s.shape[1:], a.shape[-1]
                s = np.concatenate([np.zeros((padding, *obs_shape), dtype=s.dtype), s], axis=0)
                s1 = np.concatenate([np.zeros((padding, *obs_shape), dtype=s1.dtype), s1], axis=0)
                a = np.concatenate([np.ones((padding, act_dim), dtype=a.dtype) * self.action_pad, a], axis=0)
                r = np.concatenate([np.zeros((padding), dtype=r.dtype), r], axis=0)
                togo = np.concatenate([np.zeros((padding), dtype=togo.dtype), togo], axis=0)
                t = np.concatenate([np.zeros((padding), dtype=t.dtype), t], axis=0)
                done = np.concatenate([np.zeros((padding), dtype=done.dtype), done], axis=0)
            obs.append(s)
            next_obs.append(s1)
            action.append(a)
            reward.append(r)
            reward_to_go.append(togo)
            timesteps.append(t)
            dones.append(done)
            task_ids.append(task_id)
            trj_ids.append(trj_id)
        obs = np.array(obs)
        next_obs = np.array(next_obs)
        attn_mask = np.array(attn_mask)
        # normalize after padding --> faster
        if env is not None:
            obs = env.normalize_obs(obs)
            next_obs = env.normalize_obs(next_obs)
            # add padding again
            obs[~attn_mask.astype(bool)] = 0
            next_obs[~attn_mask.astype(bool)] = 0
        data = (
            obs,
            np.array(action),
            next_obs,
            np.expand_dims(np.array(reward), axis=2),
            np.expand_dims(np.array(reward_to_go), axis=2),
            np.array(timesteps),
            attn_mask,
            np.array(dones),
            np.array(task_ids),
            np.array(trj_ids),
            np.ones_like(action)
        )
        if self.seqs_per_sample > 1:
            # reshape samples so that each has seqs_per_sample sequences per sample.
            # original tensor: [batch_size, context_len, ...]
            # reshaped tensor: [batch_size / seqs_per_sample, context_len * seqs_per_sample, ...]
            assert len(trajectory_inds) % self.seqs_per_sample == 0 and self.context_len % self.seqs_per_sample == 0
            new_samples = []
            for s in data:
                if len(s.shape) > 1:
                    new_samples.append(s.reshape(s.shape[0] // self.seqs_per_sample,
                                                 s.shape[1] * self.seqs_per_sample, *s.shape[2:]))
                else:
                    new_samples.append(s.reshape(s.shape[0] // self.seqs_per_sample, self.seqs_per_sample))
            data = new_samples

        return TrajectoryReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def is_empty(self, task_id):
        return len(self.trajectories[task_id]) == 0

    def init_buffer_from_dataset(self, paths):
        """
        Overwrite
        """
        print(f"Intitializing prompt buffer from data paths.")
        base_path, names = paths["base"], paths["names"]
        if isinstance(names, str):
            names = [names]
        paths = [Path(base_path) / name for name in names]
        states_store = []
        store_states = len(paths) == 1 and self.store_state_stats
        for task_id, path in enumerate(paths):
            print(f"Loading trajectories from: {path}")
            trajectories = self.load_trajectory_dataset(path)

            if self.init_top_p < 1:
                trajectories = filter_top_p_trajectories(trajectories, top_p=self.init_top_p)
            elif self.init_p < 1:
                trajectories = filter_trajectories_uniform(trajectories, p=self.init_p)

            for trj in tqdm(trajectories, desc="Storing trajectories"):
                if self.full:
                    break
                observations, next_observations, actions, rewards, dones, trj_id = trj["observations"],\
                    trj["next_observations"], trj["actions"], trj["rewards"], trj["terminals"], trj["trj_id"]
                if len(observations) < self.context_len \
                        or len(actions) < self.context_len or len(rewards) < self.context_len \
                        or len(dones) < self.context_len:
                        # or len(next_observations) < self.context_len \
                    continue

                trajectory = Trajectory(
                    None, None, self.max_len, relative_pos_embds=self.relative_pos_embds,
                    task_id=task_id, sample_full_seqs_only=self.sample_full_seqs_only, trj_id=trj_id,
                    init_trj_buffers=False
                )
                trajectory.add_full_trj(
                    np.vstack(observations),
                    np.vstack(next_observations) if next_observations is not None else None,
                    np.vstack(actions),
                    np.stack(rewards).reshape(-1),
                    np.stack(dones).reshape(-1),
                    task_id=task_id,
                    trj_id=trj_id
                )
                trajectory.setup_final_trj(target_return=self.target_return)
                self.add_trajectory(trajectory, task_id)
                if store_states:
                    states_store.append(np.vstack(observations))

        if store_states:
            # don't use normalization for a replay buffer consisting of mixture of datasets
            self.state_mean = np.vstack(states_store).mean(axis=0)
            # add very small epsilon to ensure it's not 0
            self.state_std = np.vstack(states_store).std(axis=0) + 1e-8

    def _get_buffer_stats(self, prefix="prompt_buffer"):
        super()._get_buffer_stats(prefix=prefix)
