import os
import collections
from .trajectory_buffer import TrajectoryReplayBuffer
from .samplers import DomainWeightedRandomSampler, DistributedSamplerWrapper, MixedBatchRandomSampler
from .trajectory import Trajectory


class MultiDomainTrajectoryReplayBuffer(TrajectoryReplayBuffer): 
    
    def __init__(self, buffer_size, observation_space, action_space, mixed=False, mixed_weighted=False,
                 domain_weights=None, **kwargs):
        """
        A trajectory replay buffer that can handle trajectories from multiple domains
        Different domains have different observation spaces and action spaces.
        Data is loaded from different data paths.
        When sampling, each batch can only contain trajectories from one domain (or from domains with same shapes).
        Otherwise the batch collating will fail.
        This buffer should only be used in offline mode for pre-training.
        It assumes that all trajectories from a particular domain are either in memrory or on disk, but no mixture.
        
        Args:
            buffer_size (int): size of the buffer
            observation_space (gym.Space): observation space
            action_space (gym.Space): action space
            mixed: Bool. Whether batches contain sequences from multiple domains.
            mixed_weighted: Bool. Whether to weight the samples in each batch per domain and by length. 
            domain_weights: None or Dict.
        
        """	
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        # domain specific 
        self.mixed = mixed
        self.domain_id = 0
        self.domain_to_indices = collections.defaultdict(list)
        self.domain_weights = domain_weights
        self.fixed_domain_weights = domain_weights is not None
        self.mixed_weighted = mixed_weighted if domain_weights is None else True
    
    def compute_trajectory_probs(self, top_k=5, weight_by="len"):
        if self.mixed: 
            return super().compute_trajectory_probs(top_k=top_k, weight_by=weight_by)
        # only supports weighting by len or uniform weighting
        if self.trj_ds_has_changed or self.trajectory_probs is None:
            self.trajectory_probs = {}
            if weight_by == "uniform":
                for i, indices in self.domain_to_indices.items(): 
                    num_trjs = len(indices)
                    self.trajectory_probs[i] = [1 / num_trjs] * num_trjs
            elif weight_by == "len":
                for i, indices in self.domain_to_indices.items(): 
                    trj_lens = [len(self.trajectories[idx]) if isinstance(self.trajectories[idx], Trajectory)
                                else self.trajectory_lengths[str(self.trajectories[idx])] 
                                for idx in indices]
                    total_samples = sum(trj_lens)
                    self.trajectory_probs[i] = [l / total_samples for l in trj_lens]
            else: 
                raise NotImplementedError()
        return self.trajectory_probs
   
    def make_sampler(self, dataset, trajectory_probs, batch_size): 
        if self.mixed: 
            return super().make_sampler(dataset, trajectory_probs, batch_size)
        
        mult = 100 if not self.ddp else 30
        batch_size = batch_size if not self.ddp else batch_size * int(os.environ["WORLD_SIZE"])
        
        if self.mixed_weighted: 
            # mix batches in proporition to domain  
            if (self.trj_ds_has_changed or self.domain_weights is None) and not self.fixed_domain_weights:
                total_samples_per_domain = {}
                for i, indices in self.domain_to_indices.items(): 
                    total_samples_per_domain[i] = sum([
                        len(self.trajectories[idx]) if isinstance(self.trajectories[idx], Trajectory)
                        else self.trajectory_lengths[str(self.trajectories[idx])] 
                        for idx in indices
                    ])
                total_samples = sum(total_samples_per_domain.values())
                self.domain_weights = {i: total_samples_per_domain[i] / total_samples for i in total_samples_per_domain}
            sampler = MixedBatchRandomSampler(weights=trajectory_probs, domain_weights=self.domain_weights, 
                                              batch_size=batch_size, num_samples=len(dataset) * mult, replacement=True)             
        else:
            sampler = DomainWeightedRandomSampler(weights=trajectory_probs, batch_size=batch_size, 
                                                  num_samples=len(dataset) * mult, replacement=True)
        if self.ddp: 
            sampler = DistributedSamplerWrapper(sampler)
        return sampler 
      
    def init_buffer_from_dataset(self, paths):
        assert isinstance(paths, (list, tuple, dict))
        if isinstance(paths, dict): 
            paths = list(paths.values())
        for i, p in enumerate(paths): 
            self.domain_id = i
            start_idx = len(self)
            super().init_buffer_from_dataset(p)
            end_idx = len(self)
            self.domain_to_indices[i] = list(range(start_idx, end_idx))
