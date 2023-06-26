import math
import torch
import torch.nn as nn
from .l2p import L2PPrompt


class L2MIA3(L2PPrompt):

    def __init__(self, length=6, top_k=1, dropout_rate=0.0, log_mod_stats=False, exclude_k=False,
                 exclude_v=False, exclude_ff=False, **kwargs):
        super().__init__(length=length, top_k=top_k, dropout_rate=dropout_rate, **kwargs)
        self.log_mod_stats = log_mod_stats
        self.exclude_k = exclude_k
        self.exclude_v = exclude_v
        self.exclude_ff = exclude_ff

    def _setup_prompt(self):
        if self.n_layer is None:
            self.prompt = nn.Parameter(torch.ones((self.pool_size, self.length, self.embed_dim)))
        else:
            self.prompt = nn.Parameter(torch.ones((self.pool_size, self.n_layer, self.length, self.embed_dim)))

    def extract_prompt(self, idx):
        """
        Extract prompt from batched prompt raw.
        Allows for multitask batches. We unsqueeze across the second vector dimension to ensure broadcasting works
        along the sequence length dimension.

        Args:
            idx: torch.Tensor. Indices to lookup.

        """
        # [batch_size x n_layer x length x embed_dim] or [batch_size x length x embed_dim]
        batched_prompt_raw = self.prompt[idx]
        if self.n_layer is None:
            # [batch_size x 1 x length x embed_dim]
            batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, length, embed_dim).unsqueeze(1)
            # --> tuple of length 3 with first two elements of shape [batch_size x 1 x embed_dim]
            # and last element of shape [batch_size x 1 x 4 * embed_dim]
            vectors = [batched_prompt[:, :, 0], batched_prompt[:, :, 1], batched_prompt[:, :, 2:].flatten(-2)]
        else:
            # [batch_size x 1 x n_layer x length x embed_dim]
            batch_size, top_k, n_layer, length, embed_dim = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, n_layer, length, embed_dim)
            # --> n_layer tuples of length 3 with first two elements of shape [batch_size x 1 x embed_dim]
            # and last element of shape [batch_size x 1 x 4 * embed_dim]
            vectors = [(p[:, :, 0], p[:, :, 1], p[:, :, 2:].flatten(-2)) for p in batched_prompt.split(1, dim=1)]

        stats = {}
        if self.log_mod_stats:
            for i, vec in enumerate(vectors):
                stats[f"mod_k_mean_{i}"] = round(vec[0].mean().item(), 3)
                stats[f"mod_k_std_{i}"] = round(vec[0].std().item(), 3)
                stats[f"mod_v_mean_{i}"] = round(vec[1].mean().item(), 3)
                stats[f"mod_v_std_{i}"] = round(vec[1].std().item(), 3)
                stats[f"mod_ff_mean_{i}"] = round(vec[2].mean().item(), 3)
                stats[f"mod_ff_std_{i}"] = round(vec[2].std().item(), 3)

        # turn off modulation vectors
        if any([self.exclude_v, self.exclude_k, self.exclude_ff]):
            # iterate each layers modulation vectors
            for i, vecs in enumerate(vectors):
                new_vecs = list(vecs)
                if self.exclude_v:
                    new_vecs[0] = None
                if self.exclude_k:
                    new_vecs[1] = None
                if self.exclude_ff:
                    new_vecs[2] = None
                vectors[i] = tuple(new_vecs)
                
        return vectors, stats    

    def add_dropout(self, batched_prompt):
        return batched_prompt
    

class L2MLoRA(L2PPrompt):

    def __init__(self, length=2, top_k=1, dropout_rate=0.0, init_prompts="zeros",
                 rank=4, mod_q=True, mod_v=True, mod_k=False, mod_ff=True, 
                 lora_alpha=None, log_mod_stats=False, **kwargs):
        self.log_mod_stats = log_mod_stats
        self.rank = rank
        self.mod_v = mod_v
        self.mod_q = mod_q
        self.mod_k = mod_k
        self.mod_ff = mod_ff
        self.lora_alpha = lora_alpha if lora_alpha is not None else self.rank * 2
        self._scaling = self.lora_alpha / self.rank
        if not mod_q: 
            length -= 1
        if not mod_v:
            length -= 1
        if mod_k: 
            length += 1
        if mod_ff: 
            # mlp is 4 * embed_dim
            length += 4
        super().__init__(length=length, top_k=top_k, dropout_rate=dropout_rate,
                         init_prompts=init_prompts, **kwargs) 
        
    @property
    def scaling(self):
        return self._scaling
        
    def _setup_prompt(self):
        self.lora_a = nn.Parameter(torch.zeros((self.pool_size, self.n_layer, self.length, self.embed_dim, self.rank)))
        self.lora_b = nn.Parameter(torch.zeros((self.pool_size, self.n_layer, self.length, self.rank, self.embed_dim)))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
    
    def extract_prompt(self, idx):
        """
        Args:
            idx: torch.Tensor. Indices to lookup.

        """
        # idx: [batch_size x 1]
        # lora_a_batched: [batch_size x n_layer x length x rank x embed_dim]
        # lora_b_batched: [batch_size x n_layer x length x embed_dim x rank]
        lora_a_batched = self.lora_a[idx].squeeze(1)
        lora_b_batched = self.lora_b[idx].squeeze(1)
        matrices = []
        idx_v, idx_k, idx_ff = int(self.mod_q), sum([self.mod_q, self.mod_v]), sum([self.mod_q, self.mod_v, self.mod_k])
        for a, b in zip(lora_a_batched.split(dim=1, split_size=1), lora_b_batched.split(dim=1, split_size=1)):
            a = a.squeeze(1)
            b = b.squeeze(1)
            matrices.append((
                (a[:, 0], b[:, 0]) if self.mod_q else None, # queries
                (a[:, idx_v], b[:, idx_v]) if self.mod_v else None, # values
                (a[:, idx_k], b[:, idx_k]) if self.mod_k else None, # keys
                (a[:, idx_ff:].permute(0, 3, 2, 1).flatten(-2).transpose(2, 1),
                 b[:, idx_ff:].transpose(2, 1).flatten(-2)) if self.mod_ff else None, # ff
                self.scaling
            ))

        return matrices, {}
        
    def add_dropout(self, batched_prompt):
        return batched_prompt
