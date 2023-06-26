import math
import torch
import torch.nn as nn
from .base_prompt import Prompt


class LoRA(Prompt):

    def __init__(self, length=2, dropout_rate=0.0, init_prompts="zeros",
                 rank=4, mod_q=True, mod_v=True, mod_k=False, mod_ff=False, 
                 lora_alpha=None, n_layer=None, n_head=None, **kwargs):
        """
        Args:
            length: Int. Defaults to 1. queries + values will be modulated. 
            init_prompts: Initialization for modulation vectors.
            dropout_rate: Float. Defaults to 0.0.
            log_mod_stats: Bool. Whether stats of modulation vectors should be logged or not. 
            n_layer: Int. Layers for Transformer. Defaults to None.
            n_head: Int. Heads for Transformer. Defaults to None.

        """
        super().__init__(length=length, dropout_rate=dropout_rate, **kwargs)
        self.n_layer = n_layer
        self.n_head = n_head
        assert self.n_layer is not None, "LoRA requires n_layer to be set."
        self.init_prompts = init_prompts
        self.rank = rank
        self.mod_v = mod_v
        self.mod_q = mod_q
        self.mod_k = mod_k
        self.mod_ff = mod_ff
        self.lora_alpha = lora_alpha if lora_alpha is not None else self.rank * 2
        self._scaling = self.lora_alpha / self.rank
        if not mod_q: 
            self.length -= 1
        if not mod_v:
            self.length -= 1
        if mod_k: 
            self.length += 1
        if mod_ff: 
            # mlp is 4 * embed_dim
            self.length += 4   
        
        # modulation matrices going into transformer
        # TODO: much easier to do this using nn.Parameter 
        # refactor (also transpose in mpdtmodel)
        self.lora_a = torch.nn.Embedding(
            self.n_layer if self.n_layer is not None else 1,
            self.length * self.rank * self.embed_dim
        )
        self.lora_b = torch.nn.Embedding(
            self.n_layer if self.n_layer is not None else 1,
            self.length * self.embed_dim * self.rank
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        
        # for lookup
        lookup = torch.arange(self.n_layer).long()
        self.register_buffer('lookup', lookup)
        
    @property
    def scaling(self):
        return self._scaling
        
    def forward(self, x_embed, task_id=None, cls_features=None, attention_mask=None, tok_to_pos=None):
        # batch size = 1 --> same matrices for every sample
        batch_size = 1
        lookup = self.lookup.unsqueeze(0).expand(batch_size, -1)
        lora_a_matrices = self.lora_a(lookup)
        lora_b_matrices = self.lora_b(lookup)
        
        lora_a_batched = lora_a_matrices.reshape(self.n_layer, self.length, self.rank, self.embed_dim)
        lora_b_batched = lora_b_matrices.reshape(self.n_layer, self.length, self.embed_dim, self.rank)
        
        # contains self.length tuples which contain 2 tuples for q and k, 
        # each containing the lora_a and lora_b matrices + the scaling factor.
        # not beautiful but works for now
        matrices = []
        idx_v, idx_k, idx_ff = int(self.mod_q), sum([self.mod_q, self.mod_v]), sum([self.mod_q, self.mod_v, self.mod_k])
        for a, b in zip(lora_a_batched.split(1), lora_b_batched.split(1)):
            a = a.squeeze(0)
            b = b.squeeze(0)
            matrices.append((
                (a[0], b[0]) if self.mod_q else None, # queries
                (a[idx_v], b[idx_v]) if self.mod_v else None, # values
                (a[idx_k], b[idx_k]) if self.mod_k else None, # keys
                (a[idx_ff:].transpose(1, 0).flatten(-2),
                 b[idx_ff:].permute(2, 0, 1).flatten(-2).transpose(1, 0)) if self.mod_ff else None, # ff
                self.scaling
            ))
            
        # matrices: routed to transformer
        return dict(prompt=matrices, infos={})   
