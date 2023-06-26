import torch
import torch.nn as nn
from .base_prompt import Prompt


class DummyPrompt(Prompt):

    def __init__(self, init_range=0.02, init_prompts="uniform", reparametrize=False, reparam_v1=False,
                 prefix_first_only=False, reduction_factor=2, n_layer=None, n_head=None, **kwargs):
        super().__init__(**kwargs)
        self.init_range = init_range
        self.init_prompts = init_prompts
        self.n_layer = n_layer
        self.n_head = n_head
        self.reparametrize = reparametrize
        self.reduction_factor = reduction_factor
        self.prefix_first_only = prefix_first_only
        if self.n_layer is None:
            hidden_dim = self.embed_dim
        else:
            if self.prefix:
                assert self.n_head is not None, "n_head must be specified if using Prefix-Prompt"
                # n_layer * 2 as will produce separate prompts for keys and values
                self.n_layer *= 2
                if self.prefix_first_only:
                    self.n_layer = 2
                hidden_dim = self.n_layer * self.embed_dim
            else:
                hidden_dim = self.n_layer * self.embed_dim
        self.prompt = torch.nn.Embedding(self.length, hidden_dim)
        if self.init_prompts == "uniform":
            nn.init.uniform_(self.prompt.weight, -1, 1)
        elif self.init_prompts == "normal":
            nn.init.normal_(self.prompt.weight, 0, self.init_range)
        elif self.init_prompts == "normal_predefined":
            nn.init.normal_(self.prompt.weight, -0.0058132107, 1.2405446)

        prefix_tokens = torch.arange(self.length).long()
        self.register_buffer('prefix_tokens', prefix_tokens)
        if self.reparametrize:
            self.control_trans = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // self.reduction_factor),
                nn.Tanh(),
                nn.Linear(hidden_dim // self.reduction_factor, hidden_dim),
            )

    def forward(self, x_embed, task_id=None, cls_features=None, attention_mask=None, tok_to_pos=None):
        batch_size = x_embed.shape[0]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        batched_prompt_raw = self.prompt(prefix_tokens)
        if self.reparametrize:
            batched_prompt_raw = self.control_trans(batched_prompt_raw)
        if self.n_layer is None:
            batched_prompt = batched_prompt_raw.reshape(batch_size, self.length, self.embed_dim)
        else:
            # [B, length, n_layer * C]
            batch_size, length, _ = batched_prompt_raw.shape
            if self.prefix:
                # --> [n_layer, B, n_head, top_k * length, C]
                batched_prompt = batched_prompt_raw.reshape(batch_size, length, self.n_layer, self.embed_dim)
                batched_prompt = batched_prompt.reshape(batch_size, length, self.n_layer, self.n_head, self.embed_dim // self.n_head)
                # permute + split up n_layers --> keys, values
                batched_prompt = batched_prompt.permute(2, 0, 3, 1, 4).split(2)
            else:
                # --> [B, n_layer, top_k * length, C]
                batched_prompt = batched_prompt_raw.reshape(batch_size, self.n_layer, length, self.embed_dim)

        if self.dropout_rate > 0:
            if self.n_layer and self.prefix:
                batched_prompt = [self.dropout(p) for p in batched_prompt]
            else:
                batched_prompt = self.dropout(batched_prompt)

        return dict(prompt=batched_prompt, infos={})
