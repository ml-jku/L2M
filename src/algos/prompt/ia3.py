
import torch
import torch.nn as nn
from .base_prompt import Prompt
from ..models.adapter import Adapter


class IA3(Prompt):

    def __init__(self, length=6, dropout_rate=0.0, init_prompts="ones", log_mod_stats=False,
                 mod_embeds=False, mod_q=False, mod_post_ff=False, use_adapter=False,
                 exclude_k=False, exclude_v=False, exclude_ff=False, adapter_first=False,
                 n_layer=None, n_head=None, img_encoder_dims=None, adapter_kwargs=None, **kwargs):
        """
        Args:
            length: Int. Defaults to 6. Keys + values + 4 * ffn
            init_prompts: Initialization for modulation vectors. Defaults to "ones".
            dropout_rate: Float. Defaults to 0.0.
            log_mod_stats: Bool. Whether stats of modulation vectors should be logged or not. 
            n_layer: Int. Layers for Transformer. Defaults to None.
            n_head: Int. Heads for Transformer. Defaults to None.
            mod_embeds: Bool. Whether to modulate embeddings or not. Defaults to False.
            mod_q: Bool. Whether to modulate queries or not. Defaults to False.
            mod_post_ff: Bool. Whether to modulate post-ffn or not. Defaults to False.
            img_encoder_dims: Int. Defaults to None.

        """
        super().__init__(length=length, dropout_rate=dropout_rate, **kwargs)
        self.n_layer = n_layer
        self.n_head = n_head
        self.log_mod_stats = log_mod_stats
        self.init_prompts = init_prompts
        self.mod_embeds = mod_embeds
        self.mod_q = mod_q
        self.mod_post_ff = mod_post_ff
        self.img_encoder_dims = img_encoder_dims
        self.use_adapter = use_adapter
        self.adapter_first = adapter_first
        self.exclude_k = exclude_k
        self.exclude_v = exclude_v
        self.exclude_ff = exclude_ff
        if mod_q or mod_post_ff: 
            self.length += 1
        
        # modulation vectors going into transformer
        self.prompt = torch.nn.Embedding(
            self.n_layer if self.n_layer is not None else 1,
            self.length * self.embed_dim
        )
        self.prompt.apply(self.init_weights)
        prefix_tokens = torch.arange(self.n_layer).long()
        self.register_buffer('prefix_tokens', prefix_tokens)
        
        # modulation vectors going into embeddings of transformer
        if self.mod_embeds: 
            self.prompt_embeds = torch.nn.Embedding(
                len(self.token_to_pos), self.embed_dim
            )
            self.prompt_embeds.apply(self.init_weights)
            embed_tokens = torch.arange(len(self.token_to_pos)).long()
            self.register_buffer('embed_tokens', embed_tokens)
        if self.img_encoder_dims is not None: 
            prompt_img_encoder = [torch.nn.Parameter(torch.ones(1, dim, 1, 1)) for dim in self.img_encoder_dims[:-1]]
            # last is a a linear layer
            prompt_img_encoder.append(torch.nn.Parameter(torch.ones(self.img_encoder_dims[-1])))
            self.prompt_img_encoder = torch.nn.ParameterList(prompt_img_encoder)
        if self.use_adapter: 
            # always 2 adapters per layer
            assert self.n_layer is not None, "n_layer must be set if using adapters"
            adapter_kwargs = adapter_kwargs if adapter_kwargs is not None else {}
            self.adapters = torch.nn.ModuleList([Adapter(input_size=self.embed_dim, **adapter_kwargs),
                                                 Adapter(input_size=self.embed_dim, **adapter_kwargs)])
        
    def init_weights(self, m):
        if self.init_prompts == "ones":
            nn.init.ones_(m.weight)
        elif self.init_prompts == "zeros":
            nn.init.zeros_(m.weight)
        elif self.init_prompts == "ones_normal":
            nn.init.normal_(m.weight, 1, 0.02)
        elif self.init_prompts == "ones_uniform":
            nn.init.uniform_(m.weight, 0.9, 1.1)
        elif self.init_prompts == "normal":
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x_embed, task_id=None, cls_features=None, attention_mask=None, tok_to_pos=None):
        # batch size = 1 --> same modulation vector for every sample
        batch_size = 1
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        batched_prompt_raw = self.prompt(prefix_tokens)
        if self.n_layer is None:
            batched_prompt = batched_prompt_raw.reshape(self.length, self.embed_dim)
            if self.mod_q: 
                vectors = [batched_prompt[0], batched_prompt[1], batched_prompt[2:6].flatten(), batched_prompt[6], None]
            elif self.mod_post_ff:
                vectors = [batched_prompt[0], batched_prompt[1], batched_prompt[2:6].flatten(), None, batched_prompt[6]]
            else: 
                vectors = [batched_prompt[0], batched_prompt[1], batched_prompt[2:].flatten()]
        else:
            batched_prompt = batched_prompt_raw.reshape(self.n_layer, self.length, self.embed_dim)
            if self.mod_q: 
                vectors = [(p.squeeze(0)[0], p.squeeze(0)[1], p.squeeze(0)[2:6].flatten(), p.squeeze(0)[6], None) 
                           for p in batched_prompt.split(1)]
            elif self.mod_post_ff:
                vectors = [(p.squeeze(0)[0], p.squeeze(0)[1], p.squeeze(0)[2:6].flatten(), None, p.squeeze(0)[6]) 
                           for p in batched_prompt.split(1)]
            else:
                vectors = [(p.squeeze(0)[0], p.squeeze(0)[1], p.squeeze(0)[2:].flatten()) for p in batched_prompt.split(1)]
        if self.dropout_rate > 0:
            if self.n_layer is None:
                vectors = [self.dropout(p) for p in vectors]
            else:
                vectors = [tuple(self.dropout(v) for v in p) for p in vectors]
        if self.use_adapter: 
            if self.adapter_first: 
                vectors[0] = tuple(list(vectors[0]) + [self.adapters[0], self.adapters[1]])
            else: 
                vectors[-1] = tuple(list(vectors[-1]) + [self.adapters[0], self.adapters[1]])
                
        embed_vectors, img_encoder_vectors = None, None
        if self.mod_embeds:
            embed_tokens = self.embed_tokens.unsqueeze(0).expand(batch_size, -1)
            embed_vectors = self.prompt_embeds(embed_tokens).reshape(-1, self.embed_dim).split(1)
        if self.img_encoder_dims is not None: 
            img_encoder_vectors = self.prompt_img_encoder
            
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
                
        # vectors: routed to transformer
        # embed_vectors: routed to embedding layer of transformer
        # img_encoder_vectors: routed to image encoder
        return dict(prompt=vectors, infos=stats, embed_vectors=embed_vectors, img_encoder_vectors=img_encoder_vectors)    
