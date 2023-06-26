import torch
import torch.nn as nn


class Prompt(nn.Module):

    def __init__(self, length=5, embed_dim=128, top_k=5, pool_size=10, dropout_rate=0.1, embed_key='last',
                 agg_token="all", input_type="s_rtg_a_r", prefix=False, pretrain=False):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.embed_key = embed_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.prefix = prefix
        self.agg_token = agg_token
        self.pretrain = pretrain
        self.token_to_pos = {token: i for i, token in enumerate(input_type.split("_"))}
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.counts_total = 0
        self.task_id = 0
        self.register_buffer('counts', torch.zeros(self.pool_size, requires_grad=False))
        self.register_buffer('inv_counts_so_far', torch.ones(self.pool_size, requires_grad=False))

    def forward(self, x_embed, task_id=None, cls_features=None, attention_mask=None, tok_to_pos=None):
        raise NotImplementedError
    
    @staticmethod
    def l2_normalize(x, dim=None, epsilon=1e-12):
        return torch.nn.functional.normalize(x, p=2.0, dim=dim, eps=epsilon)

    @staticmethod
    def compute_cosine_sim_matrix(a, b, eps=1e-8):
        a_n, b_n = a.norm(dim=1).unsqueeze(1), b.norm(dim=1).unsqueeze(1)
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt, a_norm, b_norm

    def aggregate_embeds(self, x_embed, cls_features=None, attention_mask=None, tok_to_pos=None):
        if self.agg_token != "all":
            if tok_to_pos is not None:
                # handle case if we have multiple tokens per token type
                batch_size, seq_len, embed_dim = x_embed.shape
                num_tokens = list(tok_to_pos.values())[-1] + 1
                x_embed = x_embed.reshape(batch_size, seq_len // num_tokens, num_tokens, embed_dim)
                if self.agg_token == "concat": 
                    x_embed = x_embed.reshape(batch_size, seq_len // num_tokens, -1)
                    attention_mask = attention_mask.reshape(batch_size, seq_len // num_tokens, num_tokens)[..., -1].flatten(1)
                else: 
                    token_pos = tok_to_pos[self.agg_token]
                    x_embed = x_embed[:, :, token_pos].reshape(batch_size, -1, embed_dim)
                    attention_mask = attention_mask.reshape(batch_size, seq_len // num_tokens, num_tokens)[..., token_pos].flatten(1)
            else: 
                token_pos, num_tokens = self.token_to_pos[self.agg_token], len(self.token_to_pos)
                assert x_embed.shape[1] % num_tokens == 0 and attention_mask.shape[1] % num_tokens == 0
                x_embed = x_embed[:, token_pos::num_tokens]
                attention_mask = attention_mask[:, token_pos::num_tokens]
        if self.embed_key in ["mean", "last", "first", "embed", "second", "third", "concat"]:
            if attention_mask is not None:
                # masked mean
                x_embed_mean = torch.sum(x_embed * attention_mask.float().unsqueeze(-1), dim=1) \
                               / torch.sum(attention_mask.float(), -1, keepdim=True)
            else:
                x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embed_key == "mean_no_mask":
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embed_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embed_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embed_key == 'cls':
            if cls_features is None:
                x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
            else:
                x_embed_mean = cls_features
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")
        return x_embed_mean

    def reset_counts(self, device):
        self.counts = torch.zeros(self.pool_size, requires_grad=False, device=device)
        self.counts_total = 0

    def add_counts(self, idx):
        with torch.autocast(device_type='cuda', enabled=False):
            idx_counts = torch.bincount(idx.reshape(-1), minlength=self.pool_size)
            idx_counts = idx_counts / idx_counts.sum()
            self.counts_total += 1
            self.counts = self.counts + (idx_counts - self.counts) / self.counts_total
            
    def set_task_id(self, task_id):
        self.task_id = task_id
        self.update_inv_counts()

    def update_inv_counts(self):
        with torch.autocast(device_type='cuda', enabled=False):
            inv_counts = 1.0 / (self.counts.clone() + 1e-6)
            self.inv_counts_so_far = inv_counts / inv_counts.sum()
