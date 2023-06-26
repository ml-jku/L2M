import torch
import hflayers
from .base_prompt import Prompt


class CustomHopfieldLayer(hflayers.HopfieldLayer):

    def get_association_matrix(self, input, stored_pattern_padding_mask=None, association_mask=None):
        """
        Overwrite method to ensure grad flow.

        Fetch Hopfield association matrix used for lookup gathered by passing through the specified data.

        :param input: data to be passed through the Hopfield association
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: association matrix as computed by the Hopfield core module
        """
        return self.hopfield._associate(
            data=self._prepare_input(input=input),
            return_raw_associations=True, return_projected_patterns=True,
            stored_pattern_padding_mask=stored_pattern_padding_mask,
            association_mask=association_mask)


class HopfieldPrompt(Prompt):

    def __init__(self, hidden_size=64, pattern_size=128, num_heads=1,
                 lookup_weights_as_separated=True, scaling=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pattern_size = pattern_size
        self.scaling = scaling
        self.lookup_weights_as_separated = lookup_weights_as_separated
        self.hopfield = CustomHopfieldLayer(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            output_size=self.embed_dim,
            num_heads=self.num_heads,
            pattern_size=self.pattern_size,
            scaling=self.scaling,
            quantity=self.pool_size,
            lookup_weights_as_separated=self.lookup_weights_as_separated,
            # stored_pattern_as_static=True,
            # state_pattern_as_static=True,
        )

    def forward(self, x_embed, task_id=None, cls_features=None, attention_mask=None, tok_to_pos=None):
        batch_size = x_embed.shape[0]
        # x_embed: [batch_size x seq_len x embed_dim]
        x_embed_mean = self.aggregate_embeds(x_embed, cls_features=cls_features, 
                                             attention_mask=attention_mask, tok_to_pos=tok_to_pos)
        # x_embed_mean: [batch_size x embed_dim]
        x_embed_mean = x_embed_mean.unsqueeze(1)
        # out: [ attn_output, None, xi, v ]
        # xi: raw associations --> attention scores --> take top-k of these
        # v: projected patterns --> "prompt matrix" index select top-k via xi
        out = self.hopfield.get_association_matrix(x_embed_mean)
        association_scores = out[2].reshape(batch_size, self.pool_size)
        prompt_matrix = out[3].reshape(batch_size, self.pool_size, self.pattern_size)
        # select top-k prompts
        _, idx = torch.topk(association_scores, k=self.top_k, dim=1)
        selected_prompts = prompt_matrix[torch.arange(prompt_matrix.size(0)).unsqueeze(1), idx]
        # reshape if necessary
        selected_prompts, _ = self.extract_prompt(selected_prompts)
        if self.training:
            self.add_counts(idx)
        return dict(prompt=selected_prompts, infos={"prompt_idx": idx})

    def extract_prompt(self, batched_prompt_raw):
        return batched_prompt_raw, {}
