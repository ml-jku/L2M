import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerGPT2Block,\
    DecisionTransformerGPT2Model


class GatingMechanism(torch.nn.Module):
    """From: https://github.com/dhruvramani/Transformers-RL"""

    def __init__(self, d_input, bg=0.1):
        super(GatingMechanism, self).__init__()
        self.Wr = torch.nn.Linear(d_input, d_input)
        self.Ur = torch.nn.Linear(d_input, d_input)
        self.Wz = torch.nn.Linear(d_input, d_input)
        self.Uz = torch.nn.Linear(d_input, d_input)
        self.Wg = torch.nn.Linear(d_input, d_input)
        self.Ug = torch.nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g


class GatedDecisionTransformerGPT2Block(DecisionTransformerGPT2Block):

    def __init__(self, config, **kwargs):
        """
        Adds gating mechanism as proposed by GtTrXL: https://arxiv.org/pdf/1910.06764.pdf
        Gating mechanism is a GRU.
        Args:
            config: Huggingface config.
            **kwargs:
        """
        super().__init__(config, **kwargs)
        hidden_size = config.hidden_size
        # self.gru1 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gate1 = GatingMechanism(hidden_size)
        self.gate2 = GatingMechanism(hidden_size)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # add Gating mechanism (in form of GRU) instead of regular residual
        # hidden_states = attn_output + residual
        hidden_states = self.gate1(residual, attn_output)

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        # add Gating mechanism (in form of GRU) instead of regular residual
        # hidden_states = residual + feed_forward_hidden_states
        hidden_states = self.gate2(residual, feed_forward_hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GatedDecisionTransformerGPT2Model(DecisionTransformerGPT2Model):

    def __init__(self, config):
        """
        Replaces the regular DecisionTransformerGPT2Block by a GatedDecisionTransformerGPT2Block.

        Args:
            config: Huggingface config.
        """
        super().__init__(config)
        del self.h
        self.h = nn.ModuleList(
            [GatedDecisionTransformerGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()
