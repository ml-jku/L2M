import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.utils import logging
from transformers.models.decision_transformer import DecisionTransformerGPT2Model
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerGPT2Block
from .online_decision_transformer_model import OnlineDecisionTransformerModel
from .discrete_decision_transformer_model import DiscreteDTModel
from .multi_domain_discrete_dt_model import MultiDomainDiscreteDTModel
from .adapter import Adapter


logger = logging.get_logger(__name__)


class DecisionTransformerGPT2BlockWithAdapter(DecisionTransformerGPT2Block):
    """
    Wrapper class for DecisionTransformerGPT2Block.
    Allows for using Adapters modules as introduced in: https://arxiv.org/abs/1902.00751

    """
    def __init__(self, config, layer_idx=None, adapter_kwargs=None):
        super().__init__(config, layer_idx)
        if adapter_kwargs is None:
            adapter_kwargs = {}
        self.adapter1 = Adapter(input_size=config.hidden_size, **adapter_kwargs)
        self.adapter2 = Adapter(input_size=config.hidden_size, **adapter_kwargs)

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

        # adapter layer
        attn_output, _, _ = self.adapter1(attn_output, attn_output)

        # residual connection
        hidden_states = attn_output + residual

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

        # adapter layer
        feed_forward_hidden_states, _, _ = self.adapter2(feed_forward_hidden_states, feed_forward_hidden_states)

        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class DecisionTransformerGPT2ModelWithAdapter(DecisionTransformerGPT2Model):

    """
    Wrapper class for DecisionTransformerGPT2Model.
    Allows for using Adapters modules as introduced in: https://arxiv.org/abs/1902.00751

    """
    def __init__(self, config, adapter_kwargs=None):
        super().__init__(config)
        del self.h
        self.h = nn.ModuleList(
            [DecisionTransformerGPT2BlockWithAdapter(config, layer_idx=i, adapter_kwargs=adapter_kwargs)
             for i in range(config.num_hidden_layers)]
        )


class DTWithAdapter(OnlineDecisionTransformerModel):

    def __init__(self, config, observation_space, action_space, adapter_kwargs=None, **kwargs):
        super().__init__(config, observation_space, action_space, **kwargs)
        del self.encoder
        self.encoder = DecisionTransformerGPT2ModelWithAdapter(config, adapter_kwargs=adapter_kwargs)
        self.post_init()
        
class DiscreteDTWithAdapter(DiscreteDTModel):

    def __init__(self, config, observation_space, action_space, adapter_kwargs=None, **kwargs):
        super().__init__(config, observation_space, action_space, **kwargs)
        del self.encoder
        self.encoder = DecisionTransformerGPT2ModelWithAdapter(config, adapter_kwargs=adapter_kwargs)
        self.post_init()
        

class MultiDomainDiscreteDTWithAdapter(MultiDomainDiscreteDTModel):

    def __init__(self, config, observation_space, action_space, adapter_kwargs=None, **kwargs):
        super().__init__(config, observation_space, action_space, **kwargs)
        del self.encoder
        self.encoder = DecisionTransformerGPT2ModelWithAdapter(config, adapter_kwargs=adapter_kwargs)
        self.post_init()
