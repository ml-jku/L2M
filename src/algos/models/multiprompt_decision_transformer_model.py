import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging
from transformers.models.decision_transformer import DecisionTransformerGPT2Model
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerGPT2Block,\
    DecisionTransformerGPT2Attention, DecisionTransformerGPT2MLP
from .online_decision_transformer_model import OnlineDecisionTransformerModel
from .multi_domain_discrete_dt_model import MultiDomainDiscreteDTModel
from .discrete_decision_transformer_model import DiscreteDTModel
from .adapter import Adapter


logger = logging.get_logger(__name__)


class AttentionIA3(DecisionTransformerGPT2Attention):
    """
    Wrapper class for DecisionTransformerGPT2Attention.
    Add functionality to pass IA3 vectors and scale keys and values by them.
    Unfortunately, requires to overwrite the entire forward().

    """

    def __init__(self, config, is_cross_attention=False, layer_idx=None, ia3=False, lora=False, lora_dropout=0):
        super().__init__(config=config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.ia3 = ia3
        self.lora = lora
        self.lora_dropout = lora_dropout
        if lora_dropout > 0:
            self.lora_dropout_layer = nn.Dropout(lora_dropout)

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
        modulators=None
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to"
                    " instantiate class with `DecisionTransformerGPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        if self.ia3 and modulators is not None:
            if modulators[0] is not None:
                value = value * modulators[0]
            if modulators[1] is not None:
                key = key * modulators[1]
            if len(modulators) == 5 and modulators[3] is not None and not isinstance(modulators[3], Adapter):
                query = query * modulators[3]
        if self.lora and modulators is not None: 
            scaling = modulators[-1]
            if self.lora_dropout > 0:
                hidden_states = self.lora_dropout_layer(hidden_states)
            if modulators[0] is not None:
                lora_a_q, lora_b_q = modulators[0]
                lora_a_q = lora_a_q.transpose(0, 1) if len(lora_a_q.shape) == 2 else lora_a_q
                lora_b_q = lora_b_q.transpose(0, 1) if len(lora_b_q.shape) == 2 else lora_b_q
                lora_out_q = (hidden_states @ lora_a_q @ lora_b_q) * scaling
                query = query + lora_out_q
            if modulators[1] is not None:
                lora_a_v, lora_b_v = modulators[1]
                lora_a_v = lora_a_v.transpose(0, 1) if len(lora_a_v.shape) == 2 else lora_a_v
                lora_b_v = lora_b_v.transpose(0, 1) if len(lora_b_v.shape) == 2 else lora_b_v
                lora_out_v = (hidden_states @ lora_a_v @ lora_b_v) * scaling
                value = value + lora_out_v
            if modulators[2] is not None: 
                lora_a_k, lora_b_k = modulators[2]
                lora_a_k = lora_a_k.transpose(0, 1) if len(lora_a_k.shape) == 2 else lora_a_k
                lora_b_k = lora_b_k.transpose(0, 1) if len(lora_b_k.shape) == 2 else lora_b_k
                lora_out_k = (hidden_states @ lora_a_k @ lora_b_k) * scaling
                key = key + lora_out_k  
                
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class MLPIA3(DecisionTransformerGPT2MLP):
    """
    Wrapper class for DecisionTransformerGPT2MLP.
    Add functionality to pass IA3 vectors and scale the ff output.
    Unfortunately, requires to overwrite the entire forward().

    """

    def __init__(self, intermediate_size, config, ia3=False, lora=False, ia3_lff_pre=False, lora_dropout=0):
        super().__init__(intermediate_size=intermediate_size, config=config)
        self.ia3 = ia3
        self.lora = lora
        self.ia3_lff_pre = ia3_lff_pre
        self.lora_dropout = lora_dropout
        if lora_dropout > 0:
            self.lora_dropout_layer = nn.Dropout(lora_dropout)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], modulators=None) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        # pre-activation IA3
        if self.ia3_lff_pre and self.ia3 and modulators is not None:
            if modulators[2] is not None:
                hidden_states = hidden_states * modulators[2]
        if self.lora and modulators is not None:
            hidden_states_lora = hidden_states
            if self.lora_dropout > 0:
                hidden_states_lora = self.lora_dropout_layer(hidden_states_lora)
            if modulators[3] is not None: 
                scaling = modulators[-1]
                lora_a, lora_b = modulators[3]
                lora_a = lora_a.transpose(0, 1) if len(lora_a.shape) == 2 else lora_a
                lora_b = lora_b.transpose(0, 1) if len(lora_b.shape) == 2 else lora_b
                lora_out = (hidden_states_lora @ lora_a @ lora_b) * scaling
                hidden_states = hidden_states + lora_out        
        
        hidden_states = self.act(hidden_states)
        # post-activation IA3
        if not self.ia3_lff_pre and self.ia3 and modulators is not None:
            if modulators[2] is not None:
                hidden_states = hidden_states * modulators[2]
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.ia3 and modulators is not None and len(modulators) == 5 and modulators[4] is not None \
            and not isinstance(modulators[4], Adapter):
            hidden_states = hidden_states * modulators[4]
        return hidden_states


class DecisionTransformerGPT2BlockWithIA3(DecisionTransformerGPT2Block):
    """
    Wrapper Class DecisionTransformerGPT2Block.
    Add functionality to pass IA3 vectors to the Attention and MLP layers.

    """
    def __init__(self, config, layer_idx=None, ia3=False, lora=False, ia3_lff_pre=False, lora_dropout=0):
        super().__init__(config=config, layer_idx=layer_idx)
        self.ia3 = ia3
        self.lora = lora
        self.ia3_lff_pre = ia3_lff_pre
        del self.attn, self.mlp
        self.attn = AttentionIA3(config=config, layer_idx=layer_idx, ia3=ia3, lora=lora, lora_dropout=lora_dropout)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.mlp = MLPIA3(inner_dim, config, ia3=ia3, ia3_lff_pre=self.ia3_lff_pre, 
                          lora=lora, lora_dropout=lora_dropout)

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
        modulators=None
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
            modulators=modulators if self.ia3 or self.lora else None
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        
        # first adapter block
        if modulators and isinstance(modulators[-2], Adapter):
            attn_output, _, _ = modulators[-2](attn_output, attn_output)

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
        feed_forward_hidden_states = self.mlp(hidden_states, modulators=modulators if self.ia3 or self.lora else None)
        
        # second adapter block
        if modulators and isinstance(modulators[-1], Adapter):
            feed_forward_hidden_states, _, _ = modulators[-1](feed_forward_hidden_states, attn_output)
        
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class MultiPromptDTGPT2Model(DecisionTransformerGPT2Model):
    """
    Wrapper class for DecisionTransformerGPT2Model.
    Allows for prepending prompts to every layer block input.
    Unfortunately, the entire forward() need to be overwritten to incorportate this functionality.

    """
    def __init__(self, config, ia3=False, lora=False, ia3_lff_pre=False, lora_dropout=0):
        super().__init__(config)
        self.ia3 = ia3
        self.lora = lora
        self.ia3_lff_pre = ia3_lff_pre
        del self.h
        self.h = nn.ModuleList(
            [DecisionTransformerGPT2BlockWithIA3(config, layer_idx=i, ia3=ia3, lora=lora,
                                                 ia3_lff_pre=ia3_lff_pre, lora_dropout=lora_dropout)
             for i in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prompt: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # extract layer prompts
        prompt_per_layer, modulators_per_layer = None, None
        if prompt is not None:
            if self.ia3 or self.lora:
                assert len(prompt) == self.config.n_layer
                modulators_per_layer = prompt
                prompt = None
            else:
                assert prompt.shape[1] == self.config.n_layer, "Prompt shape must be (batch_size, n_layer, hidden_size)"
                prompt_per_layer = [prompt[:, i] for i in range(self.config.n_layer)]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        original_seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
            if len(past_key_values) < len(self.h):
                past_key_values = tuple([*past_key_values, *[None] * (len(self.h) - len(past_key_values))])
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")

            # extend attention mask for prompts
            if prompt is not None:
                attention_mask = torch.cat(
                    (torch.ones(prompt_per_layer[0].shape[:2], device=device), attention_mask), dim=1
                )

            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        pruned_mask = False
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # prepend prompt
            if prompt is not None:
                hidden_states = self.prune_hidden_states(hidden_states, original_seq_len, i)
                hidden_states = torch.cat((prompt_per_layer[i], hidden_states), dim=1)
            modulators = None
            if (self.ia3 or self.lora) and modulators_per_layer is not None:
                modulators = modulators_per_layer[i]

            # prune attention mask if prefix tuning only in first layer
            if layer_past is None and past_length > 0 and not pruned_mask:
                attention_mask = attention_mask[:, :, :, past_length:]
                pruned_mask = True

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    modulators=modulators
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.prune_hidden_states(hidden_states, original_seq_len, i)
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    @staticmethod
    def prune_hidden_states(hidden_states, seq_len, layer_idx):
        if layer_idx > 0:
            hidden_states = hidden_states[:, -seq_len:]
        return hidden_states


class MultiPromptDTModel(OnlineDecisionTransformerModel):
    """
    Multi-prompt version of the UDT model. Adds functionality to prepend prompts to every layer in the transformer.
    Unfortunately, entire compute_hidden_states() method needs to be overwritten to support this.

    """

    def __init__(self, config, observation_space, action_space, ia3_lff_pre=False, lora_droput=0, **kwargs):
        super().__init__(config, observation_space, action_space, **kwargs)
        self.ia3 = self.prompt_kwargs.get('kind', False) in ["ia3", "l2m_ia3"]
        self.lora = self.prompt_kwargs.get("kind", False) in ["lora", "l2m_lora"]
        self.ia3_lff_pre = ia3_lff_pre
        del self.encoder
        self.encoder = MultiPromptDTGPT2Model(config, ia3=self.ia3, lora=self.lora, 
                                              ia3_lff_pre=self.ia3_lff_pre, lora_dropout=lora_droput)
        # self.post_init()

    def compute_hidden_states(
            self,
            states=None,
            actions=None,
            rewards=None,
            returns_to_go=None,
            timesteps=None,
            attention_mask=None,
            output_hidden_states=None,
            output_attentions=None,
            return_dict=None,
            prompt=None,
            task_id=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        batch_size, seq_length = actions.shape[0], actions.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings = self.embed_inputs(
            states, actions, returns_to_go, rewards, attention_mask
        )

        if self.use_time_embds:
            time_embeddings = self.get_time_embeddings(timesteps, attention_mask=attention_mask)
            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings = self.add_pos_embeddings(
                time_embeddings, state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings
            )
        else:
            time_embeddings = None

        # prepare inputs + masks
        inputs, masks = self.construct_inputs_and_masks(
            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings,
            attention_mask, time_embeddings=time_embeddings
        )
        stacked_inputs, stacked_attention_mask = self.prepare_inputs_and_masks(inputs, masks, batch_size, seq_length)

        prompt_infos, prompt_hidden_states, prompt_attention_mask, prompt_stacked_inputs = None, None, None, None
        if self.learnable_prompt:
            learnable_prompt_inputs = self.compute_learnable_prompt_inputs(
                stacked_inputs, stacked_attention_mask, output_attentions,
                output_hidden_states, return_dict, task_id
            )
            if learnable_prompt_inputs is not None:
                prompt_stacked_inputs = learnable_prompt_inputs["prompt_stacked_inputs"]
                prompt_stacked_attention_mask = learnable_prompt_inputs["prompt_stacked_attention_mask"]
                prompt_infos = learnable_prompt_inputs["prompt_infos"]
                # can be None in case of training keys only
                if prompt_stacked_inputs is not None and prompt_stacked_attention_mask is not None:
                    if learnable_prompt_inputs.get("img_encoder_vectors", None) is not None and len(states.shape) > 4:     
                        # reproduce state_embeddings + stacked_inputs            
                        state_embeddings = self.modulate_image_encoder(
                            states, mod_vectors=learnable_prompt_inputs["img_encoder_vectors"]
                        )
                        state_embeddings, _, _, _ = self.add_pos_embeddings(time_embeddings, state_embeddings, None, None, None)
                        
                        # prepare inputs + masks
                        inputs, masks = self.construct_inputs_and_masks(
                            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings,
                            attention_mask, time_embeddings=time_embeddings
                        )
                        stacked_inputs, stacked_attention_mask = self.prepare_inputs_and_masks(inputs, masks, 
                                                                                            batch_size, seq_length)
                    if learnable_prompt_inputs.get("embed_vectors", None) is not None:
                        # reproduce stacked_inputs
                        stacked_inputs = self.modulate_embeddings(
                            learnable_prompt_inputs["embed_vectors"], state_embeddings, action_embeddings, returns_embeddings,
                            rewards_embeddings, time_embeddings, attention_mask, batch_size, seq_length
                        )
                                    
                    if self.config.add_cross_attention:
                        stacked_inputs = prompt_stacked_inputs
                        stacked_attention_mask = prompt_stacked_attention_mask
                    if self.prompt.prefix:
                        stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)

        # make position ids
        if self.global_pos_embds:
            position_ids = torch.arange(stacked_inputs.shape[1], device=stacked_inputs.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0)
        else: 
            position_ids = torch.zeros(stacked_inputs.shape[:2], device=stacked_inputs.device, dtype=torch.long)
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # exploits default behaviour of DecisionTransformerGPT2Block to add cross attention on (latent) prompts
            encoder_hidden_states=prompt_hidden_states,
            encoder_attention_mask=prompt_attention_mask,
            prompt=prompt_stacked_inputs if not self.prompt.prefix else None,
            past_key_values=prompt_stacked_inputs if self.prompt.prefix else None
        )
        # grab last hidden state
        x = encoder_outputs['last_hidden_state']

        if (self.learnable_prompt or prompt is not None) and not self.config.add_cross_attention:
            x = x[:, -seq_length * len(inputs):]
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, len(inputs), self.hidden_size).permute(0, 2, 1, 3)
        # [batch_size, r_s_a, seq_len, hidden_size]
        return x, encoder_outputs, prompt_infos
    
    def modulate_embeddings(self, embed_vectors, state_embeddings, action_embeddings, returns_embeddings,
                            rewards_embeddings, time_embeddings, attention_mask, batch_size, seq_length):
        # modulate embeddings with learned modulation vectors
        assert embed_vectors is not None
        state_embeddings = state_embeddings * embed_vectors[0]
        action_embeddings = action_embeddings * embed_vectors[1]
        if self.rtg_condition: 
            returns_embeddings = returns_embeddings * embed_vectors[2]
        if self.reward_condition: 
            idx = 3 if self.rtg_condition else 2
            rewards_embeddings = rewards_embeddings * embed_vectors[idx]
        
        # reconstruct stacked_inputs
        inputs, masks = self.construct_inputs_and_masks(
            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings,
            attention_mask, time_embeddings=time_embeddings
        )
        stacked_inputs, _ = self.prepare_inputs_and_masks(inputs, masks, batch_size, seq_length)
        return stacked_inputs
    
    def modulate_image_encoder(self, states, mod_vectors):
        # is_image_space
        states = states.float() / 255.0    
        return self.get_state_embeddings(states, mod_vectors=mod_vectors)


class DiscreteMPDTModel(DiscreteDTModel, MultiPromptDTModel):

    def __init__(self, config, observation_space, action_space, ia3_lff_pre=False, lora_dropout=0, **kwargs):
        super().__init__(config, observation_space, action_space, **kwargs)
        # repeat init of MultiPromptDTModel
        # reasons is that post_init() has been called, and would destroy initialization of prompts/modulation vectors
        self.ia3 = self.prompt_kwargs.get('kind', False) in ["ia3", "l2m_ia3"]
        self.lora = self.prompt_kwargs.get("kind", False) in ["lora", "l2m_lora"]
        self.ia3_lff_pre = ia3_lff_pre
        del self.encoder
        self.encoder = MultiPromptDTGPT2Model(config, ia3=self.ia3, lora=self.lora, 
                                              ia3_lff_pre=self.ia3_lff_pre, lora_dropout=lora_dropout)


class MDMPDTModel(MultiDomainDiscreteDTModel, MultiPromptDTModel):

    def __init__(self, config, observation_space, action_space, ia3_lff_pre=False, lora_dropout=0, **kwargs):
        super().__init__(config, observation_space, action_space, **kwargs)
        # repeat init of MultiPromptDTModel
        # reasons is that post_init() has been called, and would destroy initialization of prompts/modulation vectors
        self.ia3 = self.prompt_kwargs.get('kind', False) in ["ia3", "l2m_ia3"]
        self.lora = self.prompt_kwargs.get("kind", False) in ["lora", "l2m_lora"]
        self.ia3_lff_pre = ia3_lff_pre
        del self.encoder
        self.encoder = MultiPromptDTGPT2Model(config, ia3=self.ia3, lora=self.lora,
                                              ia3_lff_pre=self.ia3_lff_pre, lora_dropout=lora_dropout)
