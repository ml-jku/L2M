import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from gym import spaces
from transformers import DecisionTransformerModel
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput, Conv1D
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, DiagGaussianDistribution
from stable_baselines3.common import preprocessing
from .gated_decision_transformer_model import GatedDecisionTransformerGPT2Model
from .image_encoders import make_image_encoder
from ..prompt import make_prompt


LOG_STD_MAX = 2
LOG_STD_MIN = -20


@dataclass
class OnlineDecisionTransformerOutput(DecisionTransformerOutput):
    action_log_probs: torch.FloatTensor = None
    reward_preds: torch.FloatTensor = None
    action_logits: torch.FloatTensor = None
    entropy: torch.FloatTensor = None
    last_encoder_output: torch.FloatTensor = None
    prompt_infos: dict = None
    cross_attentions: torch.FloatTensor = None


class OnlineDecisionTransformerModel(DecisionTransformerModel):

    def __init__(self, config, observation_space, action_space, num_task_heads=1, n_layer_head=1, use_time_embds=True,
                 rtg_condition=True, reward_condition=False, reward_condition_only=False,
                 stochastic_policy=True, gating=False, relative_pos_embds=False, global_pos_embds=False, 
                 both_time_embds=False, separate_prompt_embed=False, learnable_prompt=False, use_crossattn_mask=True, 
                 detach_prompt_embed=False, share_prompt_pos_embed=False, symlog_transform=False, separate_ln=False,
                 prompt_kwargs=None, encoder_kwargs=None):
        super().__init__(config)
        self.num_task_heads = num_task_heads
        self.n_layer_head = n_layer_head
        self.stochastic_policy = stochastic_policy
        self.gating = gating
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_condition = reward_condition
        self.reward_condition_only = reward_condition_only
        self.use_time_embds = use_time_embds
        self.relative_pos_embds = relative_pos_embds
        self.global_pos_embds = global_pos_embds
        self.both_time_embds = both_time_embds
        self.rtg_condition = rtg_condition
        self.separate_prompt_embed = separate_prompt_embed
        self.learnable_prompt = learnable_prompt
        self.prompt_kwargs = prompt_kwargs
        self.use_crossattn_mask = use_crossattn_mask
        self.detach_prompt_embed = detach_prompt_embed
        self.share_prompt_pos_embed = share_prompt_pos_embed
        self.symlog_transform = symlog_transform
        self.separate_ln = separate_ln
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs is not None else {}
        # in original implementation predict_action is a linear layer. We set it None and make a function out of it
        del self.predict_action
        self.is_image_space = preprocessing.is_image_space(self.observation_space, check_channels=False)
        self.is_discrete = isinstance(self.action_space, spaces.Discrete)

        if not self.rtg_condition:
            del self.embed_return
            del self.predict_return
        if self.gating:
            del self.encoder
            self.encoder = GatedDecisionTransformerGPT2Model(config)
        if self.reward_condition or self.reward_condition_only:
            self.embed_rewards = torch.nn.Linear(1, config.hidden_size)
            self.predict_reward = torch.nn.Linear(config.hidden_size, 1)
        if self.both_time_embds: 
            self.embed_timestep_rel = nn.Embedding(config.max_length, config.hidden_size)
        elif self.relative_pos_embds:
            del self.embed_timestep
            self.embed_timestep = nn.Embedding(config.max_length, config.hidden_size)
        if self.is_image_space:
            self.embed_image = make_image_encoder(observation_space=observation_space, 
                                                  features_dim=config.hidden_size, encoder_kwargs=self.encoder_kwargs)
        if self.is_discrete:
            self.embed_action = nn.Sequential(nn.Embedding(self.action_space.n, config.hidden_size), nn.Tanh())
        if self.separate_ln: 
            self.embed_ln = nn.ModuleDict({"s": nn.LayerNorm(config.hidden_size),
                                           "rtg": nn.LayerNorm(config.hidden_size),
                                           "a": nn.LayerNorm(config.hidden_size),
                                           "r": nn.LayerNorm(config.hidden_size)})
        self.setup_policy()
        self.post_init()
        self.setup_prompt()
        self.tok_to_pos = {"s": 0, "rtg": 1,  "a": 2}
        self.tok_to_pred_pos = {"s": 0, "rtg": 2,  "a": 1}

    def get_optim_groups(self, weight_decay):
        """
        From: https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136

        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, Conv1D, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Parameter)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)

        # add instances of nn.Parameter
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if (isinstance(p, torch.nn.Parameter) or isinstance(m, torch.nn.Parameter)) \
                        and (fpn not in decay and fpn not in no_decay):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optim_groups_names = [
            {"params": sorted(list(decay)), "weight_decay": weight_decay},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]
        print("Optim groups:\n", optim_groups_names)
        return optim_groups

    def get_action_head_params(self):
        if self.stochastic_policy:
            return list(self.mu.parameters()) + list(self.log_std.parameters())
        return list(self.action_pred.parameters())

    def zero_grads_for_action_heads(self, task_id):
        """
        For some reason zeroing grads for action heads that have been used in prior tasks is required when using
        nn.ModuleList. Otherwise, the backward pass will modify previously used action heads.
        Args:
            task_id: Int.

        """
        for i in range(self.num_task_heads):
            if i != task_id:
                if self.stochastic_policy:
                    self.mu[i].zero_grad(set_to_none=True)
                    self.log_std[i].zero_grad(set_to_none=True)
                else:
                    self.action_pred[i].zero_grad(set_to_none=True)

    def setup_policy(self):
        act_dim = self.config.act_dim if not self.is_discrete else self.action_space.n
        action_pred_in_dim = self.config.hidden_size
        if self.stochastic_policy:
            if self.num_task_heads > 1:
                self.mu = nn.ModuleList([self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)
                                         for _ in range(self.num_task_heads)])
                self.log_std = nn.ModuleList([self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)
                                              for _ in range(self.num_task_heads)])
            else:
                self.mu = self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)
                self.log_std = self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)
            self.action_dist = SquashedDiagGaussianDistribution(act_dim) if self.config.action_tanh \
                else DiagGaussianDistribution(act_dim)
        else:
            if self.num_task_heads > 1:
                self.action_pred = nn.ModuleList([nn.Sequential(
                    *([self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)] + ([nn.Tanh()] if self.config.action_tanh else []))
                ) for _ in range(self.num_task_heads)])
            else:
                self.action_pred = nn.Sequential(
                    *([self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)] + ([nn.Tanh()] if self.config.action_tanh else []))
                )

    @staticmethod
    def make_head(in_dim, out_dim, n_layer=1):
        layers = []
        for _ in range(n_layer - 1):
            layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def setup_prompt(self):
        if self.separate_prompt_embed:
            self.prompt_embed_timestep = nn.Embedding(self.config.max_ep_len, self.config.hidden_size)
            self.prompt_embed_return = torch.nn.Linear(1, self.config.hidden_size)
            self.prompt_embed_reward = torch.nn.Linear(1, self.config.hidden_size)
            self.prompt_embed_state = torch.nn.Linear(self.config.state_dim, self.config.hidden_size)
            self.prompt_embed_action = torch.nn.Linear(self.config.act_dim, self.config.hidden_size)
            self.prompt_embed_ln = torch.nn.LayerNorm(self.config.hidden_size)
        if self.learnable_prompt:
            prompt_kwargs = {} if self.prompt_kwargs is None else self.prompt_kwargs.copy()
            self.prompt_prob = prompt_kwargs.pop("prob", 1.0)
            mod_img_encoder = prompt_kwargs.pop("mod_img_encoder", False)
            if hasattr(self, "embed_image") and mod_img_encoder:
                img_encoder_dims = self.embed_image.get_layer_out_dims()
                prompt_kwargs["img_encoder_dims"] = img_encoder_dims
            self.prompt = make_prompt(self.config, prompt_kwargs)

    def predict_action(self, x, deterministic=False, task_id=None):
        if not self.stochastic_policy:
            if self.num_task_heads > 1:
                assert task_id is not None
                action = self.action_pred[task_id](x)
            else:
                action = self.action_pred(x)
            return action
        mean_actions, log_std, kwargs = self.get_action_dist_params(x, task_id)
        in_shape = mean_actions.shape

        mean_actions = mean_actions.reshape(-1, self.config.act_dim)
        log_std = log_std.reshape(-1, self.config.act_dim)
        # Note: the action is squashed
        action = self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic)
        return action.reshape(*in_shape)

    def get_action_dist_params(self, x_latent, task_id=None):
        if self.num_task_heads > 1:
            assert task_id is not None
            mean_actions = self.mu[task_id](x_latent)
            log_std = self.log_std[task_id](x_latent)
        else:
            mean_actions = self.mu(x_latent)
            log_std = self.log_std(x_latent)
        # Original Implementation to cap the standard deviation
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def action_log_prob(self, x_latent, task_id=None):
        mean_actions, log_std, kwargs = self.get_action_dist_params(x_latent, task_id)
        in_shape = mean_actions.shape
        mean_actions = mean_actions.reshape(-1, self.config.act_dim)
        log_std = log_std.reshape(-1, self.config.act_dim)
        # return action and associated log prob
        action, log_prob = self.action_dist.log_prob_from_params(mean_actions, log_std)
        action = action.reshape(*in_shape)
        log_prob = log_prob.reshape(-1, action.shape[1], 1)
        return action, log_prob

    def compute_log_prob_given_action(self, action):
        return self.action_dist.log_prob(action)

    def get_action_embeddings(self, action, attention_mask=None):
        if self.is_discrete:
            action = action.flatten(start_dim=1)
        emb = self.embed_action(action)
        return emb

    def get_state_embeddings(self, state, mod_vectors=None):
        if len(state.shape) > 4:
            # is_image_space
            batch_size, seq_len = state.shape[0], state.shape[1]
            state = state.reshape(-1, *self.observation_space.shape)
            return self.embed_image(state, mod_vectors=mod_vectors).reshape(batch_size, seq_len, self.config.hidden_size)
        return self.embed_state(state)

    def get_return_embeddings(self, returns):
        return_embeddings = None
        if self.rtg_condition:
            if self.symlog_transform: 
                returns = torch.sign(returns) * torch.log(1 + torch.abs(returns))
            return_embeddings = self.embed_return(returns)
        return return_embeddings
    
    def get_reward_embeddings(self, rewards): 
        reward_embeddings = None
        if self.reward_condition:
            if self.symlog_transform: 
                rewards = torch.sign(rewards) * torch.log(1 + torch.abs(rewards))
            reward_embeddings = self.embed_rewards(rewards)
        return reward_embeddings

    def get_time_embeddings(self, timesteps, attention_mask=None):
        if self.relative_pos_embds or self.both_time_embds: 
            batch_size, context_len = timesteps.shape[0], timesteps.shape[-1]
            pos = torch.arange(context_len, device=timesteps.device)
            pos = pos.repeat(batch_size).reshape(*timesteps.shape)
            if self.both_time_embds:
                return self.embed_timestep(timesteps) + self.embed_timestep_rel(pos) 
            return self.embed_timestep(pos)
        return self.embed_timestep(timesteps)

    def extract_task_id(self, states):
        return states[0, -1, -self.num_task_heads:].argmax()

    def forward(
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
        deterministic=True,
        with_log_probs=False,
        prompt=None,
        task_id=None,
        ddp_kwargs=None
    ):
        if ddp_kwargs is None:
            ddp_kwargs = {}
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.is_discrete and actions.shape[-1] == 1:
            # assumes discrete action spaces are always one-dimensional
            actions = actions.long()

        x, encoder_outputs, prompt_infos = self.compute_hidden_states(
            states=states, actions=actions, rewards=rewards, returns_to_go=returns_to_go, timesteps=timesteps,
            attention_mask=attention_mask, output_hidden_states=output_hidden_states,
            output_attentions=output_attentions, return_dict=return_dict, prompt=prompt,
            task_id=task_id
        )
        state_preds, action_preds, action_log_probs, return_preds, reward_preds, action_logits, entropy = self.get_predictions(
            x, with_log_probs=with_log_probs, deterministic=deterministic,
            task_id=task_id if self.num_task_heads != 1 else None
        )

        if not return_dict:
            if with_log_probs:
                return (state_preds, action_preds, action_log_probs, return_preds)
            return (state_preds, action_preds, return_preds)
        
        # when using DDP all output tensors need to contribute to the loss computation
        # this is not the case for: predict_reward, predict_return, predict_state
        # remove from outputs in case of DDP and not using reward prediction, return prediction or state prediction loss
        return OnlineDecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds if ddp_kwargs.get("predict_state") else None,
            action_preds=action_preds,
            return_preds=return_preds if ddp_kwargs.get("predict_return") else None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            action_log_probs=action_log_probs,
            reward_preds=reward_preds if ddp_kwargs.get("predict_reward") else None,
            action_logits=action_logits,
            entropy=entropy,
            last_encoder_output=x,
            prompt_infos=prompt_infos,
            cross_attentions=encoder_outputs.cross_attentions
        )

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

        prompt_infos, prompt_hidden_states, prompt_attention_mask = None, None, None
        if prompt is not None or (self.learnable_prompt and np.random.rand() < self.prompt_prob):
            if self.learnable_prompt:
                # L2P or Hopfield prompt
                learnable_prompt_inputs = self.compute_learnable_prompt_inputs(
                    stacked_inputs, stacked_attention_mask, output_attentions,
                    output_hidden_states, return_dict, task_id
                )
                prompt_stacked_inputs, prompt_stacked_attention_mask = None, None
                if learnable_prompt_inputs is not None: 
                    prompt_stacked_inputs = learnable_prompt_inputs["prompt_stacked_inputs"]
                    prompt_stacked_attention_mask = learnable_prompt_inputs["prompt_stacked_attention_mask"]
                    prompt_infos = learnable_prompt_inputs["prompt_infos"]
            else:
                # fixed prompt (expert trjs)
                prompt_stacked_inputs, prompt_stacked_attention_mask = self.compute_prompt_inputs(prompt)
            if prompt_stacked_inputs is None and prompt_stacked_attention_mask is None:
                pass
            elif not self.config.add_cross_attention:
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)
            else:
                prompt_hidden_states = prompt_stacked_inputs
                prompt_attention_mask = prompt_stacked_attention_mask if self.use_crossattn_mask else None

        # make position ids
        if self.global_pos_embds:
            position_ids = torch.arange(stacked_attention_mask.shape[1], device=stacked_inputs.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0)
        else: 
            position_ids = torch.zeros(stacked_attention_mask.shape, device=stacked_inputs.device, dtype=torch.long)
        
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
            encoder_attention_mask=prompt_attention_mask
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

    def embed_inputs(self, states, actions, returns_to_go, rewards, attention_mask):
        if len(states.shape) > 4:
            # is_image_space
            states = states.float() / 255.0
        state_embeddings = self.get_state_embeddings(states)
        action_embeddings = self.get_action_embeddings(actions, attention_mask=attention_mask)
        return_embeddings = self.get_return_embeddings(returns_to_go)
        reward_embeddings = self.get_reward_embeddings(rewards)
        return state_embeddings, action_embeddings, return_embeddings, reward_embeddings

    def add_pos_embeddings(self, time_embeddings, state_embeddings, action_embeddings,
                           return_embeddings, reward_embeddings):
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        if action_embeddings is not None: 
            action_embeddings = action_embeddings + time_embeddings
        if return_embeddings is not None:
            return_embeddings = return_embeddings + time_embeddings
        if reward_embeddings is not None:
            reward_embeddings = reward_embeddings + time_embeddings
        return state_embeddings, action_embeddings, return_embeddings, reward_embeddings

    def construct_inputs_and_masks(self, state_embeddings, action_embeddings, returns_embeddings, 
                                   rewards_embeddings, attention_mask, time_embeddings=None):
        if self.reward_condition or self.reward_condition_only:
            if self.reward_condition_only:
                inputs = (state_embeddings, action_embeddings, rewards_embeddings)
                self.tok_to_pred_pos = {"s": len(inputs) - 1, "a": 0, "r": 1}
                self.tok_to_pos = {"s": 0, "a": 1, "r": 2}
            else:
                inputs = (state_embeddings, returns_embeddings, action_embeddings, rewards_embeddings)
                self.tok_to_pred_pos = {"s": len(inputs) - 1, "rtg": 0, "a": 1, "r": 2}
        elif not self.rtg_condition:
            inputs = (state_embeddings, action_embeddings)
            self.tok_to_pred_pos = {"s": len(inputs) - 1, "a": 0}
            self.tok_to_pos = {"s": 0, "a": 1}
        else:
            inputs = (returns_embeddings, state_embeddings, action_embeddings)
            self.tok_to_pred_pos = {"s": 0, "rtg": len(inputs - 1), "a": 0}
            self.tok_to_pos = {"s": 1, "rtg": 0, "a": 2}
        masks = tuple([attention_mask] * len(inputs))
        return inputs, masks
    
    def prepare_inputs_and_masks(self, inputs, masks, batch_size, seq_length):
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        # shape: [batch_size, len(inputs) * context_len, hidden_size]
        if self.separate_ln: 
            # separate layernorms per token type
            inputs = self.apply_lns(inputs)
        stacked_inputs = (
            torch.stack(inputs, dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, len(inputs) * seq_length, self.hidden_size)
        )
        if not self.separate_ln: 
            stacked_inputs = self.embed_ln(stacked_inputs)

        # shape: [batch_size, len(inputs) * context_len, hidden_size]
        # to make the attention mask fit the stacked inputs, have to stack it as well
        # shape: [batch_size, len(inputs) * context_len]
        stacked_attention_mask = (
            torch.stack(masks, dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, len(masks) * seq_length)
        )
        return stacked_inputs, stacked_attention_mask

    def apply_lns(self, inputs):
        new_inputs = [0] * len(inputs)
        for tok, pos in self.tok_to_pos.items():
            if not isinstance(pos, (list, range)):
                pos = [pos]
            for p in pos:
                new_inputs[p] = self.embed_ln[tok](inputs[p])
        return new_inputs

    def compute_prompt_inputs(self, prompt):
        # TODO: refactor forward() to use this method - repetition
        states, actions, _, rewards, returns_to_go, timesteps, attention_mask, _, _, _ = prompt
        batch_size, seq_length = actions.shape[0], actions.shape[1]

        # embed each modality with a different head
        if self.separate_prompt_embed:
            state_embeddings = self.prompt_embed_state(states)
            action_embeddings = self.prompt_embed_action(actions)
            returns_embeddings = self.prompt_embed_return(returns_to_go)
        else:
            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings = self.embed_inputs(
                states, actions, returns_to_go, rewards, attention_mask
            )

        if self.use_time_embds:
            if self.share_prompt_pos_embed:
                time_embeddings = self.get_time_embeddings(timesteps.long(), attention_mask=attention_mask).detach()
            elif self.separate_prompt_embed:
                time_embeddings = self.prompt_embed_timestep(timesteps.long())
            else:
                time_embeddings = self.get_time_embeddings(timesteps.long(), attention_mask=attention_mask)
            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings = self.add_pos_embeddings(
                time_embeddings, state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings
            )
        else:
            time_embeddings = None
        if self.detach_prompt_embed:
            assert not self.separate_prompt_embed
            state_embeddings = state_embeddings.detach()
            action_embeddings = action_embeddings.detach()
            returns_embeddings = returns_embeddings.detach()
            time_embeddings = time_embeddings.detach()
            rewards_embeddings = rewards_embeddings.detach()

        inputs, masks = self.construct_inputs_and_masks(
            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings, attention_mask,
            time_embeddings=time_embeddings, 
        )

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        # shape: [batch_size, len(inputs) * context_len, hidden_size]
        stacked_inputs = (
            torch.stack(inputs, dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, len(inputs) * seq_length, self.hidden_size)
        )
        if self.separate_prompt_embed:
            stacked_inputs = self.prompt_embed_ln(stacked_inputs)
        else:
            stacked_inputs = self.embed_ln(stacked_inputs)

        # shape: [batch_size, len(inputs) * context_len, hidden_size]
        # to make the attention mask fit the stacked inputs, have to stack it as well
        # shape: [batch_size, len(inputs) * context_len]
        stacked_attention_mask = (
            torch.stack(masks, dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, len(masks) * seq_length)
        )
        return stacked_inputs, stacked_attention_mask

    def compute_learnable_prompt_inputs(self, stacked_inputs, stacked_attention_mask,
                                        output_attentions=False, output_hidden_states=False, return_dict=False,
                                        task_id=None):
        # stacked_inputs shape: [batch_size, seq_len * 3, hidden_dim]
        prompt_inputs = stacked_inputs
        
        # make position ids
        if self.global_pos_embds:
            position_ids = torch.arange(stacked_attention_mask.shape[1], device=stacked_inputs.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0)
        else: 
            position_ids = torch.zeros(stacked_attention_mask.shape, device=stacked_inputs.device, dtype=torch.long)
            
        if self.prompt.embed_key in ["last", "first", "embed", "second", "third"]:
            # perform second forward pass to obtain sequence representation.
            with torch.no_grad(): 
                encoder_outputs_prompt = self.encoder(
                    inputs_embeds=stacked_inputs,
                    attention_mask=stacked_attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                if self.prompt.embed_key == "embed": 
                    prompt_inputs = encoder_outputs_prompt["hidden_states"][0]
                elif self.prompt.embed_key == "first": 
                    prompt_inputs = encoder_outputs_prompt["hidden_states"][1]
                elif self.prompt.embed_key == "second": 
                    prompt_inputs = encoder_outputs_prompt["hidden_states"][2]
                elif self.prompt.embed_key == "third": 
                    prompt_inputs = encoder_outputs_prompt["hidden_states"][3]
                else: 
                    prompt_inputs = encoder_outputs_prompt['last_hidden_state']
        prompt_output = self.prompt(prompt_inputs, attention_mask=stacked_attention_mask, 
                                    task_id=task_id, tok_to_pos=self.tok_to_pos)
        if prompt_output is None: 
            return None
        prompt_stacked_inputs, prompt_infos = prompt_output['prompt'], prompt_output['infos']

        # make attn mask
        # prompt stacked inputs can be None, in case we do pre-training of keys and don't want to have prompt
        prompt_stacked_attention_mask = None
        if prompt_stacked_inputs is not None: 
            if self.prompt.prefix:
                shape = prompt_stacked_inputs[0].shape
                prompt_stacked_attention_mask = torch.ones(shape[1], shape[3], device=prompt_stacked_inputs[0].device)
            else:
                if isinstance(prompt_stacked_inputs, (tuple, list)):
                    if isinstance(prompt_stacked_inputs[0], (tuple, list)):
                        prompt_stacked_attention_mask = None
                    else:
                        prompt_stacked_attention_mask = torch.ones(prompt_stacked_inputs[0].shape[:2],
                                                                device=prompt_stacked_inputs[0].device)
                else:
                    prompt_stacked_attention_mask = torch.ones(prompt_stacked_inputs.shape[:2],
                                                            device=prompt_stacked_inputs.device)
        return {"prompt_stacked_inputs": prompt_stacked_inputs,
                "prompt_stacked_attention_mask": prompt_stacked_attention_mask,
                "embed_vectors": prompt_output.get("embed_vectors", None), 
                "img_encoder_vectors": prompt_output.get("img_encoder_vectors", None),
                "prompt_infos": prompt_infos}

    def get_predictions(self, x, with_log_probs=False, deterministic=False, task_id=None):
        action_log_probs, reward_preds, action_logits, entropy, return_preds = None, None, None, None, None
        state_preds = self.predict_state(x[:, self.tok_to_pred_pos["s"]])
        if self.rtg_condition:
            return_preds = self.predict_return(x[:, self.tok_to_pred_pos["rtg"]])
        if self.reward_condition:
            reward_preds = self.predict_reward(x[:, self.tok_to_pred_pos["r"]])
        if with_log_probs:
            action_preds, action_log_probs = self.action_log_prob(x[:,  self.tok_to_pred_pos["a"]], task_id=task_id)
        else:
            action_preds = self.predict_action(x[:,  self.tok_to_pred_pos["a"]],
                                               deterministic=deterministic, task_id=task_id)
        return state_preds, action_preds, action_log_probs, return_preds, reward_preds, action_logits, entropy

    def scale_action(self, action):
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def freeze(self, exclude_prompt=True, exclude_action_head=False, exclude_cross_attention=False, 
               exclude_adapters=False, exclude_img_encoder=False, exclude_embeds=False, 
               exclude_lns=False, exclude_layers=None):
        """
        Args:
            exclude_prompt: Bool. if True, the prompt embedding will not be frozen
            exclude_action_head: Bool. if True, the action head will not be frozen
            exclude_layers: List. list of encoder layers to exclude from freezing
        """
        frozen, not_frozen = [], []
        exclude_keys = set()
        if exclude_prompt:
            exclude_keys.update([name for name, _ in self.named_parameters() if "prompt" in name])
        if exclude_action_head:
            if self.stochastic_policy:
                exclude_keys.update([f"mu.{name}" for name, _ in self.mu.named_parameters()])
                exclude_keys.update([f"log_std.{name}" for name, _ in self.log_std.named_parameters()])
            else:
                if hasattr(self, "action_pred"):
                    exclude_keys.update([f"action_pred.{name}" for name, _ in self.action_pred.named_parameters()])
                if hasattr(self, "action_net"):
                    exclude_keys.update([f"action_net.{name}" for name, _ in self.action_net.named_parameters()])
        if exclude_cross_attention:
            exclude_keys.update([name for name, _ in self.named_parameters()
                                 if "crossattention" in name or "ln_cross_attn" in name])
        if exclude_adapters:
            exclude_keys.update([name for name, _ in self.named_parameters() if "adapter" in name])
        if exclude_img_encoder: 
            exclude_keys.update([name for name, _ in self.named_parameters() if "embed_image" in name])    
        if exclude_embeds: 
            exclude_keys.update(
                [name for name, _ in self.named_parameters() 
                 if any(sub in name for sub in
                        ["embed_image", "embed_state", "embed_action", "embed_return", "embed_rewards", 
                         "embed_timestep", "encoder.wte", "encoder.wpe"])]
            )
        if exclude_lns: 
            exclude_keys.update(
                [name for name, _ in self.named_parameters() 
                 if any(sub in name for sub in ["embed_ln", ".ln_"])]
            )
        if exclude_layers is not None:
            assert isinstance(exclude_layers, list) and all([l < 0 for l in exclude_layers])
            layer_idx = list(range(self.config.n_layer))
            exclude_layer_prefix = [f"encoder.h.{str(layer_idx[l])}" for l in exclude_layers]
            exclude_keys.update([name for name, _ in self.named_parameters()
                                 if any([prefix in name for prefix in exclude_layer_prefix])])
        for name, param in self.named_parameters():
            if name in exclude_keys:
                not_frozen.append(name)
                continue
            param.requires_grad = False
            frozen.append(name)
        return frozen, not_frozen

    def load_action_head_weights(self, model_dict):
        if self.stochastic_policy:
            for i in range(len(self.mu)):
                with torch.no_grad():
                    self.mu[i].weight.copy_(model_dict["mu.weight"])
                    self.mu[i].bias.copy_(model_dict["mu.bias"])
                    self.log_std[i].weight.copy_(model_dict["log_std.weight"])
                    self.log_std[i].bias.copy_(model_dict["log_std.bias"])
        else:
            for i in range(len(self.action_pred)):
                with torch.no_grad():
                    self.action_pred[i].weight.copy_(model_dict["action_pred.weight"])
                    self.action_pred[i].bias.copy_(model_dict["action_pred.bias"])
