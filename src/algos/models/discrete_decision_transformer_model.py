import torch
import torch.nn as nn
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from .online_decision_transformer_model import OnlineDecisionTransformerModel
from .token_learner import TokenLearnerModule
from ...tokenizers_custom import make_tokenizer


class DiscreteDTModel(OnlineDecisionTransformerModel):

    def __init__(self, config, observation_space, action_space, action_channels=256,
                 use_action_pad=True, tokenize_a=True, tokenize_rtg=False, tokenize_s=False, a_pos_embds=False,
                 a_tok_kwargs=None, rtg_tok_kwargs=None, s_tok_kwargs=None, patch_size=None,
                 num_learned_s_tok=None, max_act_dim=None, **kwargs):
        self.action_channels = action_channels
        super().__init__(config, observation_space, action_space, **kwargs)
        self.tokenize_a = tokenize_a
        self.tokenize_rtg = tokenize_rtg
        self.tokenize_s = tokenize_s
        self.patch_size = patch_size
        self.num_learned_s_tok = num_learned_s_tok
        self.a_pos_embds = a_pos_embds
        self.max_act_dim = max_act_dim
        self.a_tok_kwargs = a_tok_kwargs or {}
        self.rtg_tok_kwargs = rtg_tok_kwargs or {}
        self.s_tok_kwargs = s_tok_kwargs or {}
        self.use_action_pad = use_action_pad
        # embed_action is a Linear layer in original implementation, we make a function out of it.
        del self.embed_action
        if self.tokenize_a:
            a_tok_kind = self.a_tok_kwargs.pop('kind', 'minmax')
            self.action_tokenizer = make_tokenizer(a_tok_kind, {'vocab_size': self.action_channels, **self.a_tok_kwargs})
        # i.e., last one is padding idx
        self.action_pad_token = self.action_channels if self.use_action_pad else None
        self.embed_action_disc = nn.Embedding(
            self.action_channels + 1, config.hidden_size, padding_idx=self.action_pad_token
        )
        if self.a_pos_embds:
            self.embed_act_pos = nn.Embedding(
                self.config.act_dim if self.max_act_dim is None else self.max_act_dim, 
                config.hidden_size
            )
        if self.tokenize_rtg:
            del self.embed_return
            del self.predict_return
            num_rtg_bins = self.rtg_tok_kwargs.get('vocab_size', 256)
            rtg_tok_kind = self.rtg_tok_kwargs.pop('kind', 'minmax')
            # TODO: add option for symlog before tokenization then we don't need to have reward scales anywhere
            self.rtg_tokenizer = make_tokenizer(rtg_tok_kind, self.rtg_tok_kwargs)
            self.embed_return = nn.Embedding(num_rtg_bins, config.hidden_size)
            self.predict_return = nn.Linear(config.hidden_size, num_rtg_bins)
        if self.tokenize_s:
            del self.embed_state
            del self.predict_state
            num_s_bins = self.s_tok_kwargs.get('vocab_size', 256)
            s_tok_kind = self.s_tok_kwargs.pop('kind', 'minmax')
            self.s_tokenizer = make_tokenizer(s_tok_kind, self.s_tok_kwargs)
            self.embed_state = nn.Embedding(num_s_bins, config.hidden_size)
            self.predict_state = nn.Linear(config.hidden_size, num_s_bins)
        if self.patch_size is not None: 
            self.setup_patch_encoder()
            
        # i.e., for [rtg, s, a] order, predict actions given the first two tokens
        # predict rtg given the last token (i.e., predict next RTG), predict state given first token
        self.tok_to_pred_pos = {"s": 0, "rtg": 2,  "a": 1}
        self.post_init()
        self.setup_prompt()
        
    def setup_policy(self):
        if self.is_discrete: 
            self.action_channels = self.action_space.n
        
        if self.stochastic_policy:
            self.action_dist = MultiCategoricalDistribution([self.action_channels] * self.config.act_dim)
            out_dim = sum(self.action_dist.action_dims)
        else:
            out_dim = self.action_channels

        if self.num_task_heads > 1:
            self.action_net = nn.ModuleList(
                [self.make_head(self.config.hidden_size, out_dim, self.n_layer_head)
                 for _ in range(self.num_task_heads)]
            )
        else:
            self.action_net = self.make_head(self.config.hidden_size, out_dim, self.n_layer_head)
            
    def setup_patch_encoder(self): 
        # If img_size=(84, 84), patch_size=(14, 14), then P = 84 / 14 = 6.
        # embed_image --> [BT x D x P x P]. 
        self.embed_image = nn.Conv2d(
            in_channels=1,
            out_channels=self.config.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            padding="valid",
        )
        # patch size and img size are assumed to be square
        img_size = self.observation_space.shape[-1]
        num_patches = (img_size // self.patch_size) ** 2
        if self.num_learned_s_tok is not None:
            self.s_token_learner = TokenLearnerModule(self.num_learned_s_tok, self.config.hidden_size)
            num_patches = self.num_learned_s_tok
        self.embed_patch_pos = nn.Parameter(torch.randn(1, 1, num_patches, self.config.hidden_size))

    def get_action_embeddings(self, action, attention_mask=None):
        return self.embed_action(action, attention_mask)

    def embed_action(self, actions, attention_mask=None):
        # tokenize and embeds generated discrete tokens
        if self.tokenize_a and actions.is_floating_point():
            # tokenize only for continuous actions (works, but suboptimal)
            actions = self.tokenize_actions(actions)
        if self.action_pad_token is not None:
            actions[attention_mask == 0] = self.action_pad_token
        act_embeds = self.embed_action_disc(actions)
        if self.a_pos_embds:
            pos = torch.arange(act_embeds.shape[2], device=act_embeds.device)
            act_embeds = act_embeds + self.embed_act_pos(pos)
        return act_embeds
    
    def get_return_embeddings(self, returns):
        if self.tokenize_rtg:
            # "discretize" returns
            returns = self.tokenize_rtgs(returns)
            # nn.Embedding preserves original shape + latent dimension. Remove excess dimension
            return super().get_return_embeddings(returns).squeeze(2)
        return super().get_return_embeddings(returns)

    def get_state_embeddings(self, state, mod_vectors=None):
        if len(state.shape) > 4:
            # is_image_space --> [B x T x C x W x H] 
            batch_size, seq_len, obs_shape = state.shape[0], state.shape[1], state.shape[2:]
            state = state.reshape(-1, *obs_shape)
            if self.patch_size is not None: 
                # patchify -->  [BT x D x P x P]
                image_emb = self.embed_image(state).permute(0, 2, 3, 1)
                if self.num_learned_s_tok is not None: 
                    # employ token learner
                    image_emb = image_emb.reshape(batch_size * seq_len, -1, self.config.hidden_size)
                    image_emb = self.s_token_learner(image_emb)
                # reshape to [B X T x P * P x D] 
                image_emb = image_emb.reshape(batch_size, seq_len, -1, self.config.hidden_size)
                image_emb = image_emb + self.embed_patch_pos
                return image_emb
                        
            return self.embed_image(state, mod_vectors=mod_vectors).reshape(batch_size, seq_len, self.config.hidden_size)
        
        if self.tokenize_s:
            # "discretize" states
            state = self.tokenize_states(state)
            # nn.Embedding preserves original shape + latent dimension. Remove excess dimension
            return super().get_state_embeddings(state).squeeze(2)
        return self.embed_state(state)

    def add_pos_embeddings(self, time_embeddings, state_embeddings, action_embeddings,
                           returns_embeddings, rewards_embeddings):
        if (self.tokenize_s or self.patch_size is not None) and len(state_embeddings.shape) > 3:
            s_shape = state_embeddings.shape
            state_embeddings = state_embeddings + time_embeddings.repeat(1, 1, s_shape[2]).reshape(s_shape)
        else:
            state_embeddings = state_embeddings + time_embeddings
        if returns_embeddings is not None:
            returns_embeddings = returns_embeddings + time_embeddings
        if rewards_embeddings is not None:
            rewards_embeddings = rewards_embeddings + time_embeddings
        if action_embeddings is not None: 
            act_shape = action_embeddings.shape
            action_embeddings = action_embeddings + time_embeddings.repeat(1, 1, act_shape[2]).reshape(act_shape)
        return state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings

    def construct_inputs_and_masks(self, state_embeddings, action_embeddings, returns_embeddings, 
                                   rewards_embeddings, attention_mask, time_embeddings=None):
        action_embeddings = torch.unbind(action_embeddings, dim=2)
        state_embeddings = torch.unbind(state_embeddings, dim=2) \
            if (self.tokenize_s or self.patch_size is not None) and len(state_embeddings.shape) > 3 \
            else (state_embeddings,)
        act_dim = len(action_embeddings)
        s_dim = len(state_embeddings)
        rtg_dim, r_dim = 1, 1
        if self.reward_condition or self.reward_condition_only:
            if self.reward_condition_only:
                inputs = (*state_embeddings, *action_embeddings, rewards_embeddings)
                self.tok_to_pred_pos = {
                    "s": len(inputs) - 1,
                    "a": range(s_dim - 1, s_dim - 1 + act_dim),
                    "r": len(inputs) - 2
                }
                self.tok_to_pos = {"s": range(s_dim), "a": range(s_dim, s_dim + act_dim), "r": len(inputs) - 1}
            else:
                inputs = (*state_embeddings, returns_embeddings, *action_embeddings, rewards_embeddings)
                self.tok_to_pred_pos = {
                    "s": len(inputs) - 1,
                    "rtg": s_dim - 1,
                    "a": range(s_dim + rtg_dim - 1, s_dim + rtg_dim - 1 + act_dim),
                    "r": len(inputs) - 2
                }
                self.tok_to_pos = {"s": range(s_dim), "rtg": s_dim, "a": range(s_dim + rtg_dim, s_dim + rtg_dim + act_dim),
                                   "r": len(inputs) - 1}
        elif not self.rtg_condition:
            inputs = (*state_embeddings, *action_embeddings)
            self.tok_to_pred_pos = {"s": len(inputs) - 1, "a": range(act_dim)}
        else:
            inputs = (returns_embeddings, *state_embeddings, *action_embeddings)
            self.tok_to_pred_pos = {
                "rtg": len(inputs) - 1,
                "s": 0,
                "a": range(rtg_dim + s_dim - 1, rtg_dim + s_dim - 1 + act_dim)
            }
            self.tok_to_pos = {"rtg": 0, "s": range(rtg_dim, rtg_dim + s_dim),
                               "a": range(s_dim + rtg_dim, s_dim + rtg_dim + act_dim)}
        masks = tuple([attention_mask] * len(inputs))
        return inputs, masks
    
    def action_log_prob_logits(self, x_latent):
        if not self.stochastic_policy:
            action_logits = self.action_net(x_latent)
            action = self.get_action_from_logits(action_logits)
            return action, None, action_logits, None
        # in ims: batch_size x context_len x tf_hidden_dim
        batch_size, context_len = x_latent.shape[0], x_latent.shape[1]
        # dims:  batch_size x context_len x action_net_hidden_dim
        action_logits = self.action_net(x_latent)
        # dims: batch_size * context_len x action_net_hidden_dim
        action_logits = action_logits.reshape(-1, self.action_channels * self.config.act_dim)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        # dims: batch_size * context_len x act_dim
        action = distribution.get_actions(deterministic=False)
        # dims: batch_size * context_len x 1
        log_prob = distribution.log_prob(action)
        if self.tokenize_a and action_logits.shape[-1] == self.action_channels:
            # continous action identified based on action_channels, suboptimal
            action = self.inv_tokenize_actions(action)
        # reshape back
        action = action.reshape(batch_size, context_len, self.config.act_dim)
        log_prob = log_prob.reshape(batch_size, context_len)
        # entropy = entropy.reshape(batch_size, context_len)
        action_logits = action_logits.reshape(batch_size, context_len, self.config.act_dim, self.action_channels)
        return action, log_prob, action_logits, None
    
    def get_action_from_logits(self, action_logits): 
        action = torch.argmax(action_logits, dim=-1)
        if self.tokenize_a and action_logits.shape[-1] == self.action_channels: 
            # continous action identified based on action_channels, suboptimal
            action = self.inv_tokenize_actions(action)
        if len(action.shape) == 2:
            action = action.unsqueeze(0)
        return action

    def get_predictions(self, x, with_log_probs=False, deterministic=False, task_id=None):
        action_log_probs, reward_preds, action_logits, entropy = None, None, None, None
        # x: [batch_size x tokens x context_len x hidden_dim]
        return_preds = self.predict_return(x[:, self.tok_to_pred_pos["rtg"]])  # predict next return given state and action
        state_preds = self.predict_state(x[:, self.tok_to_pred_pos["s"]])  # predict next state given state and action
        x_actions = x[:, self.tok_to_pred_pos["a"]].permute(0, 2, 1, 3)
        action_preds, action_log_probs, action_logits, entropy = self.action_log_prob_logits(x_actions)
        if self.reward_condition:
            reward_preds = self.predict_reward(x[:, self.tok_to_pred_pos["r"]])
        return state_preds, action_preds, action_log_probs, return_preds, reward_preds, action_logits, entropy

    def tokenize_actions(self, a):
        return self.action_tokenizer.tokenize(a)

    def inv_tokenize_actions(self, a):
        return self.action_tokenizer.inv_tokenize(a)

    def tokenize_rtgs(self, rtg):
        return self.rtg_tokenizer.tokenize(rtg)

    def inv_tokenize_rtgs(self, rtg):
        return self.rtg_tokenizer.inv_tokenize(rtg)

    def tokenize_states(self, s):
        return self.s_tokenizer.tokenize(s)

    def inv_tokenize_states(self, s):
        return self.s_tokenizer.inv_tokenize(s)

    @staticmethod
    def sample_from_logits(logits, temperature=1.0, top_k=0, top_p=0.0):
        """
        Adjusted from:
            - https://github.com/google-research/google-research/tree/master/multi_game_dt
            - https://github.com/etaoxing/multigame-dt
        """
        if top_p > 0.0:
            # percentile: 0 to 100, quantile: 0 to 1
            percentile = torch.quantile(logits, top_p, dim=-1)
            logits = torch.where(logits > percentile.unsqueeze(-1), logits.double(), -float("inf"))
        if top_k > 0:
            logits, top_indices = torch.topk(logits, top_k)
        sample = torch.distributions.Categorical(logits=temperature * logits).sample()
        if top_k > 0:
            sample_shape = sample.shape
            # Flatten top-k indices and samples for easy indexing.
            top_indices = torch.reshape(top_indices, [-1, top_k])
            sample = sample.flatten()
            sample = top_indices[torch.arange(len(sample)), sample]
            # Reshape samples back to original dimensions.
            sample = torch.reshape(sample, sample_shape)
        return sample

    def sample_from_rtg_logits(self, logits, num_samples=128, opt_weight=0.0, temperature=1.0, top_k=0, top_p=0.5):
        assert len(logits.shape) == 2
        # either of top_k or top_p must be set, but not both
        assert (top_k > 0) ^ (top_p > 0.0) 
        # Add optimality bias.
        if opt_weight > 0.0:
            # Calculate log of P(optimality=1|return) := exp(return) / Z.
            logits_opt = torch.linspace(0.0, 1.0, logits.shape[1])
            logits_opt = torch.repeat_interleave(logits_opt.unsqueeze(0), logits.shape[0], dim=0) 
            # Sample from log[P(optimality=1|return)*P(return)].
            logits = logits + opt_weight * logits_opt
        logits = torch.repeat_interleave(logits.unsqueeze(0), num_samples, dim=0)
        ret_sample = self.sample_from_logits(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        # Pick the highest return sample.
        ret_sample, _ = torch.max(ret_sample, dim=0)
        ret_sample = self.inv_tokenize_rtgs(ret_sample)
        return ret_sample
    
    def load_action_head_weights(self, model_dict):
        for i in range(len(self.action_net)):
            with torch.no_grad():
                self.action_net[i].weight.copy_(model_dict["action_net.weight"])
                self.action_net[i].bias.copy_(model_dict["action_net.bias"])
