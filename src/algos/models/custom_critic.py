import torch
import torch.nn as nn
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.torch_layers import create_mlp
from .extractors import create_cwnet


class CustomContinuousCritic(ContinuousCritic):

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        action_proj_dim=0,
        initializer_range=0.02,
        layer_norm=False,
        raw_state=False,
        action_layer_norm=False,
        raw_state_only=False,
        train_action_proj=True,
        repeat_action=False,
        update_with_critic=False,
        **kwargs
    ):
        action_dim = get_action_dim(action_space)
        obs_dim = get_obs_shape(observation_space)[0]
        if action_proj_dim > 0:
            features_dim = features_dim + action_proj_dim - action_dim
        if repeat_action:
            self.action_repeat = features_dim // action_dim
            features_dim = features_dim + (action_dim * self.action_repeat) - action_dim
        if raw_state:
            features_dim += obs_dim
        if raw_state_only:
            features_dim = obs_dim
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs)
        self.raw_state = raw_state
        self.raw_state_only = raw_state_only
        self.repeat_action = repeat_action
        self.update_with_critic = update_with_critic
        self.initializer_range = initializer_range
        self.ln = torch.nn.LayerNorm(features_dim + action_dim) if layer_norm else None
        self.action_ln = torch.nn.LayerNorm(action_dim) if action_layer_norm else None
        self.action_proj = torch.nn.Linear(action_dim, action_proj_dim) if action_proj_dim > 0 else None
        if not train_action_proj and self.action_proj is not None:
            for param in self.action_proj.parameters():
                param.requires_grad = False
        self.apply(self._init_weights)

    def forward(self, action_preds, obs=None, actions=None,
                returns_to_go=None, timesteps=None, attention_mask=None, features=None, raw=False):
        if features is None:
            with torch.set_grad_enabled(not self.share_features_extractor):
                features = self.extract_features(obs, actions, returns_to_go, timesteps, attention_mask, raw=raw)
        if self.action_ln is not None:
            action_preds = self.action_ln(action_preds)
        if self.action_proj is not None:
            action_preds = self.action_proj(action_preds)
        if self.raw_state_only:
            features = obs
        if self.repeat_action:
            action_preds = torch.repeat_interleave(action_preds, self.action_repeat, dim=-1)
        qvalue_input = torch.cat([features, action_preds], dim=1)
        if self.raw_state:
            qvalue_input = torch.cat([qvalue_input, obs], dim=1)
        if self.ln is not None:
            qvalue_input = self.ln(qvalue_input)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, action_preds, obs=None, actions=None,
                returns_to_go=None, timesteps=None, attention_mask=None, features=None, raw=False):
        if features is None:
            with torch.no_grad():
                features = self.extract_features(obs, actions, returns_to_go, timesteps, attention_mask, raw=raw)
        if self.action_ln is not None:
            action_preds = self.action_ln(action_preds)
        if self.action_proj is not None:
            action_preds = self.action_proj(action_preds)
        if self.raw_state_only:
            features = obs
        if self.repeat_action:
            action_preds = torch.repeat_interleave(action_preds, self.action_repeat, dim=-1)
        inp = torch.cat([features, action_preds], dim=1)
        if self.raw_state:
            inp = torch.cat([inp, obs], dim=1)
        if self.ln is not None:
            inp = self.ln(inp)
        return self.q_networks[0](inp)

    def extract_features(self, obs, actions, returns_to_go, timesteps, attention_mask, raw=False, flatten=True):
        assert self.features_extractor is not None, "No features extractor was set"
        features, _, _ = self.features_extractor.compute_hidden_states(
            states=obs, actions=actions, returns_to_go=returns_to_go,
            timesteps=timesteps.long(), attention_mask=attention_mask, return_dict=True
        )
        if raw:
            return features

        features = features[:, 1]

        if flatten:
            features = features.reshape(-1, features.shape[-1])
        return features

    def parameters(self):
        if self.share_features_extractor and not self.update_with_critic:
            return [param for name, param in self.named_parameters() if "features_extractor" not in name]
        else:
            # return super().parameters()
            # filter out action_pred and the likes
            blacklist = ["_pred", "features_extractor.mu", "features_extractor.log_std"]
            return [param for name, param in self.named_parameters() if all(seq not in name for seq in blacklist)]

    def get_optim_groups(self, weight_decay):
        """
        Same as in online_decision_transformer_model.py
        Separates out all parameters to those that will and won't experience regularizing weight decay.

        """
        if self.share_features_extractor and not self.update_with_critic:
            return self.parameters()
        blacklist = ["_pred", "features_extractor.mu", "features_extractor.log_std"]

        from transformers.models.decision_transformer.modeling_decision_transformer import Conv1D
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
            {"params": [param_dict[pn] for pn in sorted(list(decay)) if all(seq not in pn for seq in blacklist)],
             "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if all(seq not in pn for seq in blacklist)],
             "weight_decay": 0.0},
        ]
        optim_groups_names = [
            {"params": [pn for pn in sorted(list(decay)) if all(seq not in pn for seq in blacklist)],
             "weight_decay": weight_decay},
            {"params": [pn for pn in sorted(list(no_decay)) if all(seq not in pn for seq in blacklist)],
             "weight_decay": 0.0},
        ]
        print("Optim groups:\n", optim_groups_names)

        return optim_groups

    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MultiHeadContinuousCritic(ContinuousCritic):
    """
    For Continual World we use a mult-head architecture with one head per task.

    """
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        num_task_heads=1,
        activation_fn=nn.ReLU,
        cw_net=False,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            **kwargs
        )
        self.num_task_heads = num_task_heads

        # recreate Q-networks with multiple task heads
        del self.q_networks
        action_dim = get_action_dim(self.action_space)
        self.q_networks = []
        for idx in range(self.n_critics):
            if cw_net:
                q_net = create_cwnet(features_dim + action_dim, self.num_task_heads, net_arch)
            else:
                q_net = create_mlp(features_dim + action_dim, self.num_task_heads, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs, actions):
        q_values = super().forward(obs, actions)
        # extract desired task head preds
        if self.num_task_heads > 1:
            task_id = self.extract_task_id(obs)
            return tuple(self.extract_task_head_pred(q, task_id) for q in q_values)
        return q_values

    def q1_forward(self, obs, actions):
        q_values = super().q1_forward(obs, actions)
        # extract desired task head preds
        if self.num_task_heads > 1:
            task_id = self.extract_task_id(obs)
            return self.extract_task_head_pred(q_values, task_id)
        return q_values

    def extract_task_head_pred(self, pred, task_id):
        shape = pred.shape[:-1]
        # in shape: [batch_size * seq_len, num_task_heads]
        pred = pred.reshape(*shape, self.num_task_heads, 1)
        # --> [batch_size * seq_len, 1]
        pred = torch.index_select(pred, len(shape), task_id).flatten(start_dim=len(shape))
        return pred

    def extract_task_id(self, states):
        # shape: [batch_size * context_len,  obs_dim + num_task_heads]
        return states[-1, -self.num_task_heads:].argmax()

    def get_optim_groups(self, weight_decay):
        """
        Same as in online_decision_transformer_model.py
        Separates out all parameters to those that will and won't experience regularizing weight decay.

        """
        if self.share_features_extractor and not self.update_with_critic:
            return self.parameters()
        blacklist = ["_pred", "features_extractor.mu", "features_extractor.log_std"]

        from transformers.models.decision_transformer.modeling_decision_transformer import Conv1D
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
            {"params": [param_dict[pn] for pn in sorted(list(decay)) if all(seq not in pn for seq in blacklist)],
             "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if all(seq not in pn for seq in blacklist)],
             "weight_decay": 0.0},
        ]
        optim_groups_names = [
            {"params": [pn for pn in sorted(list(decay)) if all(seq not in pn for seq in blacklist)],
             "weight_decay": weight_decay},
            {"params": [pn for pn in sorted(list(no_decay)) if all(seq not in pn for seq in blacklist)],
             "weight_decay": 0.0},
        ]
        print("Optim groups:\n", optim_groups_names)

        return optim_groups
