import copy
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.policies import ContinuousCritic
from .agent_utils import make_loss_fn
from .decision_transformer_sb3 import DecisionTransformerSb3
from .models import CustomContinuousCritic, DummyUDTModel, MultiHeadContinuousCritic
from ..optimizers import make_optimizer


class UDT(DecisionTransformerSb3):

    def __init__(self, policy, env, policy_delay=1, target_update_interval=1,
                 critic_reward_scale=1, critic_lr=1e-4, critic_gamma=0.99, tau=0.005, target_policy_noise=0.0,
                 critic_optim_kind="adam", target_step_kind="update", use_critic=False,
                 use_policy_target=False, policy_target_as_dt_head=False,
                 critic_as_dt_head=False, critic_share_extractor=False, update_with_critic=False, use_bc_loss=False,
                 detach_policy_dt=False, target_return_loss_fn_type=None,
                 target_reward_loss_fn_type=None, target_state_loss_fn_type=None, critic_arch=None, 
                 critic_grad_norm=None, critic_kwargs=None, critic_last_seq_only=False, policy_ent=True, **kwargs):
        # these args are required for super().__init__(), as this will call _setup_model()
        self.detach_policy_dt = detach_policy_dt
        self.use_policy_target = use_policy_target
        self.policy_target_as_dt_head = policy_target_as_dt_head
        self.critic_as_dt_head = critic_as_dt_head
        self.use_critic = use_critic
        self.critic_arch = critic_arch
        self.critic_share_extractor = critic_share_extractor
        self.critic_kwargs = critic_kwargs if critic_kwargs is not None else {}
        self.critic_optim_kind = critic_optim_kind
        self.target_return_loss_fn_type = target_return_loss_fn_type
        self.target_reward_loss_fn_type = target_reward_loss_fn_type
        self.target_state_loss_fn_type = target_state_loss_fn_type
        self.use_bc_loss = use_bc_loss
        self.update_with_critic = update_with_critic
        self.critic_lr = critic_lr
        self.critic_gamma = critic_gamma
        # call __init__ before initalizing other args. otherwise superclass init may overwrite args.
        super().__init__(policy, env, **kwargs)
        self.target_update_interval = target_update_interval
        self.policy_delay = policy_delay
        self.tau = tau
        self.target_policy_noise = target_policy_noise
        self.target_step_kind = target_step_kind
        # always make sure that critic_reward_scale is scaled by actual environment reward_scale
        self.critic_reward_scale = self.reward_scale * critic_reward_scale
        self.critic_last_seq_only = critic_last_seq_only
        self.critic_grad_norm = critic_grad_norm
        self.policy_ent = policy_ent
        self.dt_state_context_loc = 1 if self.policy.rtg_condition else 0
        # this is just for debugging, keep in for now
        self.use_dummy_udt = isinstance(self.policy, DummyUDTModel)

    def _setup_loss_fns(self, reduction="mean"):
        super()._setup_loss_fns(reduction=reduction)
        self.target_return_loss_fn = None
        self.target_reward_loss_fn = None
        if self.target_return_loss_fn_type is not None:
            self.target_return_loss_fn = make_loss_fn(self.target_return_loss_fn_type).to(self.device)
            self.ddp_kwargs["predict_return"] = True

        if self.target_reward_loss_fn_type is not None:
            assert self.target_reward is not None
            self.target_reward_loss_fn = make_loss_fn(self.target_reward_loss_fn_type).to(self.device)
            self.ddp_kwargs["predict_reward"] = True
        if self.use_bc_loss:
            self.loss_fn = make_loss_fn("mse").to(self.device)
        if self.target_state_loss_fn_type == "mse":
            self.target_state_loss_fn = make_loss_fn("mse").to(self.device)
            self.ddp_kwargs["predict_state"] = True

    def _setup_policy(self):
        super()._setup_policy()
        if self.detach_policy_dt:
            del self.optimizer
            self.optimizer = make_optimizer(self.optimizer_kind,
                                            self.policy.get_action_head_params(), lr=self.learning_rate)

        # setup additional target policy
        if self.use_policy_target:
            if self.policy_target_as_dt_head:
                # only works for deterministic policy right now
                self.policy_target = copy.deepcopy(self.policy.action_pred)
            else:
                self.policy_target = copy.deepcopy(self.policy)
            for param in self.policy_target.parameters():
                param.requires_grad = False
            self.policy_target = self.policy_target.to(self.device)

    def _setup_critic(self):
        if self.use_critic:
            if self.critic_as_dt_head:
                input_dim = self.policy.config.hidden_size
                net_arch = [256] if self.critic_arch is None else self.critic_arch
                critic_class = CustomContinuousCritic
                extractor = self.policy if self.critic_share_extractor else copy.deepcopy(self.policy)
                critic_kwargs = {"action_proj_dim": 64} if not self.critic_kwargs else self.critic_kwargs
                critic_kwargs["update_with_critic"] = self.update_with_critic
            else:
                input_dim = self.observation_space.shape[0]
                net_arch = [256, 256] if self.critic_arch is None else self.critic_arch
                extractor = nn.Identity()
                critic_kwargs = self.critic_kwargs
                if "num_task_heads" in critic_kwargs:
                    critic_class = MultiHeadContinuousCritic
                else:
                    # TODO: for now use MultiHeadContinuousCritic as it supports cw_net architecture
                    critic_class = MultiHeadContinuousCritic
                    # critic_class = ContinuousCritic
            self.critic = critic_class(
                self.observation_space, self.action_space,
                net_arch=net_arch,
                features_extractor=extractor,
                features_dim=input_dim,
                normalize_images=False,
                share_features_extractor=self.critic_share_extractor if not self.update_with_critic else False,
                **critic_kwargs
            )
            self.critic_target = copy.deepcopy(self.critic)
            for param in self.critic_target.parameters():
                param.requires_grad = False
            if self.critic_as_dt_head and self.critic_share_extractor and self.use_policy_target:
                self.critic_target.features_extractor = self.policy_target

            self.critic = self.critic.to(self.device)
            self.critic_target = self.critic_target.to(self.device)
            self.critic_optimizer = make_optimizer(
                self.critic_optim_kind,
                self.critic.get_optim_groups(self.weight_decay),
                lr=self.critic_lr
            )
            self.critic_schedulers = dict()
            if self.critic_as_dt_head:
                if self.warmup_steps > 0 and (not self.critic_share_extractor or self.update_with_critic):
                    # i.e., only use scheduler if we training the extractor individually or
                    # if we share the extractor and update with both actor and critic losses.
                    self.critic_schedulers["critic_warmup"] = torch.optim.lr_scheduler.LambdaLR(
                        self.critic_optimizer,
                        lambda steps: min((steps + 1) / self.warmup_steps, 1)
                    )
            print(self.critic)
            print(self.critic_optimizer)

    def update_policy(self, policy_output, action_targets, attention_mask, ent_coef,
                      ent_tuning=True, return_targets=None, reward_targets=None, state_targets=None,
                      timesteps=None, dones=None, next_states=None, action_mask=None):
        critic_loss_dict, policy_loss_dict = {}, {}
        if self.use_critic:
            critic_loss_dict = self.update_critic(
                policy_output,
                states=state_targets, action_targets=action_targets,
                rewards=reward_targets, rewards_to_go=return_targets,
                ent_coef=ent_coef, attention_mask=attention_mask, timesteps=timesteps,
                dones=dones, next_states=next_states
            )
        if self._n_updates % self.policy_delay == 0:
            policy_loss_dict = super().update_policy(
                policy_output, action_targets, attention_mask, ent_coef,
                ent_tuning=ent_tuning, return_targets=return_targets,
                reward_targets=reward_targets, state_targets=state_targets, dones=dones,
                timesteps=timesteps, next_states=next_states, action_mask=action_mask
            )
        if self._n_updates % self.target_update_interval == 0 and self.target_step_kind == "update":
            if self.use_critic:
                self.update_critic_target()
            if self._n_updates % self.policy_delay == 0 and self.use_policy_target:
                # i.e. only update target policy when current policy has been updated.
                self.update_policy_target()
        return {**policy_loss_dict, **critic_loss_dict}

    def compute_policy_loss(self, policy_output, action_targets, attention_mask, ent_coef,
                            ent_tuning=True, return_targets=None, reward_targets=None,  state_targets=None, dones=None,
                            timesteps=None, next_states=None, action_mask=None):
        loss, loss_dict = self.compute_main_policy_loss(
            policy_output, action_targets, attention_mask, ent_coef,
            ent_tuning=ent_tuning, return_targets=return_targets, state_targets=state_targets,
            dones=dones, timesteps=timesteps, reward_targets=reward_targets, action_mask=action_mask
        )

        # compute return loss
        if self.target_return_loss_fn is not None:
            loss_returns, loss_returns_dict = self.compute_return_prediction_loss(
                return_targets, policy_output.return_preds, attention_mask
            )
            loss += loss_returns
            loss_dict = {**loss_dict, **loss_returns_dict}

        # compute reward loss
        if self.target_reward_loss_fn is not None and self.target_reward is not None:
            loss_rewards, loss_rewards_dict = self.compute_reward_prediction_loss(
                reward_targets, policy_output.reward_preds, attention_mask
            )
            loss += loss_rewards
            loss_dict = {**loss_dict, **loss_rewards_dict}

        # compute bc loss
        if self.use_bc_loss:
            if self.update_with_critic:
                # requires another forward pass
                policy_output = self.policy(
                    states=state_targets, actions=action_targets, rewards=reward_targets, returns_to_go=return_targets,
                    timesteps=timesteps.long(), attention_mask=attention_mask, return_dict=True,
                    with_log_probs=self.stochastic_policy, deterministic=False
                )

            loss_actions, loss_actions_dict = self.compute_action_prediction_loss(
                action_targets, policy_output.action_preds, attention_mask
            )
            loss += loss_actions
            loss_dict = {**loss_dict, **loss_actions_dict}

        # compute state prediction loss
        if self.target_state_loss_fn_type is not None:
            loss_state, loss_state_dict = self.compute_state_prediction_loss(
                state_targets, policy_output.state_preds, attention_mask
            )
            loss += loss_state
            loss_dict = {**loss_dict, **loss_state_dict}

        if hasattr(policy_output, "prompt_infos") and policy_output.prompt_infos is not None:
            loss_dict["loss_actions"] = loss.item()
            loss = self.compute_prompt_loss(loss, policy_output)

        # overwrite previously stored loss
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def compute_main_policy_loss(self, policy_output, action_targets, attention_mask, ent_coef,
                                 ent_tuning=True, return_targets=None, reward_targets=None,
                                 state_targets=None, dones=None, timesteps=None, action_mask=None):
        # TODO: need to refactor this function
        action_preds, action_log_probs = policy_output.action_preds, policy_output.action_log_probs

        # compute loss + update
        loss_dict = {}
        act_dim = action_preds.shape[2]
        # shape: [batch_size, context_len, action_dim] (before masking)
        if len(action_preds.shape) == 3:
            # don't do in discrete action case
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        if self.use_critic and self.loss_fn_type in ("td3", "td3+bc", "sac"):
            # compute Q-values of action_preds:
            if self.critic_as_dt_head:
                if not self.loss_fn_type == "sac" or self.update_with_critic:
                    # in case critic is a DT head, need to make another forward pass
                    # critic is updated before policy, hence already "used" original outputs.
                    policy_output = self.policy(
                        states=state_targets, actions=action_targets, rewards=reward_targets, returns_to_go=return_targets,
                        timesteps=timesteps.long(), attention_mask=attention_mask, return_dict=True,
                        with_log_probs=self.stochastic_policy, deterministic=False
                    )
                    action_preds, action_log_probs = policy_output.action_preds, policy_output.action_log_probs
                    # mask + extract q inputs
                    action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

                if self.critic_share_extractor:
                    if self.use_dummy_udt:
                        q_input = policy_output.last_encoder_output.reshape(-1, self.policy.config.hidden_size)[attention_mask.reshape(-1) > 0]
                    else:
                        q_input = policy_output.last_encoder_output[:, self.dt_state_context_loc].\
                            reshape(-1, self.policy.config.hidden_size)[attention_mask.reshape(-1) > 0]
                else:
                    q_input = self.critic.extract_features(
                        obs=state_targets,
                        actions=action_targets, returns_to_go=return_targets,
                        timesteps=timesteps.long(), attention_mask=attention_mask, raw=self.use_dummy_udt
                    )
                    q_input = q_input.reshape(-1, self.policy.config.hidden_size)[attention_mask.reshape(-1) > 0]

                # we need to detach() regardless if extractor is shared or not.
                # we just want to compute grads wrt to the action_preds not for the
                # q-input here. otherwise q_input grads can modify weights to produce "artificially" high q-vals
                masked_states = state_targets.reshape(-1, state_targets.shape[-1])[attention_mask.reshape(-1) > 0]
                q_input = q_input.detach()
                q_inputs = {"features": q_input, "action_preds": action_preds, "obs": masked_states}
            else:
                q_input = state_targets.reshape(-1, state_targets.shape[-1])[attention_mask.reshape(-1) > 0]
                q_inputs = {"obs": q_input, "actions": action_preds}
            if self.loss_fn_type == "sac":
                q_values = torch.cat(self.critic(**q_inputs), dim=1)
                q_values, _ = torch.min(q_values, dim=1, keepdim=True)
            else:
                q_values = self.critic.q1_forward(**q_inputs)
            if self.last_seq_only:
                # we just integrate this for TD3, TD3+BC, DQN, SAC
                is_last_seq = torch.zeros_like(attention_mask)
                is_last_seq[:, -1] = 1
                is_last_seq = is_last_seq.reshape(-1)[attention_mask.reshape(-1) > 0] > 0
                q_values = q_values[is_last_seq]

        if self.loss_fn_type == "td3":
            loss = -q_values.mean()
        elif self.loss_fn_type == "td3+bc":
            lmbda = 2.5 / q_values.abs().mean().detach()
            td3_loss = -lmbda * q_values.mean()
            # shape: [batch_size, context_len, action_dim] (before masking)
            action_targets = action_targets.reshape(-1, action_targets.shape[-1])[attention_mask.reshape(-1) > 0]
            if self.last_seq_only:
                action_targets = action_targets[is_last_seq]
                action_preds = action_preds[is_last_seq]
            bc_loss = self.loss_fn(action_preds, action_targets)
            loss = bc_loss + td3_loss
            loss_dict["td3_loss"] = td3_loss.item()
            loss_dict["bc_loss"] = bc_loss.item()
            loss_dict["td3_lambda"] = lmbda.item()
        elif self.loss_fn_type == "dqn":
            assert self. use_policy_target
            # action_preds are basically the current Q-values
            # need to make forward pass with target "policy" to compute target Q-values
            with torch.no_grad():
                target_output = self.policy_target(
                    states=state_targets, actions=action_targets, returns_to_go=return_targets,
                    timesteps=timesteps.long(), attention_mask=attention_mask, return_dict=True,
                    with_log_probs=self.stochastic_policy
                )
                next_q_values = target_output.action_preds[:, 1:].reshape(-1, act_dim)
                next_q_values, _ = next_q_values.max(dim=1, keepdim=True)
                reward = reward_targets[:, :-1].reshape(-1, 1)
                dones = dones[:, 1:].reshape(-1, 1)
                target_q_values = (reward * self.critic_reward_scale) + (1 - dones) * self.critic_gamma * next_q_values

            # masking
            mask = attention_mask[:, :-1].reshape(-1) > 0
            target_q_values = target_q_values.reshape(-1, 1)[mask]
            current_q_values = policy_output.action_preds[:, :-1].reshape(-1, act_dim)[mask]
            # get q-values for actions from the replay buffer
            action_targets = action_targets[:, :-1].reshape(-1, action_targets.shape[-1])[mask]
            current_q_values = torch.gather(current_q_values, dim=1, index=action_targets.long())

            if self.last_seq_only:
                # loss is already masked. Only use last sequence for backward pass.
                is_last_seq = torch.zeros_like(attention_mask[:, :-1])
                is_last_seq[:, -1] = 1
                is_last_seq = is_last_seq.reshape(-1)[mask] > 0
                target_q_values = target_q_values[is_last_seq]
                current_q_values = current_q_values[is_last_seq]

            loss = self.loss_fn(current_q_values, target_q_values)
            loss_dict["q_mean"] = current_q_values.mean().item()
            loss_dict["q_min"] = current_q_values.min().item()
            loss_dict["q_max"] = current_q_values.max().item()
        elif self.stochastic_policy:
            if self.loss_fn_type == "nll":
                # shape: [batch_size, context_len, action_dim] (before masking)
                action_targets = action_targets.reshape(-1, action_targets.shape[-1])
                action_targets_log_prob = self.policy.compute_log_prob_given_action(action_targets)
                loss_actions = -action_targets_log_prob.reshape(-1, 1)[attention_mask.reshape(-1) > 0].mean()
            elif self.loss_fn_type == "ce":
                is_continuous_action = action_targets.is_floating_point()
                action_logits = policy_output.action_logits
                act_dim, logits_latent_dim = action_preds.shape[2], action_logits.shape[3]
                # shape: [batch_size x context_len x action_dim] (before masking)
                action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
                if self.policy.tokenize_a and is_continuous_action: 
                    action_targets = self.policy.tokenize_actions(action_targets)
                # shape: [batch_size x context_len x act_dim x latent_dim] (before masking)
                action_logits = action_logits.reshape(-1, act_dim, logits_latent_dim)[attention_mask.reshape(-1) > 0]
                # reshape to have tokens vs. logits
                action_targets, action_logits = action_targets.reshape(-1), action_logits.reshape(-1, logits_latent_dim)
                loss_actions = self.loss_fn(action_logits, action_targets).mean()
            elif self.loss_fn_type == "sac":
                action_log_probs = action_log_probs.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
                if self.last_seq_only:
                    action_log_probs = action_log_probs[is_last_seq]
                if self.policy_ent:
                    loss_actions = (ent_coef * action_log_probs - q_values).mean()
                else:
                    # action_log_probs will be very high if using pretrained CDT, since already saw similar states and
                    # learned corresponding actions. When task changes log probs are not useful. Optionally exclude
                    # when learning prompts
                    loss_actions = -q_values.mean()
            else:
                # shape: [batch_size, context_len, action_dim] (before masking)
                action_targets = action_targets.reshape(-1, action_targets.shape[-1])[attention_mask.reshape(-1) > 0]
                loss_actions = self.loss_fn(action_preds, action_targets)
            if ent_tuning and self.loss_fn_type != "sac":
                # shape: [batch_size, context_len, 1] (before masking)
                action_log_probs = action_log_probs.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
                entropy = -torch.mean(action_log_probs)
                loss = loss_actions - (ent_coef * entropy)
                loss_dict["entropy"] = entropy.item()
            else:
                loss = loss_actions
            loss_dict["loss_actions"] = loss_actions.item()
        else:
            if self.loss_fn_type == "ce" or self.loss_fn_type == "dist_ce":
                action_logits = policy_output.action_logits
                act_dim, logits_latent_dim = action_logits.shape[2], action_logits.shape[3]
                is_continuous_action = action_targets.is_floating_point()
                
                if self.last_seq_only: 
                    action_logits = action_logits[:, -1]
                    action_preds, action_targets = action_preds[:, -1], action_targets[:, -1]
                    attention_mask, action_mask = attention_mask[:, -1], action_mask[:, -1]

                # shape: [batch_size x context_len x action_dim] (before masking)
                action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
                # assumes that discreate action spaces have a single action dimension only
                action_target_tokens = self.policy.tokenize_actions(action_targets) \
                    if self.policy.tokenize_a and is_continuous_action else action_targets.long()
                # shape: [batch_size x context_len x act_dim x latent_dim] (before masking)
                action_logits = action_logits.reshape(-1, act_dim, logits_latent_dim)[attention_mask.reshape(-1) > 0]
                # reshape to have tokens vs. logits
                action_target_tokens, action_logits = action_target_tokens.reshape(-1), action_logits.reshape(-1, logits_latent_dim)
                
                if is_continuous_action and action_mask is not None: 
                    # mask padded action dimensions
                    action_mask = action_mask[attention_mask > 0].reshape(-1)
                    action_target_tokens = action_target_tokens[action_mask > 0]
                    action_logits = action_logits[action_mask > 0]
                loss = self.loss_fn(action_logits, action_target_tokens)   
                
                # compute stats 
                loss_dict["action_acc"] = torchmetrics.functional.classification.accuracy(
                    action_logits, action_target_tokens, num_classes=action_logits.shape[-1], task="multiclass"
                ).item()
                loss_dict["action_acc_top5"] = torchmetrics.functional.classification.accuracy(
                    action_logits, action_target_tokens, num_classes=action_logits.shape[-1], task="multiclass", top_k=5
                ).item()
                if is_continuous_action:
                    # log MSE for continuous actions
                    action_preds = policy_output.action_preds if not self.last_seq_only else policy_output.action_preds[:, -1]
                    action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
                
                    if is_continuous_action and action_mask is not None: 
                        action_preds = action_preds.reshape(-1)[action_mask > 0]
                        action_targets = action_targets.reshape(-1)[action_mask > 0]
                    loss_dict["action_mse"] = torchmetrics.functional.mean_squared_error(action_preds, action_targets).item()
            else:
                action_targets = action_targets.reshape(-1, action_targets.shape[-1])[attention_mask.reshape(-1) > 0]
                loss = self.loss_fn(action_preds, action_targets)
            loss_dict["loss_actions"] = loss.item()

        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def update_critic(self, policy_output, states, action_targets, rewards, rewards_to_go,
                      attention_mask, timesteps, ent_coef=None, dones=None, next_states=None):
        # extract relevant inputs
        state_dim, act_dim = states.shape[-1], action_targets.shape[-1]
        state, next_state = states.reshape(-1, state_dim), next_states.reshape(-1, state_dim)
        action = action_targets.reshape(-1, act_dim)
        reward, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        if self.policy_target_as_dt_head or self.critic_as_dt_head:
            if self.critic_share_extractor and not self.update_with_critic:
                last_encoder_output = policy_output.last_encoder_output.detach() if self.use_dummy_udt \
                    else policy_output.last_encoder_output[:, self.dt_state_context_loc].detach()
            else:
                last_encoder_output = self.critic.extract_features(
                    obs=states, actions=action_targets, returns_to_go=rewards_to_go,
                    timesteps=timesteps.long(), attention_mask=attention_mask, raw=self.use_dummy_udt, flatten=False
                )
            current_eo, next_eo = last_encoder_output.reshape(-1, self.policy.config.hidden_size), \
                                  last_encoder_output.reshape(-1, self.policy.config.hidden_size)

        with torch.no_grad():
            # compute next Q-values
            if self.use_policy_target:
                if self.policy_target_as_dt_head:
                    next_action = self.policy_target(next_eo)
                else:
                    # TODO: won't work anymore, needs refactoring for using next states/actions/etc.
                    target_output = self.policy_target(
                        states=states, actions=action_targets, rewards=rewards, returns_to_go=rewards_to_go,
                        timesteps=timesteps.long(), attention_mask=attention_mask, return_dict=True,
                        with_log_probs=self.stochastic_policy, deterministic=False
                    )
                    next_action = target_output.action_preds.reshape(-1, act_dim)
                    next_features = target_output.last_encoder_output
                    if not self.use_dummy_udt:
                        next_features = next_features[:, self.dt_state_context_loc]
                    next_features = next_features.reshape(-1, next_features.shape[-1])
            else:
                # construct next inputs - append dummies + shift
                next_actions = torch.cat([action_targets, torch.zeros_like(action_targets[:, -1]).unsqueeze(1)], dim=1)
                next_rewards = torch.cat([rewards, torch.zeros_like(rewards[:, -1]).unsqueeze(1)], dim=1)
                next_returns_to_go = torch.cat([rewards_to_go, (rewards_to_go[:, -1] - rewards[:, -1]).unsqueeze(1)], dim=1)
                # ensure timesteps is is not OOB
                next_ts = (timesteps[:, -1] + 1).unsqueeze(1)
                next_ts[next_ts >= self.policy.config.max_ep_len] = self.policy.config.max_ep_len - 1
                next_timesteps = torch.cat([timesteps, next_ts], dim=1)
                next_attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1]).unsqueeze(1)], dim=1)
                # shift
                next_actions, next_rewards, next_returns_to_go, next_timesteps, next_attention_mask = next_actions[:, 1:], \
                    next_rewards[:, 1:], next_returns_to_go[:, 1:], next_timesteps[:, 1:], next_attention_mask[:, 1:]

                next_output = self.policy(
                    states=next_states,
                    actions=next_actions,
                    rewards=next_rewards,
                    returns_to_go=next_returns_to_go,
                    timesteps=next_timesteps.long(),
                    attention_mask=next_attention_mask,
                    return_dict=True,
                    with_log_probs=self.stochastic_policy,
                    deterministic=False,
                    task_id=self.current_task_id_tensor
                )
                next_action = next_output.action_preds.reshape(-1, act_dim)
                action_log_probs = next_output.action_log_probs.reshape(-1, 1)

            if self.target_policy_noise > 0:
                noise = next_action.clone().data.normal_(0, self.target_policy_noise).clamp(-0.5, 0.5)
                next_action = (next_action.clone() + noise).clamp(-1, 1)

            if self.critic_as_dt_head:
                if self.critic_share_extractor and self.use_policy_target:
                    # next_features now comes from target policy here. critic/target policy features
                    # are basically same as critic and policy share encoder (at different syncing freqs, however)
                    next_q_values = self.critic_target(features=next_features, action_preds=next_action, obs=next_state)
                else:
                    next_features = self.critic_target.extract_features(
                        obs=states, actions=action_targets,
                        returns_to_go=rewards_to_go, timesteps=timesteps,
                        attention_mask=attention_mask, raw=self.use_dummy_udt, flatten=False
                    )
                    next_features = next_features.reshape(-1, next_features.shape[-1])
                    next_q_values = self.critic_target(features=next_features, action_preds=next_action, obs=next_state)
            else:
                next_q_values = self.critic_target(next_state, next_action)

            next_q_values = torch.cat(next_q_values, dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            if self.loss_fn_type == "sac":
                # add entropy term
                next_q_values = next_q_values - ent_coef * action_log_probs
            target_q_values = (reward * self.critic_reward_scale) + (1 - dones) * self.critic_gamma * next_q_values

        current_q_values = self.critic(state, action) if not self.critic_as_dt_head \
            else self.critic(action_preds=action, features=current_eo, obs=state)

        # masking
        mask = attention_mask.reshape(-1) > 0
        current_q_values = [q.reshape(-1, 1)[mask] for q in current_q_values]
        target_q_values = target_q_values.reshape(-1, 1)[mask]

        if self.last_seq_only or self.critic_last_seq_only:
            # loss is already masked. Only use last sequence for backward pass.
            is_last_seq = torch.zeros_like(attention_mask)
            is_last_seq[:, -1] = 1
            is_last_seq = is_last_seq.reshape(-1)[mask] > 0
            target_q_values = target_q_values[is_last_seq]
            current_q_values = [q[is_last_seq] for q in current_q_values]

        # Compute critic loss
        critic_loss = 0.5 * sum([F.mse_loss(q, target_q_values) for q in current_q_values])
        self.critic_optimizer.zero_grad(set_to_none=False)
        critic_loss.backward()
        if self._n_updates % 100 == 0 and self.debug:
            self.grad_plotter.plot_grad_flow(self.critic.named_parameters(), f"critic_update,critic,step={self._n_updates}.png")
            self.grad_plotter.plot_grad_flow(self.policy.named_parameters(), f"critic_update,policy,step={self._n_updates}.png")
        if self.critic_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_norm)
        if self.debug:
            params_before = {k: v.clone() for k, v in self.critic.named_parameters()}
        self.critic_optimizer.step()
        if self.debug:
            params_after = {k: v.clone() for k, v in self.critic.named_parameters()}
            has_changed = {k: not torch.equal(params_before[k], params_after[k]) for k in params_before}
            print("Critic udpate", {k: v for k, v in has_changed.items() if v})

        # learning rate schedule
        for name, sched in self.critic_schedulers.items():
            if "warmup" not in name and self._n_updates < self.warmup_steps:
                continue
            sched.step()
            self.logger.record("train/critic_learning_rate", sched.get_last_lr()[0])

        return {"critic_loss": critic_loss.item(),
                "q_mean": current_q_values[0].mean().item(),
                "q_std": current_q_values[0].std().item(),
                "q_min": current_q_values[0].min().item(),
                "q_max": current_q_values[0].max().item(),
                "target_q_mean": target_q_values.mean().item(),
                "target_q_max": target_q_values.max().item()}

    def update_critic_target(self):
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
        if self.critic_share_extractor and not self.use_policy_target:
            # features_extractor params will not be included in critic.parameters()
            polyak_update(self.policy.parameters(), self.critic_target.features_extractor.parameters(), self.tau)

    def update_policy_target(self):
        policy_params = self.policy.action_pred.parameters() if self.policy_target_as_dt_head \
            else self.policy.parameters()
        polyak_update(policy_params, self.policy_target.parameters(), self.tau)

    def compute_return_prediction_loss(self, return_targets, return_preds, attention_mask):
        return_targets = return_targets.reshape(-1)[attention_mask.reshape(-1) > 0]
        if self.target_return_loss_fn_type == "ce":
            return_preds = return_preds.reshape(-1, return_preds.shape[-1])[attention_mask.reshape(-1) > 0]
            return_targets = self.policy.tokenize_rtgs(return_targets).long()
        else:
            return_preds = return_preds.reshape(-1)[attention_mask.reshape(-1) > 0]
        loss = self.target_return_loss_fn(return_preds, return_targets)
        return loss, {"loss_returns": loss.item()}

    def compute_reward_prediction_loss(self, reward_targets, reward_preds, attention_mask):
        reward_targets = reward_targets.reshape(-1)[attention_mask.reshape(-1) > 0]
        if self.target_return_loss_fn_type == "ce":
            reward_preds = reward_preds.reshape(-1, reward_preds.shape[-1])[attention_mask.reshape(-1) > 0]
            reward_targets = self.policy.tokenize_r(reward_targets).long()
        else:
            reward_preds = reward_preds.reshape(-1)[attention_mask.reshape(-1) > 0]
        loss = self.target_return_loss_fn(reward_preds, reward_targets)
        return loss, {"loss_rewards": loss.item()}

    def compute_action_prediction_loss(self, action_targets, action_preds, attention_mask):
        # rather "compute_bc_loss"
        act_dim = action_targets.shape[-1]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(action_preds, action_targets)
        return loss, {"loss_bc": loss.item()}

    def compute_state_prediction_loss(self, state_targets, state_preds, attention_mask):
        obs_dim = state_targets.shape[-1]
        state_targets = state_targets.reshape(-1, obs_dim)[attention_mask.reshape(-1) > 0]
        state_preds = state_preds.reshape(-1, obs_dim)[attention_mask.reshape(-1) > 0]
        loss = self.target_state_loss_fn(state_preds, state_targets)
        return loss, {"loss_state": loss.item()}

    def _on_step(self) -> None:
        """
        This method is called in ``collect_rollouts()`` after each step in the environment.

        """
        super()._on_step()
        if self.target_step_kind == "env" and self.num_timesteps % self.target_update_interval == 0:
            self.update_policy_target()

    def _record_param_count(self):
        super()._record_param_count()
        if self.use_critic:
            counts = self.get_param_count(self.critic, "critic")
            for k, v in counts.items():
                self.logger.record(f"param_counts/{k}", v)
