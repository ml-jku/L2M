"""
EWC implementation is based on avalanche_rl and avalanche:
    - https://github.com/ContinualAI/avalanche-rl/blob/ea4516beaaeb1c3ff3562ece24d872b793c6a133/avalanche_rl/training/plugins/ewc.py#L10
    - https://github.com/ContinualAI/avalanche/blob/24f764e5efe090cbfb529184e6c1ff54c5e4dcba/avalanche/training/plugins/ewc.py#L184

"""
import collections
import itertools
import torch
from .universal_decision_transformer_sb3 import UDT
from .discrete_decision_transformer_sb3 import DiscreteDecisionTransformerSb3


def copy_params_dict(model, copy_grad=False):
    """
    Create a list of (name, parameter), where parameter is copied from model.
    The list has as many parameters as model, with the same size.
    :param model: a pytorch model
    :param copy_grad: if True returns gradients instead of parameter values
    """

    if copy_grad:
        return [(k, p.grad.data.clone()) for k, p in model.named_parameters()]
    else:
        return [(k, p.data.clone()) for k, p in model.named_parameters()]


def zerolike_params_dict(model):
    """
    Create a list of (name, parameter), where parameter is initalized to zero.
    The list has as many parameters as model, with the same size.
    :param model: a pytorch model
    """

    return [
        (k, torch.zeros_like(p).to(p.device))
        for k, p in model.named_parameters()
    ]


def onelike_params_dict(model):
    """
    Create a list of (name, parameter), where parameter is initalized to one.
    The list has as many parameters as model, with the same size.
    :param model: a pytorch model
    """
    return [
        (k, torch.ones_like(p).to(p.device))
        for k, p in model.named_parameters()
    ]


class UDTWithEWC(UDT):

    def __init__(self, policy, env, fisher_updates_per_step=10, ewc_start_timestep=0, ewc_start_experience=1,
                 ewc_lambda=400, ewc_mode="separate", keep_importance_data=False,
                 l2=False, ewc_decay_factor=None, **kwargs):
        """
        Allows for including EWC regularization (https://arxiv.org/abs/1612.00796) during DT training.
        We do not include task-specific biases and gains as in the original paper.

        Args:
            policy: Sb3 policy.
            env: Gym Environment.
            fisher_updates_per_step: How many times batches are sampled from the ReplayMemory during
              computation of the Fisher importance. Defaults to 10.
            ewc_start_timestep: Start computing importances and adding penalty only after this many steps.
              Defaults to 0.
            ewc_start_experience: Start computing importances and adding penalty only after this many experiences.
              Defaults to 0.
            ewc_lambda: The larger the lambda, the larger the regularization.
            ewc_mode: `separate` to keep a separate penalty for each previous experience.
              `online` to keep a single penalty summed with a decay factor over all previous tasks.
            l2: If True, use L2 regularization instead of EWC.
            **kwargs
        """
        super().__init__(policy, env, **kwargs)
        self.fisher_updates_per_step = fisher_updates_per_step
        self.ewc_start_timestep = ewc_start_timestep
        self.ewc_start_experience = ewc_start_experience
        self.ewc_lambda = ewc_lambda
        self.ewc_mode = ewc_mode
        self.ewc_decay_factor = ewc_decay_factor
        self.keep_importance_data = keep_importance_data
        self.l2 = l2
        self.saved_params = collections.defaultdict(list)
        self.importances = collections.defaultdict(list)

    def compute_policy_loss(self, policy_output, action_targets, attention_mask, ent_coef,
                            ent_tuning=True, return_targets=None, reward_targets=None,  state_targets=None, dones=None,
                            timesteps=None, next_states=None, action_mask=None):
        loss, loss_dict = super().compute_policy_loss(
            policy_output, action_targets, attention_mask, ent_coef, ent_tuning,
            return_targets, reward_targets, state_targets, dones, timesteps, next_states, action_mask
        )
        if self.num_timesteps >= self.ewc_start_timestep and self.current_task_id >= self.ewc_start_experience:
            penalty = self._before_backward()
            loss += penalty
            loss_dict["ewc_penalty"] = penalty.item()
            loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def _before_backward(self):
        exp_counter = self.current_task_id
        penalty = torch.tensor(0).float().to(self.device)

        if self.ewc_mode == "separate":
            for experience in range(exp_counter):
                for (cur_name, cur_param), (save_name, saved_param), (imp_name, imp) in zip(
                        self.policy.named_parameters(),
                        self.saved_params[experience],
                        self.importances[experience],
                ):
                    assert cur_name == save_name == imp_name, "Parameter names do not match"
                    # dynamic models may add new units
                    # new units are ignored by the regularization
                    n_units = saved_param.shape[0]
                    cur_param = cur_param[:n_units]
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.ewc_mode == "online":
            prev_exp = exp_counter - 1
            for (cur_name, cur_param), (save_name, saved_param), (imp_name, imp) in zip(
                    self.policy.named_parameters(),
                    self.saved_params[prev_exp],
                    self.importances[prev_exp],
            ):
                assert cur_name == save_name == imp_name, "Parameter names do not match"
                # dynamic models may add new units
                # new units are ignored by the regularization
                n_units = saved_param.shape[0]
                cur_param = cur_param[:n_units]
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError("Wrong EWC mode.")

        return self.ewc_lambda * penalty

    def _after_training_exp(self):
        # compute fisher information on task switch
        importances = self.compute_importances()
        self.update_importances(importances)
        self.saved_params[self.current_task_id] = copy_params_dict(self.policy)
        # clear previous parameter values
        if self.current_task_id > 0 and not self.keep_importance_data:
            del self.saved_params[self.current_task_id - 1]

    def compute_importances(self):
        self.policy.train()
        if self.l2:
            return onelike_params_dict(self.policy)
        importances = zerolike_params_dict(self.policy)

        for _ in range(self.fisher_updates_per_step):
            observations, actions, next_observations, rewards, rewards_to_go, timesteps, attention_mask, \
                dones, task_ids, trj_ids, action_targets, action_mask, prompt = self.sample_batch(self.batch_size)

            # forward pass
            policy_output = self.policy(
                states=observations,
                actions=actions,
                rewards=rewards,
                returns_to_go=rewards_to_go,
                timesteps=timesteps.long(),
                attention_mask=attention_mask,
                return_dict=True,
                with_log_probs=self.stochastic_policy,
                deterministic=False,
                task_id=self.current_task_id_tensor
            )

            # compute loss
            loss, loss_dict = self.compute_policy_loss(
                policy_output, action_targets, attention_mask, 0,
                ent_tuning=False, return_targets=rewards_to_go,
                reward_targets=rewards, state_targets=observations,
                dones=dones.float(), timesteps=timesteps, next_states=next_observations
            )
            # compute gradients & record importances
            self.optimizer.zero_grad()
            loss.backward()

            for (k1, p), (k2, imp) in zip(self.policy.named_parameters(), importances):
                assert (k1 == k2)
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over number of batches
        for _, imp in importances:
            imp /= float(self.fisher_updates_per_step)
        self.optimizer.zero_grad()

        return importances

    @torch.no_grad()
    def update_importances(self, importances):
        if self.ewc_mode == "separate" or self.current_task_id == 0:
            self.importances[self.current_task_id] = importances
        elif self.ewc_mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                    self.importances[self.current_task_id - 1],
                    importances,
                    fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    self.importances[self.current_task_id].append((k2, curr_imp))
                    continue

                assert k1 == k2, "Error in importance computation."

                self.importances[self.current_task_id].append(
                    (k1, (self.ewc_decay_factor * old_imp + curr_imp))
                )

            # clear previous parameter importances
            if self.current_task_id > 0 and (not self.keep_importance_data):
                del self.importances[self.current_task_id - 1]

        else:
            raise ValueError("Wrong EWC mode.")

    def _on_task_switch(self):
        self._after_training_exp()
        super()._on_task_switch()
        
        
class DiscreteUDTWithEWC(UDTWithEWC, DiscreteDecisionTransformerSb3):
    
    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)
