import torch
from .universal_decision_transformer_sb3 import UDT


class DiscreteDecisionTransformerSb3(UDT):

    def __init__(self, policy, env, loss_fn="ce", rtg_sample_kwargs=None, **kwargs):
        super().__init__(policy, env, loss_fn=loss_fn, **kwargs)
        self.rtg_sample_kwargs = {} if rtg_sample_kwargs is None else rtg_sample_kwargs

    def get_action_pred(self, policy, states, actions, rewards, returns_to_go, timesteps, attention_mask,
                        deterministic, prompt, is_eval=False, task_id=None, env_act_dim=None):
        inputs = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "attention_mask": attention_mask,
            "return_dict": True,
            "deterministic": deterministic,
            "prompt": prompt,
            "task_id": task_id,
            "ddp_kwargs": self.ddp_kwargs 
        }
        
        # exper-action inference mechanism
        if self.target_return_type == "infer":
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                policy_output = policy(**inputs)
            return_logits = policy_output.return_preds[:, -1]
            return_sample = policy.sample_from_rtg_logits(return_logits, **self.rtg_sample_kwargs)
            inputs["returns_to_go"][0, -1] = return_sample
            
        # autoregressive action prediction
        # e.g., for discretizes continuous action space need to predict each action dim after another
        act_dim = actions.shape[-1] if env_act_dim is None else env_act_dim
        for i in range(act_dim):
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                policy_output = policy(**inputs)
            if not is_eval and self.num_timesteps % 10000 == 0 and self.log_attn_maps:
                self._record_attention_maps(policy_output.attentions, step=self.num_timesteps, prefix="rollout")
                if policy_output.cross_attentions is not None:
                    self._record_attention_maps(policy_output.cross_attentions, step=self.num_timesteps + i,
                                                prefix="rollout_cross", lower_triu=False)
            inputs["actions"][0, -1, i] = policy_output.action_preds[0, -1, i]
        
        action = inputs["actions"][0, -1]
        if env_act_dim is not None: 
            action = action[:act_dim]
        return action, action
