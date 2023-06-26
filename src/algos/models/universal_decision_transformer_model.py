import torch.nn as nn
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, DiagGaussianDistribution
from stable_baselines3.common import preprocessing
from .online_decision_transformer_model import OnlineDecisionTransformerModel, OnlineDecisionTransformerOutput
from .image_encoders import make_image_encoder


class DummyUDTModel(OnlineDecisionTransformerModel):

    def __init__(self, config, observation_space, action_space, **kwargs):
        """
        Class for testing purposes.
        Replaces transformer policy by regular feedforward net. Keeps the function headers the same in order
        to avoid unnecessary overhead.

        Args:
            config:
            action_space:
            **kwargs:
        """
        super().__init__(config, observation_space, action_space, **kwargs)

    def setup_policy(self):
        features_dim = self.config.hidden_size
        act_dim = self.config.act_dim if not self.is_discrete else self.action_space.n
        if self.is_image_space:
            self.encoder = make_image_encoder(observation_space=self.observation_space,
                                              features_dim=self.config.hidden_size, encoder_kwargs=self.encoder_kwargs)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.config.state_dim, features_dim),
                nn.LayerNorm(features_dim),
                nn.Tanh(),
                nn.Linear(features_dim, features_dim),
                nn.LeakyReLU(),
                nn.Linear(features_dim, features_dim),
                nn.LeakyReLU(),
            )
        if self.stochastic_policy:
            self.mu = nn.Linear(features_dim, act_dim)
            self.log_std = nn.Linear(features_dim, act_dim)
            self.action_dist = SquashedDiagGaussianDistribution(act_dim) if self.config.action_tanh \
                else DiagGaussianDistribution(act_dim)
        else:
            self.action_pred = nn.Sequential(
                *([nn.Linear(features_dim, act_dim)] + ([nn.Tanh()] if self.config.action_tanh else []))
            )
        del self.embed_timestep
        del self.embed_return
        del self.embed_state
        del self.embed_action
        del self.embed_ln
        del self.predict_state
        del self.predict_return

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
    ):
        """
        Overwrites the original forward, as Transformer steps are not required for this model.
        Just takes states and predicts actions.
        """
        if self.is_image_space:
            states = preprocessing.preprocess_obs(states, observation_space=self.observation_space, normalize_images=True)

        if self.is_image_space and len(states.shape) > 4:
            batch_size, seq_len = states.shape[0], states.shape[1]
            state = states.reshape(-1, *self.observation_space.shape)
            x = self.encoder(state).reshape(batch_size, seq_len, self.config.hidden_size)
        else:
            x = self.encoder(states)
        state_preds, action_preds, action_log_probs, return_preds, reward_preds, action_logits, entropy = self.get_predictions(
            x, with_log_probs=with_log_probs, deterministic=deterministic
        )
        return OnlineDecisionTransformerOutput(
            last_hidden_state=None,
            state_preds=state_preds,
            action_preds=action_preds,
            return_preds=return_preds,
            hidden_states=None,
            attentions=None,
            action_log_probs=action_log_probs,
            reward_preds=reward_preds,
            action_logits=action_logits,
            entropy=entropy,
            last_encoder_output=x
        )

    def get_predictions(self, x, with_log_probs=False, deterministic=False, task_id=None):
        action_log_probs, reward_preds, action_logits, entropy = None, None, None, None
        if with_log_probs:
            action_preds, action_log_probs = self.action_log_prob(x, task_id=task_id)
        else:
            action_preds = self.predict_action(x, deterministic=deterministic, task_id=task_id)
        if self.reward_condition:
            reward_preds = self.predict_reward(x)
        return None, action_preds, action_log_probs, None, reward_preds, action_logits, entropy

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
        return self.encoder(states), None, None

    def _init_weights(self, module):
        # use default initialization for dummy net
        pass
