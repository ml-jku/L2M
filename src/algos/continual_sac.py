"""
Continual SAC wrappers for sb3.
Supports:
    - resetting replay buffer after every task (after every k steps) using ContinualReplayBuffer
    - multi-head architecture for Actor and critic --> i.e., separate head per task
    - handling of Success rate logging (Continual World uses 'success' instead of 'is_success' in info dict)

"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, DiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.utils import polyak_update
from .models import MultiHeadContinuousCritic
from .models.extractors import create_cwnet
from ..envs.env_utils import extract_env_name


class ContinualReplayBuffer(ReplayBuffer):

    def reset(self) -> None:
        print("Reinitializing buffer...")
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape,
                                     dtype=self.observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape,
                                          dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def __len__(self):
        return self.pos


class MultiHeadActor(Actor):

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        num_task_heads=1,
        cw_net=False,
        squash=True,
        **kwargs
    ):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, **kwargs)
        self.num_task_heads = num_task_heads
        # make sure use_sde is not used, as this would make it a bit more complex
        assert not self.use_sde
        del self.action_dist
        del self.mu
        del self.log_std
        self.act_dim = get_action_dim(self.action_space)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.mu = nn.Linear(last_layer_dim, self.act_dim * self.num_task_heads)
        self.log_std = nn.Linear(last_layer_dim, self.act_dim * self.num_task_heads)
        if squash:
            self.action_dist = SquashedDiagGaussianDistribution(self.act_dim)
        else:
            self.action_dist = DiagGaussianDistribution(self.act_dim)
        if cw_net:
            del self.latent_pi
            self.latent_pi = nn.Sequential(*create_cwnet(features_dim, -1, net_arch))

    def get_action_dist_params(self, obs):
        mean_actions, log_std, infos = super().get_action_dist_params(obs)
        if self.num_task_heads > 1:
            task_id = self.extract_task_id(obs)
            mean_actions = self.extract_task_head_pred(mean_actions, task_id)
            log_std = self.extract_task_head_pred(log_std, task_id)
        return mean_actions, log_std, infos

    def extract_task_head_pred(self, pred, task_id):
        shape = pred.shape[:-1]
        # in shape: [batch_size, num_task_heads]
        pred = pred.reshape(*shape, self.num_task_heads, self.act_dim)
        # --> [batch_size, 1]
        pred = torch.index_select(pred, len(shape), task_id).flatten(start_dim=len(shape))
        return pred

    def extract_task_id(self, states):
        # shape: [batch_size,  obs_dim + num_task_heads]
        return states[-1, -self.num_task_heads:].argmax()


class MultiHeadSACPolicy(SACPolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        num_task_heads=1,
        cw_net=False,
        squash=True,
        **kwargs
    ):
        self.num_task_heads = num_task_heads
        self.cw_net = cw_net
        self.squash = squash
        super().__init__(
            observation_space,
            action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )

    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["num_task_heads"] = self.num_task_heads
        actor_kwargs["cw_net"] = self.cw_net
        actor_kwargs["squash"] = self.squash
        return MultiHeadActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["num_task_heads"] = self.num_task_heads
        critic_kwargs["cw_net"] = self.cw_net
        return MultiHeadContinuousCritic(**critic_kwargs).to(self.device)


class ContinualSAC(SAC):

    def __init__(self, policy, env, steps_per_task=1e6, num_task_heads=1, reward_scale=1, save_dir=None,
                 replay_buffer_class=ContinualReplayBuffer, save_buffer=False, target_output_std=None, debug=False,
                 **kwargs):
        """
        Wrapper for SAC, that provides functionality to reset the replay buffer every steps_per_task steps.
        Required for Continual RL setup.
        """
        self.num_task_heads = num_task_heads
        self.target_output_std = target_output_std
        super().__init__(policy, env, replay_buffer_class=replay_buffer_class, **kwargs)
        self.steps_per_task = steps_per_task
        self.save_buffer = save_buffer
        self.save_dir = save_dir
        self.reward_scale = reward_scale
        self.debug = debug

    def _setup_model(self):
        # ensure that MultiHeadSACPolicy is used.
        del self.policy_class
        self.policy_class = MultiHeadSACPolicy
        self.policy_kwargs["num_task_heads"] = self.num_task_heads
        super()._setup_model()
        if self.target_output_std is not None:
            # overwrite target entropy to compute as:
            # https://github.com/awarelab/continual_world/blob/a48e1c0221b22865bd797a569501cde080efe93b/continualworld/sac/sac.py#L199
            target_1d_entropy = np.log(self.target_output_std * math.sqrt(2 * math.pi * math.e))
            self.target_entropy = (
                np.prod(self.env.action_space.shape).astype(np.float32) * target_1d_entropy
            )
            print("Target entropy:", self.target_entropy)

    def learn(self,
              total_timesteps: int,
              callback=None,
              log_interval: int = 4,
              eval_env=None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "SAC",
              eval_log_path=None,
              reset_num_timesteps: bool = True):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps % self.steps_per_task == 0:
                # Reinitialize buffer, after every task --> Continual RL
                self.replay_buffer.reset()

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts \
                    and len(self.replay_buffer) > self.batch_size:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()
        if self.save_buffer:
            assert self.save_dir is not None
            save_path = Path(self.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            env_name = extract_env_name(self.env)
            save_path = save_path / env_name
            # dump replay buffer
            print(f"Saving replay buffer to {save_path}")
            self.save_replay_buffer(save_path)
        return self

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        The original implementation of train() has not reward scale. This is an important param in
        SAC however. We add this here.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                # added reward scale here.
                target_q_values = replay_data.rewards * self.reward_scale + \
                                  (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if self.debug:
            self.logger.record("train/target_q_mean", target_q_values.mean().item())
            self.logger.record("train/target_q_std", target_q_values.std().item())
            self.logger.record("train/target_q_min", target_q_values.min().item())
            self.logger.record("train/target_q_max", target_q_values.max().item())
            self.logger.record("train/q_mean", min_qf_pi.mean().item())
            self.logger.record("train/q_std",  min_qf_pi.std().item())
            self.logger.record("train/q_min",  min_qf_pi.min().item())
            self.logger.record("train/q_max",  min_qf_pi.max().item())
            self.logger.record("train/action_log_probs_mean", log_prob.mean().item())
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _update_info_buffer(self, infos, dones=None) -> None:
        super()._update_info_buffer(infos, dones)
        for idx, info in enumerate(infos):
            # continual_world adds "success", but not is_success
            maybe_is_success = info.get("success")
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def _dump_logs(self) -> None:
        for env in self.env.envs:
            # only used for CW20, no need to handle multiple envs.
            if hasattr(env, "pop_successes"):
                avg_success = np.mean(env.pop_successes())
                self.logger.record("rollout/success", avg_success)
        super()._dump_logs()
