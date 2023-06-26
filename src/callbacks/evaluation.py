import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, VecTransposeImage, is_vecenv_wrapped
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape, \
    is_image_space, is_image_space_channels_first
from ..algos import DecisionTransformerSb3
from ..envs.env_utils import extract_env_name


def custom_evaluate_policy(
    model: "DecisionTransformerSb3",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    task_id: int = 0
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Custom version of evaluate_policy() which works with Decision Transformer. Decision Transformer requires us
    to keep track of the sequences.

    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
        env = VecTransposeImage(env)
    
    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    obs_shape = get_obs_shape(env.observation_space)
    action_dim = get_action_dim(env.action_space)
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    task_id_tensor = torch.tensor(task_id, device=model.device, dtype=torch.int32)
    current_timestep = 0
    target_returns = []
    target_return_val = model.compute_target_return_val(env=env, task_id=task_id)
    target_return = torch.tensor(target_return_val, device=model.device, dtype=torch.float32).reshape(1, 1)
    state = torch.from_numpy(observations[0]).to(device=model.device).reshape(1, *obs_shape)
    states = torch.zeros((1, *obs_shape), device=model.device, dtype=torch.float32)
    states[0] = state
    actions = torch.zeros((0, action_dim), device=model.device, dtype=torch.float32)
    rewards = torch.zeros(0, device=model.device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=model.device, dtype=torch.long).reshape(1, 1)
    
    # domain specific reward scale
    reward_scale = model.get_reward_scale_for_env(envid=extract_env_name(env))
    
    prompt = None
    if model.use_prompt_buffer:
        if not model.prompt_buffer.is_empty(task_id):
            prompt = model.prompt_buffer.sample(batch_size=1, env=model._vec_normalize_env, task_id=task_id)
            prompt = prompt._replace(rewards_to_go=prompt.rewards_to_go / reward_scale,
                                     rewards=prompt.rewards / reward_scale)

    while (episode_counts < episode_count_targets).any():
        actions = torch.cat([actions, torch.zeros((1, action_dim), device=model.device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=model.device)])

        action, _ = model.predict(
            model.policy, states, actions, rewards, target_return, timesteps,
            state=None, episode_start=None, deterministic=deterministic, context_len=model.eval_context_len,
            prompt=prompt, task_id=task_id_tensor, is_eval=True, env_act_dim=action_dim
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()
        if isinstance(env.action_space, gym.spaces.Discrete):
            action = action.astype(int)
        else: 
            action = [action]
        state, reward, done, infos = env.step(action)
        current_rewards += reward
        current_lengths += 1
        current_timestep += 1
        # TODO: build in safeguards to prune context - if gets too long (e.g., Atari)
        # we would store a large amount of unnecessary tensors

        # required for sequence model
        cur_state = torch.from_numpy(state).to(device=model.device).reshape(1, *obs_shape)
        states = torch.cat([states, cur_state], dim=0)
        buffer_reward = torch.from_numpy(reward).to(device=model.device)
        rewards[-1] = buffer_reward / reward_scale
        pred_return = torch.minimum(
            target_return[0, -1] - (buffer_reward / reward_scale),
            torch.tensor(target_return_val, device=model.device, dtype=torch.float32).reshape(1, 1)
        )
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=model.device, dtype=torch.long) * (current_timestep)], dim=1)
        target_returns.append(target_return.mean().item())

        # we will always every have one environment for this evaluation loop.
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                episode_starts[i] = done
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                if done:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(reward)
                        episode_lengths.append(current_lengths[0])
                        episode_counts[i] += 1
                    current_timestep = 0
                    if model.persist_context:
                        # prune context
                        states = states[-model.eval_context_len:]
                        actions = actions[-model.eval_context_len:]
                        rewards = rewards[-model.eval_context_len:]
                        target_return = target_return[:, -model.eval_context_len:]
                        timesteps = timesteps[:, -model.eval_context_len:]
                    else:
                        states = torch.zeros((1, *obs_shape), device=model.device, dtype=torch.float32)
                        target_return = torch.tensor(target_return_val, device=model.device, dtype=torch.float32).reshape(1, 1)
                        actions = torch.zeros((0, action_dim), device=model.device, dtype=torch.float32)
                        rewards = torch.zeros(0, device=model.device, dtype=torch.float32)
                        timesteps = torch.tensor(0, device=model.device, dtype=torch.long).reshape(1, 1)

                    # renew prompt after evaluation episode
                    if model.use_prompt_buffer:
                        if not model.prompt_buffer.is_empty(task_id):
                            prompt = model.prompt_buffer.sample(batch_size=1, env=model._vec_normalize_env,
                                                                task_id=task_id)
                            prompt = prompt._replace(rewards_to_go=prompt.rewards_to_go / reward_scale,
                                                     rewards=prompt.rewards / reward_scale)

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
