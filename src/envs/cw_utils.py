import numpy as np
import gym
from copy import deepcopy
from stable_baselines3.common.monitor import Monitor
from continualworld.envs import (
    MT50, META_WORLD_TIME_HORIZON,
    RandomizationWrapper, OneHotAdder, TimeLimit, SuccessCounter,
    get_task_name, get_subtasks,
    ContinualLearningEnv
)


def get_single_env(
        task,
        one_hot_idx: int = 0,
        one_hot_len: int = 1,
        randomization: str = "random_init_all",
        add_task_ids: bool = True
):
    """
    Wrappers for original get_single_env() in CW. Adds functionality to optionally add task ids.

    Returns a single task environment.

    Appends one-hot embedding to the observation, so that the model that operates on many envs
    can differentiate between them.

    Args:
      task: task name or MT50 number
      one_hot_idx: one-hot identifier (indicates order among different tasks that we consider)
      one_hot_len: length of the one-hot encoding, number of tasks that we consider
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.
      add_task_ids: Bool.


    Returns:
      gym.Env: single-task environment
    """
    task_name = get_task_name(task)
    env = MT50.train_classes[task_name]()
    env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
    if add_task_ids:
        env = OneHotAdder(env, one_hot_idx=one_hot_idx, one_hot_len=one_hot_len)
    # else:
    #     env = OneHotAdder(env, one_hot_idx=0, one_hot_len=1)
    # Currently TimeLimit is needed since SuccessCounter looks at dones.
    env = TimeLimit(env, META_WORLD_TIME_HORIZON)
    env = SuccessCounter(env)
    env.name = task_name
    env.num_envs = 1
    return env


class ContinualLearningEnvv2(gym.Env):
    def __init__(self, envs, steps_per_env: int) -> None:
        """
        Same as ContinualLearningEnv, but removes observation_space asserts.
        v2 envs have a different observation space than v1 envs. Thus, cannot use the same asserts.
        """
        self.action_space = envs[0].action_space
        self.observation_space = deepcopy(envs[0].observation_space)
        self.envs = envs
        self.num_envs = len(envs)
        self.steps_per_env = steps_per_env
        self.steps_limit = self.num_envs * self.steps_per_env
        self.cur_step = 0
        self.cur_seq_idx = 0

    def _check_steps_bound(self) -> None:
        if self.cur_step >= self.steps_limit:
            raise RuntimeError("Steps limit exceeded for ContinualLearningEnv!")

    def pop_successes(self):
        all_successes = []
        self.avg_env_success = {}
        for env in self.envs:
            successes = env.pop_successes()
            all_successes += successes
            if len(successes) > 0:
                self.avg_env_success[env.name] = np.mean(successes)
        return all_successes

    def step(self, action):
        self._check_steps_bound()
        obs, reward, done, info = self.envs[self.cur_seq_idx].step(action)
        info["seq_idx"] = self.cur_seq_idx

        self.cur_step += 1
        if self.cur_step % self.steps_per_env == 0:
            # If we hit limit for current env, end the episode.
            # This may cause border episodes to be shorter than 200.
            done = True
            info["TimeLimit.truncated"] = True

            self.cur_seq_idx += 1

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        self._check_steps_bound()
        return self.envs[self.cur_seq_idx].reset()


def get_cl_env(
        tasks, steps_per_task: int, randomization: str = "random_init_all", add_task_ids: bool = True, v2: bool = False
):
    """
    Wrappers for original get_single_env() in CW. Adds functionality to optionally add task ids.

    Returns continual learning environment.

    Args:
      tasks: list of task names or MT50 numbers
      steps_per_task: steps the agent will spend in each of single environments
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.

    Returns:
      gym.Env: continual learning environment
    """
    task_names = [get_task_name(task) for task in tasks]
    num_tasks = len(task_names)
    envs = []
    for i, task_name in enumerate(task_names):
        env = MT50.train_classes[task_name]()
        env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
        if add_task_ids:
            env = OneHotAdder(env, one_hot_idx=i, one_hot_len=num_tasks)
        # else:
        #     env = OneHotAdder(env, one_hot_idx=0, one_hot_len=1)
        env.name = task_name
        env = TimeLimit(env, META_WORLD_TIME_HORIZON)
        env = SuccessCounter(env)
        envs.append(env)
    if v2:
        cl_env = ContinualLearningEnvv2(envs, steps_per_task)
    else:
        cl_env = ContinualLearningEnv(envs, steps_per_task)
    cl_env.name = "ContinualLearningEnv"
    return cl_env


def get_single_cw_env(task, one_hot_idx, one_hot_len, randomization, add_task_ids):
    def make():
        return Monitor(get_single_env(task, one_hot_idx=one_hot_idx, one_hot_len=one_hot_len,
                                      randomization=randomization, add_task_ids=add_task_ids))
    return make


def get_cw_env_constructors(env_names, randomization, remove_task_ids=False, add_task_ids=True):
    if not isinstance(env_names, (list, tuple)):
        env_names = [env_names]
    constructors = []
    one_hot_len = len(env_names) if not remove_task_ids else 1
    for i, task in enumerate(env_names):
        one_hot_idx = i if not remove_task_ids else 0
        constructors.append(
            get_single_cw_env(task, one_hot_idx=one_hot_idx, one_hot_len=one_hot_len,
                              randomization=randomization, add_task_ids=add_task_ids)
        )
    return constructors
