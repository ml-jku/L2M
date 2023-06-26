import gym
from typing import Any, Callable, Dict, Optional, Type, Union
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env
from .atari_utils import AtariWrapperCustom


def extract_env_name(env, task_id=None): 
    """
    Extracts the name of a Gym environment from the env object.
    Instaces of Meta-World environments have a .name attribute, but instances of
    regular Gym environments do not. For regular Gym environments, we extract
    the name via env.unwrapped.spec.id.
    Also handles usage of SubprocVecEnvs from Sb3, which may contain different env instances.

    Args:
        env: Gym environment. 
        task_id: Int. Only used if env is a SubprocVecEnv.
    Returns: 
        env_name: String. Name of the environment.
        
    """    
    if hasattr(env, "envs"): 
        if hasattr(env.envs[0], "envs"):
            # is ContinualLearningEnv
            env_name = env.envs[0].envs[task_id].name
        else:
            # is SubprocVecEnv, but instances may be different
            env_instance = env.envs[task_id] if len(env.envs) > 1 else env.envs[0]
            if hasattr(env_instance, "name"):
                env_name = env_instance.name
            else: 
                env_name = env_instance.unwrapped.spec.id
    else: 
        if hasattr(env, "name"):
            env_name = env.name
        else: 
            env_name = env.unwrapped.spec.id
            
    return env_name
    
    
def make_multi_vec_env(
    env_id,
    seed: Optional[int] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Custom version of make_vec_env function from sb3. 
    We modify the function to allow for multiple different (!) envs.
    
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: List.
    :param seed: the initial seed for the random number generator
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs
    if not isinstance(env_id, list):
        env_id = [env_id]

    def make_env(env_name):
        def _init():
            env = gym.make(env_name, **env_kwargs)
            if seed is not None:
                env.seed(seed)
                env.action_space.seed(seed)
            # timelimit cannot be modified somehow via kwargs in Atari envs, need to set it manually
            # TODO: this is highly atari specific, need to find a better way 
            env._max_episode_steps = 27000 
            env = Monitor(env, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(env_name) for env_name in env_id], **vec_env_kwargs)


def make_multi_atari_env(
    env_id: Union[str, Type[gym.Env]],
    seed: Optional[int] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Custom version of make_atari_env function from sb3. 
    We modify the function to allow for multiple different (!) envs.
    
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: the environment ID or the environment class
    :param seed: the initial seed for the random number generator
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    if env_kwargs is None:
        env_kwargs = {}

    def atari_wrapper(env: gym.Env) -> gym.Env:
        return AtariWrapperCustom(env, **wrapper_kwargs)

    return make_multi_vec_env(
        env_id,
        seed=seed,
        wrapper_class=atari_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
    )


def make_atari_env_custom(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def atari_wrapper(env: gym.Env) -> gym.Env:
        env = AtariWrapperCustom(env, **wrapper_kwargs)
        return env

    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=atari_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
    )
