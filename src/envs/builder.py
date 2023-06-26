import collections
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder, VecFrameStack,\
    VecTransposeImage, VecExtractDictObs, VecMonitor
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv, make_atari_env
from .env_names import ID_TO_NAMES, ID_TO_DOMAIN, TASK_SEQS
from .env_utils import make_multi_vec_env, make_multi_atari_env, make_atari_env_custom
from .cw_utils import get_cw_env_constructors
from .dmcontrol_utils import get_dmcontrol_constructor


def extract_eval_env_names(env_names):
    """
    Extractst the actual full env names from given env_names. Env names may contain
    abbreviations such as "atari46" or "mt40", which need to be mapped to the actual environment 
    names for the 46 specified Atari games or the 40 Meta-world tasks, respetively.

    Args:
        env_names: Str or List. 

    Returns: List of env names
    
    """
    if not env_names:
        return None
    if not isinstance(env_names, (list, tuple)):
        env_names = [env_names]
    all_names = []
    for env_name in env_names: 
        names = ID_TO_NAMES.get(env_name, env_name) if isinstance(env_name, str) else list(env_name)
        all_names += names
    return all_names


def get_domains_for_env_names(env_names):
    """
    Extract domain names for each env names. 
    Constructs a dict that maps domains to env names. 

    Args:
        env_names: List of env names.

    Returns:
        Dict. Maps domains to env names.
        
    """
    domain_to_envs = collections.defaultdict(list)
    for name in env_names:
        domain = ID_TO_DOMAIN.get(name, "other")
        if "minihack" in name.lower():
            domain_to_envs["minihack"].append(name)
        else: 
            domain_to_envs[domain].append(name)
    return domain_to_envs


def make_eval_envs(eval_env_names, env_params):
    """
    Generate eval envs from env_names of each domain.

    Args:
        domain_to_envs: Dict. Contains domain-envname pairs. Each envname should be generated.   
        
    Returns: VecEnv
    
    """
    eval_envs = []
    if not isinstance(eval_env_names, (list, tuple)):
        eval_env_names = [eval_env_names]
    domain_to_envs = get_domains_for_env_names(eval_env_names)
    for domain, env_names in domain_to_envs.items(): 
        if domain == "atari": 
            env = make_multi_atari_env(env_names, seed=1, vec_env_cls=DummyVecEnv, 
                                       env_kwargs=env_params.get("env_kwargs", {}),
                                       wrapper_kwargs=env_params.get("wrapper_kwargs", {}))
        elif domain == "other":
            env = make_multi_vec_env(env_names, seed=1, vec_env_cls=DummyVecEnv, 
                                     env_kwargs=env_params.get("env_kwargs", {}))
        elif domain == "mt50": 
            env = DummyVecEnv(
                get_cw_env_constructors(
                    env_names, randomization=env_params.randomization,
                    remove_task_ids=env_params.get("remove_task_ids", False),
                    add_task_ids=env_params.add_task_ids
                )
            )
        elif domain == "dmcontrol": 
            env = DummyVecEnv([get_dmcontrol_constructor(name, env_params.get("dmc_env_kwargs", {})) for name in env_names])
        else: 
            raise NotImplementedError(f"Domain {domain} not implemented yet.")
        env.num_envs = 1
        eval_envs.append(env)
    
    if len(eval_envs) == 1: 
        return eval_envs[0]
    
    def make_env_fn(env):
        def _init():
            return env
        return _init
    
    # compose all envs to new DummyVecEnv. hacky, but necessary for sb3.
    eval_env_fns = []
    for env in eval_envs:
        eval_env_fns += [make_env_fn(e) for e in env.envs]
    eval_env = DummyVecEnv(eval_env_fns)
    eval_env.num_envs = 1 
    return eval_env


def make_env(config, logdir):
    """
    Make train and eval environments. 
    Currently supports creating environments for Gym environments, Atari, Procgen, Meta-world,
    Continual-world and Minihack.
    
    Args:
        config: Hydra config.
        logdir: Str. Path to logdir for saving videos.

    Returns:
        env, eval_env both of type VecEnv.
        
    """
    env_params = config.env_params
    if "Delayed" in env_params.envid:
        # ensures that custom Delayed reward envs are used.
        import gym_mujoco_delayed
    
    domain = ID_TO_DOMAIN.get(env_params.envid, "other")
    eval_env_names = extract_eval_env_names(env_params.get("eval_env_names", None))
        
    if domain == "procgen":
        from procgen import ProcgenEnv
        config.seed = None
        env = ProcgenEnv(env_name=env_params.envid, num_envs=env_params.num_envs,
                         distribution_mode=env_params.distribution_mode)
        eval_env = ProcgenEnv(env_name=env_params.envid, num_envs=1, distribution_mode=env_params.distribution_mode)
        # monitor to obtain ep_rew_mean, ep_rew_len + extract rgb images from dict states
        env = VecMonitor(VecExtractDictObs(env, 'rgb'))
        eval_env = VecMonitor(VecExtractDictObs(eval_env, 'rgb'))
    elif "minihack" in env_params.envid.lower():
        import minihack
        env = make_vec_env(env_params.envid, n_envs=env_params.num_envs, seed=0, vec_env_cls=DummyVecEnv)
        eval_env = make_vec_env(env_params.envid, n_envs=1, seed=1, vec_env_cls=DummyVecEnv)
        env = VecExtractDictObs(env, 'glyphs')
        eval_env = VecExtractDictObs(eval_env, 'glyphs')
    elif domain == "dmcontrol": 
        env = DummyVecEnv([get_dmcontrol_constructor(env_params.envid, env_params.get("dmc_env_kwargs", {}))])
        if not eval_env_names:         
            eval_env = DummyVecEnv([get_dmcontrol_constructor(env_params.envid, env_params.get("dmc_env_kwargs", {}))])   
    elif domain == "cw10":
        from .cw_utils import get_cl_env
        env_names = TASK_SEQS[env_params.envid] if not isinstance(env_params.envid, (list, tuple)) else env_params.envid
        env = VecMonitor(DummyVecEnv([lambda: get_cl_env(tasks=env_names, steps_per_task=env_params.steps_per_task,
                                                         randomization=env_params.randomization,
                                                         add_task_ids=env_params.add_task_ids,
                                                         v2=env_params.envid.endswith("v2"))]))
        eval_env = DummyVecEnv(get_cw_env_constructors(env_names, randomization=env_params.randomization,
                                                       add_task_ids=env_params.add_task_ids))
        # force this to 1 for now. It's 20 individual environments, but in sb3, num_envs refers to env parallelism.
        env.num_envs = 1
        eval_env.num_envs = 1
    elif domain == "mt50":
        env = DummyVecEnv(get_cw_env_constructors(env_params.envid, randomization=env_params.randomization,
                                                  add_task_ids=env_params.add_task_ids))
        if not eval_env_names:
            eval_env = DummyVecEnv(get_cw_env_constructors(env_params.envid,
                                                           randomization=env_params.randomization,
                                                           add_task_ids=env_params.add_task_ids))
            eval_env.num_envs = 1
        env.num_envs = 1
    else:
        env_kwargs = env_params.get("env_kwargs", {})
        wrapper_kwargs = env_params.get("wrapper_kwargs", {})
        env_fn = make_atari_env_custom if domain == "atari" else make_vec_env
        env = env_fn(env_params.envid, n_envs=env_params.num_envs,
                     seed=1, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs, wrapper_kwargs=wrapper_kwargs)
        if not eval_env_names: 
            eval_env = env_fn(env_params.envid, n_envs=1, seed=1, vec_env_cls=DummyVecEnv,
                              env_kwargs=env_kwargs, wrapper_kwargs=wrapper_kwargs)
            eval_env.num_envs = 1
        
    # make eval envs for all given names (can be from multiple domains)
    if eval_env_names: 
        eval_env = make_eval_envs(eval_env_names, env_params)
    
    # additional wrappers (not applied to eval envs, if eval_env_names given)
    if hasattr(env_params, "frame_stack"):
        env = VecFrameStack(env, n_stack=env_params.frame_stack)
        eval_env = VecFrameStack(eval_env, n_stack=env_params.frame_stack) if not eval_env_names else eval_env
    if domain in ["atari", "procgen"]:
        env = VecTransposeImage(env)
        eval_env = VecTransposeImage(eval_env) if not eval_env_names else eval_env
    norm_reward = env_params.get("norm_reward", False)
    if env_params.norm_obs or norm_reward:
        env = VecNormalize(env, norm_obs=env_params.norm_obs, norm_reward=norm_reward, clip_reward=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=env_params.norm_obs, norm_reward=False) \
            if not eval_env_names else eval_env
    if env_params.record:
        env = VecVideoRecorder(env, f"{logdir}/videos",
                               record_video_trigger=lambda x: x % env_params.record_freq == 0,
                               video_length=env_params.record_length)
    return env, eval_env
