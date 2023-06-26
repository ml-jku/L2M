import cv2
import numpy as np
import gym
import dmc2gym_custom
from gym import spaces
from stable_baselines3.common.monitor import Monitor
from dm_control import suite


def extract_obs_dims(exclude_domains=[]):
    obstype_to_dim = {}
    for domain_name, task_name in suite.BENCHMARKING:
        env = suite.load(domain_name, task_name)
        time_step = env.reset()
        print(f"{domain_name}-{task_name}",
              {k: v.shape for k, v in time_step.observation.items()},
              env.action_spec().shape,
              "\n")
        if any(domain in domain_name for domain in exclude_domains):
            continue
        for k, v in time_step.observation.items():
            v = np.array([v]) if np.isscalar(v) else v.ravel()
            obstype_to_dim[k] = max(obstype_to_dim.get(k, 0), v.shape[0])
    return obstype_to_dim 

def extract_obstype_to_startidx(obstype_to_dim):
    cum_dim = 0
    obstype_to_start_idx = {}
    for k, v in obstype_to_dim.items():
        obstype_to_start_idx[k] = cum_dim
        cum_dim += v
    return obstype_to_start_idx


DMC_OBSTYPE_TO_DIM = {
    'orientations': 14, 'velocity': 27, 'position': 8, 'touch': 5, 'target_position': 2, 'dist_to_target': 1, 
    'joint_angles': 21, 'upright': 1, 'target': 3, 'head_height': 1, 'extremities': 12, 'torso_vertical': 3, 
    'com_velocity': 3, 'arm_pos': 16, 'arm_vel': 8, 'hand_pos': 4, 'object_pos': 4, 'object_vel': 3, 'target_pos': 4, 
    'orientation': 2, 'to_target': 2, 'joints': 14, 'body_velocities': 45, 'height': 1
}

DMC_FULL_OBS_DIM = sum(DMC_OBSTYPE_TO_DIM.values())

DMC_OBSTYPE_TO_STARTIDX = {
    'orientations': 0, 'velocity': 14, 'position': 41, 'touch': 49, 'target_position': 54, 'dist_to_target': 56, 
    'joint_angles': 57, 'upright': 78, 'target': 79, 'head_height': 82, 'extremities': 83, 'torso_vertical': 95,
    'com_velocity': 98, 'arm_pos': 101, 'arm_vel': 117, 'hand_pos': 125, 'object_pos': 129, 'object_vel': 133, 
    'target_pos': 136, 'orientation': 140, 'to_target': 142, 'joints': 144, 'body_velocities': 158, 'height': 203
}


def map_obs_to_full_space(obs):
    dtype = obs.dtype if hasattr(obs, "dtype") else np.float32
    full_obs = np.zeros(DMC_FULL_OBS_DIM, dtype=dtype)
    for k, v in obs.items():
        start_idx = DMC_OBSTYPE_TO_STARTIDX[k]
        v = np.array([v]) if np.isscalar(v) else v.ravel()
        full_obs[start_idx: start_idx + v.shape[0]] = v
    return full_obs


def map_flattened_obs_to_full_space(obs, obs_spec): 
    if not isinstance(obs, np.ndarray): 
        obs = np.array(obs)
    is_one_dim = len(obs.shape) == 1
    if is_one_dim: 
        obs = np.expand_dims(obs, axis=0)
    full_obs = np.zeros((*obs.shape[:-1], DMC_FULL_OBS_DIM))
    flat_start_idx = 0
    for k, v in obs_spec.items():
        dim = np.prod(v.shape) if len(v.shape) > 0 else 1
        full_start_idx = DMC_OBSTYPE_TO_STARTIDX[k]
        full_obs[..., full_start_idx: full_start_idx + dim] = obs[..., flat_start_idx: flat_start_idx + dim]
        flat_start_idx += dim
    if is_one_dim:
        full_obs = full_obs.ravel()
    return full_obs

            
class DmcFullObsWrapper(gym.ObservationWrapper):
    """
    Converts a given state observation to the full observation space of all DMControl environments.
    
    Unforunately, dmc2gym always flattens the obsevation by default. Therefore, this wrapper should
    always be used with dmc2gym custom, which make flattening the observation optional. 

    Args: 
        env: Gym environment.
    """

    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        low, high = np.array([-float("inf")] * DMC_FULL_OBS_DIM), np.array([float("inf")] * DMC_FULL_OBS_DIM)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def observation(self, obs):
        return map_obs_to_full_space(obs)
    
    
class GrayscaleWrapper(gym.ObservationWrapper):
    """
    Converts a given frame to grayscale. The given frame must be channel last. 

    Args: 
        env: Gym environment.
    """

    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        channels, height, width, = env.observation_space.shape
        assert channels != 1, "Image is grayscale already."
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, height, width), dtype=env.observation_space.dtype
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        return np.expand_dims(frame, 0)  


def get_dmcontrol_constructor(envid, env_kwargs=None):
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    def make():
        domain_name, task_name = envid.split("-")
        env = dmc2gym_custom.make(domain_name=domain_name, task_name=task_name, **env_kwargs)
        # change envid to make more readable than default in dmc2gym_custom
        env.spec.id = f"{domain_name}-{task_name}"
        if env_kwargs.get("from_pixels", False): 
            env = GrayscaleWrapper(env)
        if not env_kwargs.get("flatten_obs", True): 
            env = DmcFullObsWrapper(env)
        return Monitor(env)
    return make


if __name__ == "__main__": 
    # extract relevant dimensions/indices
    obstype_to_dim = extract_obs_dims()
    print(obstype_to_dim, sum(obstype_to_dim.values()))
    obs_dim = sum(obstype_to_dim.values())
    print(obs_dim)
    obstype_to_start_idx = extract_obstype_to_startidx(obstype_to_dim)
    print(obstype_to_start_idx)
    
    # individual vs full observation space in dmc2gym_custom    
    env = dmc2gym_custom.make(domain_name="cartpole", task_name="swingup")
    print(env.reset())
    env = dmc2gym_custom.make(domain_name="cartpole", task_name="swingup", flatten_obs=False)
    print(env.reset())
    print(env.observation_spec())
    
    # full observation space wrapper
    env = get_dmcontrol_constructor("cartpole-swingup", env_kwargs={"flatten_obs": False})()
    print(env.reset())
    
    # flattened observation to full observation space mapping
    for domain_name, task_name in suite.BENCHMARKING:
        env = dmc2gym_custom.make(domain_name=domain_name, task_name=task_name)
        obs = env.reset()
        print(env.observation_spec(), obs, map_flattened_obs_to_full_space(obs, env.observation_spec()), "\n")

    # flattened observation to full observation space mapping
    env = dmc2gym_custom.make(domain_name="cartpole", task_name="swingup")
    obs = env.reset()
    obs = np.repeat(obs, 100).reshape(100, -1)
    print(env.observation_spec(), obs, map_flattened_obs_to_full_space(obs, env.observation_spec()), "\n")