import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


register(
    id="HopperDelayed-v3",
    entry_point="gym_mujoco_delayed.envs.hopper_delayed_v3:HopperEnvDelayed",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)


register(
    id="HalfCheetahDelayed-v3",
    entry_point="gym_mujoco_delayed.envs.halfcheetah_delayed_v3:HalfCheetahEnvDelayed",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)


register(
    id="Walker2dDelayed-v3",
    max_episode_steps=1000,
    entry_point="gym_mujoco_delayed.envs.walker2d_delayed_v3:Walker2dEnvDelayed",
)
