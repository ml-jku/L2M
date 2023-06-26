from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv


class HalfCheetahEnvDelayed(HalfCheetahEnv):

    def __init__(self, **kwargs):
        """
        By default this does not work for Halfcheetah, as done is always False for this env.
        Always executes 1000 steps by default without exiting earlier.
        This is controlled via max_episode_steps in the TimeLimit wrapper.
        For simplicity, we give this and arg that specifies the max_episode steps, keeps track of the number of
        times step() was called, and returns self.reward_sum once self.max_episode_steps is reached.

        Args:
            **kwargs:
        """
        self.reward_sum = 0
        self.max_episode_steps = 1000
        self.count = 0
        super().__init__(**kwargs)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.count += 1
        self.reward_sum += reward
        reward = self.reward_sum if self.count > self.max_episode_steps else 0
        return observation, reward, done, info

    def reset_model(self):
        observation = super().reset_model()
        self.reward_sum = 0
        self.count = 0
        return observation
