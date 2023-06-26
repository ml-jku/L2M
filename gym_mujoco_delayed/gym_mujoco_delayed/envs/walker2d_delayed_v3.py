from gym.envs.mujoco.walker2d_v3 import Walker2dEnv


class Walker2dEnvDelayed(Walker2dEnv):

    def __init__(self, **kwargs):
        self.reward_sum = 0
        super().__init__(**kwargs)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.reward_sum += reward
        reward = self.reward_sum if done else 0
        return observation, reward, done, info

    def reset_model(self):
        observation = super().reset_model()
        self.reward_sum = 0
        return observation
