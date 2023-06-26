"""
The DQN Replay Dataset was produced with the reduced Atari action set for each game. 
To train the agent across all games, the action space needs to be unified. 
In particular, we need to map the agent's action prediction to the correct action in the 
full action space. The agent's action prediction will always be in the limited action set then.

We adjust the code from: 
- https://github.com/google-research/google-research/blob/master/multi_game_dt/Multi_game_decision_transformers_public_colab.ipynb

This functinoality is not used anymore. 
We directly save the trajectories with actions mapped to the full action space.

"""
import gym
import cv2
import numpy as np
from gym import spaces
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv, EpisodicLifeEnv, \
    FireResetEnv, ClipRewardEnv 

GAME_NAMES = [
    'AirRaidNoFrameskip-v4', 'AlienNoFrameskip-v4', 'AmidarNoFrameskip-v4', 'AssaultNoFrameskip-v4', 'AsterixNoFrameskip-v4',
    'AsteroidsNoFrameskip-v4', 'AtlantisNoFrameskip-v4', 'BankHeistNoFrameskip-v4', 'BattleZoneNoFrameskip-v4', 'BeamRiderNoFrameskip-v4',
    'BerzerkNoFrameskip-v4', 'BowlingNoFrameskip-v4', 'BoxingNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'CarnivalNoFrameskip-v4',
    'CentipedeNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4', 'DemonAttackNoFrameskip-v4',
    'DoubleDunkNoFrameskip-v4', 'ElevatorActionNoFrameskip-v4', 'EnduroNoFrameskip-v4', 'FishingDerbyNoFrameskip-v4', 'FreewayNoFrameskip-v4',
    'FrostbiteNoFrameskip-v4', 'GopherNoFrameskip-v4', 'GravitarNoFrameskip-v4', 'HeroNoFrameskip-v4', 'IceHockeyNoFrameskip-v4',
    'JamesbondNoFrameskip-v4', 'JourneyEscapeNoFrameskip-v4', 'KangarooNoFrameskip-v4', 'KrullNoFrameskip-v4', 'KungFuMasterNoFrameskip-v4',
    'MontezumaRevengeNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'NameThisGameNoFrameskip-v4', 'PhoenixNoFrameskip-v4', 'PitfallNoFrameskip-v4',
    'PongNoFrameskip-v4', 'PooyanNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'QbertNoFrameskip-v4', 'RiverraidNoFrameskip-v4',
    'RoadRunnerNoFrameskip-v4', 'RobotankNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'SkiingNoFrameskip-v4', 'SolarisNoFrameskip-v4',
    'SpaceInvadersNoFrameskip-v4', 'StarGunnerNoFrameskip-v4', 'TennisNoFrameskip-v4', 'TimePilotNoFrameskip-v4', 'TutankhamNoFrameskip-v4',
    'UpNDownNoFrameskip-v4', 'VentureNoFrameskip-v4', 'VideoPinballNoFrameskip-v4', 'WizardOfWorNoFrameskip-v4', 'YarsRevengeNoFrameskip-v4',
    'ZaxxonNoFrameskip-v4'
]

_FULL_ACTION_SET = [
    'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
    'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
    'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
]

_LIMITED_ACTION_SET = {
    "AirRaidNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "AlienNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "AmidarNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE"
    ],
    "AssaultNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "AsterixNoFrameskip-v4": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT"
    ],
    "AsteroidsNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE"
    ],
    "AtlantisNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "BankHeistNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "BattleZoneNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "BeamRiderNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "UPRIGHT",
        "UPLEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "BerzerkNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "BowlingNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "DOWN",
        "UPFIRE",
        "DOWNFIRE"
    ],
    "BoxingNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "BreakoutNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT"
    ],
    "CarnivalNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "CentipedeNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "ChopperCommandNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "CrazyClimberNoFrameskip-v4": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT"
    ],
    "DemonAttackNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "DoubleDunkNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "ElevatorActionNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "EnduroNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "DOWN",
        "DOWNRIGHT",
        "DOWNLEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "FishingDerbyNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "FreewayNoFrameskip-v4": [
        "NOOP",
        "UP",
        "DOWN"
    ],
    "FrostbiteNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "GopherNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "GravitarNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "HeroNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "IceHockeyNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "JamesbondNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "JourneyEscapeNoFrameskip-v4": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "KangarooNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "KrullNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "KungFuMasterNoFrameskip-v4": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "DOWNRIGHT",
        "DOWNLEFT",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "MontezumaRevengeNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "MsPacmanNoFrameskip-v4": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT"
    ],
    "NameThisGameNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "PhoenixNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "DOWN",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE"
    ],
    "PitfallNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "PongNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "PooyanNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "DOWN",
        "UPFIRE",
        "DOWNFIRE"
    ],
    "PrivateEyeNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "QbertNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN"
    ],
    "RiverraidNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "RoadRunnerNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "RobotankNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "SeaquestNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "SkiingNoFrameskip-v4": [
        "NOOP",
        "RIGHT",
        "LEFT"
    ],
    "SolarisNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "SpaceInvadersNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "RIGHT",
        "LEFT",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "StarGunnerNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "TennisNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "TimePilotNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE"
    ],
    "TutankhamNoFrameskip-v4": [
        "NOOP",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "UpNDownNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "DOWN",
        "UPFIRE",
        "DOWNFIRE"
    ],
    "VentureNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "VideoPinballNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE"
    ],
    "WizardOfWorNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE"
    ],
    "YarsRevengeNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ],
    "ZaxxonNoFrameskip-v4": [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
    ]
}


# An array that Converts an action from a game-specific to full action set.
LIMITED_ACTION_TO_FULL_ACTION = {
    game_name: np.array(
[_FULL_ACTION_SET.index(i) for i in _LIMITED_ACTION_SET[game_name]])
    for game_name in GAME_NAMES
}

# An array that Converts an action from a full action set to a game-specific
# action set (Setting 0=NOOP if no game-specific action exists).
FULL_ACTION_TO_LIMITED_ACTION = {
    game_name: np.array([(_LIMITED_ACTION_SET[game_name].index(i)
                          if i in _LIMITED_ACTION_SET[game_name] else 0)
                         for i in _FULL_ACTION_SET]) 
    for game_name in GAME_NAMES
}


class ToLimitedActionWrapper(gym.Wrapper):

    def __init__(self, env):
        """        

        Args:
            env: gym.Env
    
        """
        super().__init__(env)
        self.game_name = env.unwrapped.spec.id

    def step(self, action):
        action = FULL_ACTION_TO_LIMITED_ACTION[self.game_name][action]
        return self.env.step(action)


class WarpFrameCustom(gym.ObservationWrapper):
    """
    Modified original wrapper from Stable Baselines 3:
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py
    
    Adds option to repeat grayscale channel. 

    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, to_rgb=False) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.to_rgb = to_rgb
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3 if self.to_rgb else 1),
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame
        :param frame: environment frame
        :return: the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]
        if self.to_rgb:
           frame = np.repeat(frame, 3, axis=2)
        return frame


class AtariWrapperCustom(gym.Wrapper):
    """
    Modified original wrapper from Stable Baselines 3: 
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py
        
    Add option to repeat the gray-scale channels to make "RGB" image. 
    
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        to_rgb=False
    ) -> None:
        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrameCustom(env, width=screen_size, height=screen_size, to_rgb=to_rgb)
        if clip_reward:
            env = ClipRewardEnv(env)

        super(AtariWrapperCustom, self).__init__(env)
