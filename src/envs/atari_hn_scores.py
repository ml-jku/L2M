# Adjusted from: 
#  - https://github.com/deepmind/dqn_zoo/blob/807379c19f8819e407329ac1b95dcaccb9d536c3/dqn_zoo/atari_data.py
#  - https://github.com/etaoxing/multigame-dt/blob/master/atari_data.py

# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities to compute human-normalized Atari scores.
The data used in this module is human and random performance data on Atari-57.
It comprises of evaluation scores (undiscounted returns), each averaged
over at least 3 episode runs, on each of the 57 Atari games. Each episode begins
with the environment already stepped with a uniform random number (between 1 and
30 inclusive) of noop actions.
The two agents are:
* 'random' (agent choosing its actions uniformly randomly on each step)
* 'human' (professional human game tester)
Scores are obtained by averaging returns over the episodes played by each agent,
with episode length capped to 108,000 frames (i.e. timeout after 30 minutes).
The term 'human-normalized' here means a linear per-game transformation of
a game score in such a way that 0 corresponds to random performance and 1
corresponds to human performance.
"""

import math
from .env_names import ATARI_NAME_TO_ENVID


# Game: score-tuple dictionary. Each score tuple contains
#  0: score random (float) and 1: score human (float).
ENVID_TO_HNS = {
    "alien": (227.8, 7127.7),
    "amidar": (5.8, 1719.5),
    "assault": (222.4, 742.0),
    "asterix": (210.0, 8503.3),
    "asteroids": (719.1, 47388.7),
    "atlantis": (12850.0, 29028.1),
    "bank-heist": (14.2, 753.1),
    "battle-zone": (2360.0, 37187.5),
    "beam-rider": (363.9, 16926.5),
    "berzerk": (123.7, 2630.4),
    "bowling": (23.1, 160.7),
    "boxing": (0.1, 12.1),
    "breakout": (1.7, 30.5),
    "centipede": (2090.9, 12017.0),
    "chopper-command": (811.0, 7387.8),
    "crazy-climber": (10780.5, 35829.4),
    "defender": (2874.5, 18688.9),
    "demon-attack": (152.1, 1971.0),
    "double-dunk": (-18.6, -16.4),
    "enduro": (0.0, 860.5),
    "fishing-derby": (-91.7, -38.7),
    "freeway": (0.0, 29.6),
    "frostbite": (65.2, 4334.7),
    "gopher": (257.6, 2412.5),
    "gravitar": (173.0, 3351.4),
    "hero": (1027.0, 30826.4),
    "ice-hockey": (-11.2, 0.9),
    "jamesbond": (29.0, 302.8),
    "kangaroo": (52.0, 3035.0),
    "krull": (1598.0, 2665.5),
    "kung-fu-master": (258.5, 22736.3),
    "montezuma-revenge": (0.0, 4753.3),
    "ms-pacman": (307.3, 6951.6),
    "name-this-game": (2292.3, 8049.0),
    "phoenix": (761.4, 7242.6),
    "pitfall": (-229.4, 6463.7),
    "pong": (-20.7, 14.6),
    "private-eye": (24.9, 69571.3),
    "qbert": (163.9, 13455.0),
    "riverraid": (1338.5, 17118.0),
    "road-runner": (11.5, 7845.0),
    "robotank": (2.2, 11.9),
    "seaquest": (68.4, 42054.7),
    "skiing": (-17098.1, -4336.9),
    "solaris": (1236.3, 12326.7),
    "space-invaders": (148.0, 1668.7),
    "star-gunner": (664.0, 10250.0),
    "surround": (-10.0, 6.5),
    "tennis": (-23.8, -8.3),
    "time-pilot": (3568.0, 5229.2),
    "tutankham": (11.4, 167.6),
    "up-n-down": (533.4, 11693.2),
    "venture": (0.0, 1187.5),
    # Note the random agent score on Video Pinball is sometimes greater than the
    # human score under other evaluation methods.
    "video-pinball": (16256.9, 17667.9),
    "wizard-of-wor": (563.5, 4756.5),
    "yars-revenge": (3092.9, 54576.9),
    "zaxxon": (32.5, 9173.3),
}

# add scores for actual env ids
keys = list(ENVID_TO_HNS.keys())
for k in keys: 
    if not k in ATARI_NAME_TO_ENVID:
        continue
    envid = ATARI_NAME_TO_ENVID[k]
    ENVID_TO_HNS[envid] = ENVID_TO_HNS[k]


def get_human_normalized_score(game: str, raw_score: float, random_col=0, human_col=1) -> float:
    """Converts game score to human-normalized score."""
    game_scores = ENVID_TO_HNS.get(game, (math.nan, math.nan))
    random, human = game_scores[random_col], game_scores[human_col]
    return (raw_score - random) / (human - random)
