"""
For DMControl there are no human-normalized scores. 
Therefore, we normalize the scores based on the performance the expert agent reaches at the end of training. 
--> Data normalized

"""
import math
import dmc2gym
import numpy as np
import pandas as pd
from .env_names import DM_CONTROL_ENVS


# Task: score-tuple dictionary. Each score tuple contains
#  0: score random (float) and 1: mean scores in the datasets (float).
ENVID_TO_DNS = {
    'acrobot-swingup': (8.351, 4.877), 'ball_in_cup-catch': (0.0, 926.719), 'cartpole-balance': (350.391, 938.506),
    'cartpole-swingup': (27.414, 766.15), 'cheetah-run': (3.207, 324.045), 'finger-spin': (0.2, 834.629),
    'finger-turn_easy': (57.8, 800.645), 'finger-turn_hard': (40.6, 676.144), 'fish-swim': (67.675, 78.212),
    'fish-upright': (229.406, 547.962), 'hopper-hop': (0.076, 62.794), 'hopper-stand': (1.296, 266.783), 
    'humanoid-run': (0.741, 0.794), 'humanoid-stand': (4.327, 5.053), 'humanoid-walk': (0.913, 1.194), 
    'manipulator-bring_ball': (0.0, 0.429), 'manipulator-insert_ball': (0.0, 43.307), 
    'manipulator-insert_peg': (0.235, 78.477), 'pendulum-swingup': (0.0, 614.491), 
    'point-mass_easy': (1.341, 779.273), 'reacher-easy': (33.0, 849.241), 'reacher-hard': (8.0, 779.947), 
    'swimmer-swimmer15': (78.817, 152.297), 'swimmer-swimmer6': (229.834, 167.082), 'walker-run': (23.427, 344.794),
    'walker-stand': (134.701, 816.322), 'walker-walk': (30.193, 773.174)
}


def get_data_normalized_score(task: str, raw_score: float, random_col=0, data_col=1) -> float:
    """Converts task score to data-normalized score."""
    scores = ENVID_TO_DNS.get(task, (math.nan, math.nan))
    random, data = scores[random_col], scores[data_col]
    return (raw_score - random) / (data - random)


def compute_random_dmcontrol_scores(): 
    random_scores = {}
    for envid in DM_CONTROL_ENVS:
        domain_name, task_name = envid.split("-")
        print(f"Computing random scores for {envid} ...")
        env = dmc2gym.make(domain_name=domain_name, task_name=task_name)
        random_scores[envid] = evaluate_random_policy(env)
    return random_scores


def evaluate_random_policy(env, n_eval_episodes=10):
    returns = []
    for _ in range(n_eval_episodes):
        _ = env.reset()
        done = False
        episode_return = 0
        while not done:
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            episode_return += reward
        returns.append(episode_return)
    return np.mean(returns)


if __name__ == "__main__": 
    # extract mean scores in data
    # data scores support different file format, than current envids --> map  
    df = pd.read_csv("/home/thomas/Projects-Linux/CRL_with_transformers/continual_rl_with_transformers/postprocessing/data_stats/dm_control_1M_v3/r_stats.csv", index_col=0)
    data_scores = df["returns.mean"].to_dict()
    print(data_scores)    
    
    # compute random scores
    random_scores = compute_random_dmcontrol_scores()
    print(random_scores)
    
    scores = {}
    for k, v in data_scores.items(): 
        scores[k] = (round(random_scores[k], 3), round(v, 3))
    print(scores)
