"""
Adjusted from https://github.com/kzl/decision-transformer/blob/master/gym/data/download_d4rl_datasets.py

"""
import argparse
import pathlib
import gym
import numpy as np
import collections
import pickle
import d4rl


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default="hopper", type=str)
	parser.add_argument("--ds_type", default="medium", type=str)
	parser.add_argument("--save_dir", default="../data/d4rl", type=str)
	args = parser.parse_args()

	save_dir = pathlib.Path(args.save_dir)
	save_dir.mkdir(exist_ok=True, parents=True)
	env_names = ['halfcheetah', 'hopper', 'walker2d'] if args.env_name == "all" else [args.env_name]
	dataset_types = ['medium', 'medium-replay', 'expert'] if args.ds_type == "all" else [args.ds_type]

	for env_name in env_names:
		for dataset_type in dataset_types:
			name = f'{env_name}-{dataset_type}-v2'
			env = gym.make(name)
			dataset = env.get_dataset()

			N = dataset['rewards'].shape[0]
			data_ = collections.defaultdict(list)

			use_timeouts = False
			if 'timeouts' in dataset:
				use_timeouts = True

			episode_step = 0
			paths = []
			for i in range(N):
				done_bool = bool(dataset['terminals'][i])
				if use_timeouts:
					final_timestep = dataset['timeouts'][i]
				else:
					final_timestep = (episode_step == 1000-1)
				for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
					data_[k].append(dataset[k][i])
				if done_bool or final_timestep:
					episode_step = 0
					episode_data = {}
					for k in data_:
						episode_data[k] = np.array(data_[k])
					paths.append(episode_data)
					data_ = collections.defaultdict(list)
				episode_step += 1

			returns = np.array([np.sum(p['rewards']) for p in paths])
			num_samples = np.sum([p['rewards'].shape[0] for p in paths])
			print(f'Number of samples collected: {num_samples}')
			print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, '
				  f'max = {np.max(returns)}, min = {np.min(returns)}')

			with open(save_dir / f'{name}.pkl', 'wb') as f:
				pickle.dump(paths, f)
