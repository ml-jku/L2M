# dmc2gym_custom
The original repository is available at: https://github.com/denisyarats/dmc2gym

The original code base always flattens the observations by default. However, this is impractical for our purposes, 
as we need to construct the full observation space. Therefore, we make `flatten_obs` optional. 

### Instalation
To install `dmc2gym_custom`, run: 
```
pip install -e .
```
from the root of this directory.

### Usage
```python
import dmc2gym_custom
env = dmc2gym_custom.make(domain_name='point_mass', task_name='easy', seed=1, flatten_obs=True)
done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```