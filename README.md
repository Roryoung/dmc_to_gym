# OpenAI Gym wrapper for the DeepMind Control Suite.
A lightweight wrapper around the DeepMind Control Suite that provides the standard OpenAI Gym interface.
### Instalation

```
pip3 install git+git://github.com/Roryoung/dmc_to_gym.git
```

### Usage
```python
from dm_control import suite
import dmc_to_gym

dmc_env = suite.load('cartpole', 'swingup')

env = dmc_to_gym(dmc_env)

done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```
