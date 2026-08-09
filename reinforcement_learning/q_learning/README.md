# Q-Learning

This project implements Q-learning using the FrozenLake environment
from `gymnasium`.

## Files

- `0-load_env.py`: Loads the pre-made FrozenLakeEnv environment.

## Requirements

- gymnasium
- numpy

## Usage

```python
load_frozen_lake = __import__('0-load_env').load_frozen_lake

env = load_frozen_lake()
print(env.unwrapped.desc)
```
