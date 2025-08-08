FourRooms: Native PufferLib Environment (MiniGrid-style)
# Four Rooms (PufferLib Native)

High-performance native PufferLib environment replicating MiniGrid FourRooms (variable size). Implemented in C and exposed via a tiny Python extension. Trains at millions of SPS using PufferLib’s vectorization.

## Highlights
- Native Puffer environment with zero-copy vectorization
- 1M+ SPS on typical workstations
- Same task as MiniGrid FourRooms; variable `size`
- Clean, standalone package

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install pufferlib gymnasium numpy torch
cd my_four_rooms
python setup.py build_ext --inplace
```

## Train
```bash
cd my_four_rooms
python train.py --config my_four_rooms/config/four_rooms.ini
```

Config follows PufferLib’s `.ini` format. Edit `my_four_rooms/config/four_rooms.ini` to change vectorization, env, and training params.
Key fields:
- [vec]: `num_workers`, `num_envs`
- [env]: `num_envs`, `size`
- [train]: `total_timesteps`, `learning_rate`, `minibatch_size`, …

## Structure
```
my_four_rooms/
  four_rooms.py
  four_rooms.h
  four_rooms.c
  binding.c
  setup.py
  train.py
  config/
    four_rooms.ini
  resources/
  README.md
  .gitignore
```

## Notes
- Efficiency: PufferLib native buffers + multiprocessing achieve multi‑million SPS.
- Task: Mirrors MiniGrid FourRooms; obs is a 7×7×3 egocentric encoding; actions are 7 discrete (turn/forward used).
- Rendering: Headless build by default (`-DNO_RAYLIB=1`). Install Raylib and remove the macro to enable on‑screen render.

## License
MIT

FourRooms is a fast, native C + Python RL environment implemented with PufferLib’s minimal environment API. It’s a drop-in clone of the well-known MiniGrid FourRooms task (variable grid size, partial observability), but runs at millions of steps per second on a single machine thanks to PufferLib’s zero-copy vectorization and tight C loop.

Highlights
- Native Puffer env (no Gym step loop overhead). Millions of SPS on commodity hardware.
- Observation: 7x7x3 MiniGrid-style encoding (object, color, state) in front of the agent.
- Action space: 7 discrete actions; only Left/Right/Forward used for navigation.
- Variable size (default 19). Automatic episode timeout at size*size steps.

Install (editable)
  python -m venv .venv && source .venv/bin/activate
  pip install -U pip setuptools wheel numpy gymnasium torch
  python setup.py build_ext --inplace

Train
  python train.py --config config/four_rooms.ini

The training uses PufferLib’s Default policy wrapped by LSTM, and the same config parsing as the library’s .ini files. You can tweak anything in config/four_rooms.ini (e.g., vec.num_workers, env.size, train.learning_rate, etc.).

Config format
[base]
env_name = four_rooms

[vec]
num_workers = 12
num_envs = 12

[env]
num_envs = 256
size = 11

[train]
total_timesteps = 10_000_000
learning_rate = 0.015
minibatch_size = 32768
gamma = 0.99

Project structure
four_rooms/
  __init__.py            # exposes FourRooms
  env.py                 # Python side of the env (Puffer API)
  binding.c              # C-Python bridge (vectorized init/step/reset)

config/
  four_rooms.ini         # editable training config

train.py                 # runs PPO training with Default + LSTMWrapper
setup.py                 # builds the C extension (headless by default)

Notes
- Rendering via Raylib is disabled by default in the C build (-DNO_RAYLIB=1). You can enable it by removing that macro and linking raylib if you want on-screen visualization.
- The environment is a copy of MiniGrid’s FourRooms in spirit and interface (observation/action), not a code fork.
- Thanks to PufferLib, this runs with vectorized stepping in C and zero-copy buffers, which is why SPS is so high.

