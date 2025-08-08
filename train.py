import functools
import os
import sys
import torch
import argparse
import configparser
import ast

import pufferlib
import pufferlib.pufferl as pufferl

# Ensure the project root is on sys.path so local packages import
sys.path.insert(0, os.path.dirname(__file__))
from policy import Policy

# Already inserted project root above for local imports
from four_rooms import FourRooms


def main():
    parser = argparse.ArgumentParser(description='Train FourRooms with PufferLib config-style overrides', add_help=False)
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'config', 'four_rooms.ini'))
    # Parse and strip our local flags so pufferl.load_config doesn't see them
    cli_args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    # 1) Load baseline defaults from library, then override from local ini using the same typing rules
    args = pufferl.load_config('default')

    def puffer_type(value):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value

    cfg = configparser.ConfigParser()
    cfg.read(cli_args.config)
    for section in cfg.sections():
        if section not in args:
            args[section] = {}
        for key, val in cfg[section].items():
            args[section][key] = puffer_type(val)

    # 2) Resolve device (honor config; support CUDA/MPS/CPU)
    def resolve_device(pref):
        if isinstance(pref, str):
            pref = pref.lower()
        if pref in {'cuda', 'gpu', 'cuda:0'}:
            return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        if pref == 'mps':
            return 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        if pref == 'cpu':
            return 'cpu'
        # auto: prefer CUDA, then MPS, then CPU
        return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    device_pref = args.get('train', {}).get('device', None)
    device = resolve_device(device_pref)
    args.setdefault('train', {})['device'] = device

    # 3) Build vec env from config
    vec_cfg = args.get('vec', {})
    env_cfg = args.get('env', {})

    num_envs = int(vec_cfg.get('num_envs', 1))
    num_workers = vec_cfg.get('num_workers', None)
    # Allow 'auto' to pass through; only cast to int if an explicit number
    if isinstance(num_workers, str) and num_workers.lower() == 'auto':
        pass
    elif num_workers is not None:
        num_workers = int(num_workers)

    batch_size = vec_cfg.get('batch_size', 1)
    if batch_size is None:
        batch_size = 1
    # Allow 'auto' batch size to pass through for pufferlib to resolve
    if isinstance(batch_size, str) and batch_size.lower() == 'auto':
        pass
    else:
        batch_size = int(batch_size)
    overwork = bool(vec_cfg.get('overwork', True))

    env_kwargs = {
        'num_envs': int(env_cfg.get('num_envs', 1)),
        'size': int(env_cfg.get('size', 19)),
    }

    vecenv = pufferlib.vector.make(
        FourRooms,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=batch_size,
        backend=pufferlib.vector.Multiprocessing,
        overwork=overwork,
        env_kwargs=env_kwargs,
    )

    # 4) Policy
    policy = Policy(
        vecenv.driver_env,
        hidden_size=128,
        lstm_hidden_size=128,
        device=device,
    )

    # 5) Train
    pufferl.train(env_name='four_rooms', args=args, vecenv=vecenv, policy=policy)


if __name__ == '__main__':
    main()


