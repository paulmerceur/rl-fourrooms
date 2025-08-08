import functools
import os
import sys
import torch
import argparse
import configparser
import ast

import pufferlib
import pufferlib.pufferl as pufferl
from pufferlib.models import Default, LSTMWrapper

# Ensure the project root is on sys.path so `import four_rooms` resolves to this folder as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from four_rooms import FourRooms


def main():
    parser = argparse.ArgumentParser(description='Train FourRooms with PufferLib config-style overrides', add_help=False)
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'config', 'four_rooms.ini'))
    # Parse and strip our local flags so pufferl.load_config doesn't see them
    cli_args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    # 1) Build vec env from local creator
    env_creator = functools.partial(FourRooms, num_envs=4096, size=19)
    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=2,
        num_workers=2,
        batch_size=1,
        backend=pufferlib.vector.Multiprocessing,
        env_kwargs={'num_envs': 4096, 'size': 19},
    )

    # 2) Policy = Default + LSTMWrapper
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_policy = Default(vecenv.driver_env, hidden_size=128)
    lstm_policy = LSTMWrapper(vecenv.driver_env, base_policy, input_size=128, hidden_size=128).to(device)

    class StateInitPolicy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            # Surface attrs the trainer expects
            self.hidden_size = getattr(inner, 'hidden_size', 128)
            self.is_continuous = getattr(inner, 'is_continuous', False)

        def forward_eval(self, observations, state=None):
            if state is None:
                state = {}
            state.setdefault('lstm_h', None)
            state.setdefault('lstm_c', None)
            return self.inner.forward_eval(observations, state)

        def forward(self, observations, state=None):
            return self.forward_eval(observations, state)

    policy = StateInitPolicy(lstm_policy)

    # 3) Load baseline defaults from library, then override from local ini using the same typing rules
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

    args['train']['device'] = device

    # 4) Train
    # Config values now come from default + four_rooms.ini; you can edit the ini to tweak params
    pufferl.train(env_name='four_rooms', args=args, vecenv=vecenv, policy=policy)


if __name__ == '__main__':
    main()


