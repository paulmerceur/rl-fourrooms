import os
import sys
import glob
import argparse
import configparser

import torch

import pufferlib
import pufferlib.pufferl as pufferl
from pufferlib.models import Default, LSTMWrapper


# Ensure the project root is on sys.path so `import four_rooms` resolves to this folder as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from four_rooms import FourRooms


def main():
    parser = argparse.ArgumentParser(description='Evaluate FourRooms policy with PufferLib renderer')
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'config', 'four_rooms.ini'))
    parser.add_argument('--load', type=str, default=None, help='Path to a .pt checkpoint (or "latest" to pick newest in experiments/)')
    parser.add_argument('--render-mode', type=str, default='raylib', choices=['raylib', 'ansi', 'rgb_array'])
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='eval.gif')
    args_cli = parser.parse_args()

    # Load env params from local ini (e.g., size)
    size = 19
    if os.path.exists(args_cli.config):
        cfg = configparser.ConfigParser()
        cfg.read(args_cli.config)
        if cfg.has_section('env') and cfg.has_option('env', 'size'):
            try:
                size = int(cfg['env']['size'])
            except Exception:
                pass

    # 1) Vec env: Serial, single env for rendering
    env_creator = FourRooms
    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=1,
        batch_size=1,
        backend=pufferlib.vector.Serial,
        env_kwargs={'num_envs': 1, 'size': size, 'render_mode': args_cli.render_mode},
    )

    # 2) Build policy identical to train setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_policy = Default(vecenv.driver_env, hidden_size=128)
    lstm_policy = LSTMWrapper(vecenv.driver_env, base_policy, input_size=128, hidden_size=128).to(device)

    class StateInitPolicy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
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

    # Optional: load checkpoint weights
    if args_cli.load is not None:
        load_path = args_cli.load
        if load_path == 'latest':
            try:
                load_path = max(glob.glob('experiments/*.pt'), key=os.path.getctime)
            except ValueError:
                raise FileNotFoundError('No checkpoints found in experiments/*.pt')
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict, strict=False)

    # 3) Build config expected by pufferlib.eval
    pl_args = pufferl.load_config('default')
    pl_args['train']['device'] = device
    pl_args['train']['use_rnn'] = True
    pl_args['save_frames'] = int(args_cli.save_frames)
    pl_args['gif_path'] = args_cli.gif_path
    pl_args['fps'] = float(args_cli.fps)

    # 4) Run library eval loop (renders via env.render())
    pufferl.eval(env_name='four_rooms', args=pl_args, vecenv=vecenv, policy=policy)


if __name__ == '__main__':
    main()


