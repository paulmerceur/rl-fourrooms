import gymnasium
import numpy as np

import pufferlib
try:
    from . import binding  # when imported as a package module
except ImportError:
    import binding  # when run from this folder as a script/cwd module


class FourRooms(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=19, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=10,
            shape=(7*7*3,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(7)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, size=size)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)




