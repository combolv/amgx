import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ConfigOptEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # ---- Variable definitions ----
        self.int_sizes = [15, 4, 9, 11, 7, 13, 6, 10, 8]  # 9 discrete
        self.num_int = len(self.int_sizes)
        self.num_float = 2
        self.total_dim = self.num_int + self.num_float  # 11

        # ---- Action space (depends on step) ----
        self.max_discrete = max(self.int_sizes)

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_discrete,),  # enough to cover largest discrete
            dtype=np.float32
        )

        # ---- Observation space ----
        self.observation_space = spaces.Dict({
            "mask": spaces.MultiBinary(self.total_dim), 
            "current": spaces.MultiBinary(self.total_dim), 
        })

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        # store chosen values
        self.int_vars = [None] * self.num_int
        self.float_vars = [None] * self.num_float

        return self._get_obs(), {}

    def _get_obs(self):
        mask = np.zeros(self.total_dim, dtype=np.int8)
        mask[:self.current_step] = 1

        current = np.zeros(self.total_dim, dtype=np.int8)
        if self.current_step < self.total_dim:
            current[self.current_step] = 1

        return {
            "mask": mask,
            "current": current
        }

    def step(self, action):
        terminated = False
        truncated = False

        idx = self.current_step

        # ---- Decode action ----
        if idx < self.num_int:
            # discrete variable
            size = self.int_sizes[idx]

            # take first "size" entries and argmax
            val = int(np.argmax(action[:size]))
            self.int_vars[idx] = val

        else:
            # float variable
            float_idx = idx - self.num_int
            val = float(action[0])  # use first dimension
            self.float_vars[float_idx] = np.clip(val, 0.0, 1.0)

        self.current_step += 1

        # ---- If all variables selected → evaluate ----
        if self.current_step == self.total_dim:
            value = self.F(self.int_vars, self.float_vars)
            reward = -value  # minimize
            terminated = True
        else:
            reward = 0.0  # sparse reward

        return self._get_obs(), reward, terminated, truncated, {}

    def F(self, int_vars, float_vars):
        # Replace with your real expensive function
        return np.sum(int_vars) + np.sum(float_vars)