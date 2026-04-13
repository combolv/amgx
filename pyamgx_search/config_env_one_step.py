import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyamgx_config_modifier import get_input_para_ranges, generate_all
from pyamgx_sol import robust_wrapped_solver


class ConfigOptSimpleEnv(gym.Env):
    def __init__(self, A_sp_list, b_list, template_config):
        #  Variable definitions
        int_sizes, float_ranges = get_input_para_ranges()
        self.int_sizes = int_sizes
        self.float_ranges = float_ranges
        self.num_int = len(self.int_sizes)
        self.num_float = len(self.float_ranges)
        self.total_dim = self.num_int + self.num_float

        self.float_bin = 15  # number of bins for each float variable

        self.action_space = spaces.MultiDiscrete(self.int_sizes + [2] * (self.num_float * self.float_bin))

        # # Action space: unified vector (first 9 entries for discrete, last 2 for float)
        # self.action_space = spaces.Dict({
        #     "int_vars": spaces.MultiDiscrete(self.int_sizes),
        #     "float_vars": spaces.Box(
        #         low=self.float_ranges[0][0], high=self.float_ranges[0][1], shape=(self.num_float,), dtype=np.float32
        #     )
        # })
    
        # Observation space: now report the time of all instances.
        self.observation_space = spaces.Box(low=0.0, high=1.1, shape=(len(A_sp_list),), dtype=np.float32)

        # Now the data is stored in the environment, and the objective function can access it
        self.num_Ab_pair = len(A_sp_list)
        self.A_sp_list = A_sp_list
        self.b_list = b_list
        self.template_config = template_config
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((len(self.A_sp_list),)), {}
    
    def F(self, int_vars, float_vars, idx):
        int_vars = [int(int_var) for int_var in int_vars]  # Ensure discrete variables are integers
        float_vars = [float(float_var) for float_var in float_vars]  # Ensure float variables are floats
        config_dict = generate_all(self.template_config, int_vars, float_vars)
        res = robust_wrapped_solver(self.A_sp_list[idx], self.b_list[idx], config_dict, repeat=1)
        if res["err_code"] != 0:
            return 1.0
        ret = np.mean(res["all_time_list"])
        if not 0 < ret < 1:
            return 1.0
        return ret

    def step(self, action):
        def convert_bin_list_to_float(bin_list):
            # binary: 0.01010101xxx
            value = 0.0
            for i, bit in enumerate(bin_list):
                value += bit * (0.5 ** (i + 1))
            return value
        int_vars = [int(i) for i in action[:self.num_int]]  # first part for discrete variables
        float_vars = [convert_bin_list_to_float(action[self.num_int + i * self.float_bin:self.num_int + (i + 1) * self.float_bin]) for i in range(self.num_float)]

        # For simplicity, we just use the first matrix/vector pair (idx=0) for evaluation
        raw_reward_list = []
        all_reward = 0.0
        for i in range(self.num_Ab_pair):
            value = self.F(int_vars, float_vars, idx=i)
            raw_reward_list.append(value)
            if value > 0.9:
                all_reward -= 10  # heavy penalty for invalid configs
            else:
                all_reward += 10 ** (2.0 - value) - 5  # reward for valid configs
        reward = all_reward  # total reward across all matrix/vector pairs

        terminated = True
        truncated = False

        return np.array(raw_reward_list, dtype=np.float32), reward, terminated, truncated, {}

gym.envs.register(
    id="config_env_one_step/ConfigOptSimple-v0",
    entry_point="config_env_one_step:ConfigOptSimpleEnv",
)