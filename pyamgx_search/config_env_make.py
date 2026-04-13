import scipy.sparse as sp
import numpy as np
import json
from pathlib import Path
import gymnasium
import config_env_one_step
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

def collect_mat_dataset(folder_prefix="/home/combo/env/ncg/generated/poisson_tetmesh/", config_path="/home/combo/env/amgx/test_mat/test.json"):
    all_mat = list(Path(folder_prefix + "mat").glob("*.npz"))
    all_rhs = list(Path(folder_prefix + "rhs").glob("*.npy"))
    all_mat.sort()
    all_rhs.sort()
    A_sp_list = [sp.load_npz(str(mat_path)) for mat_path in all_mat]
    b_list = [np.load(str(rhs_path)) for rhs_path in all_rhs]
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return A_sp_list, b_list, config_dict

def make_env(env_name, A_sp_list, b_list, config_dict):
    # env = gymnasium.make(env_name, A_sp_list=A_sp_list, b_list=b_list, template_config=config_dict)
    vec_env = make_vec_env(env_name, n_envs=1, vec_env_cls=DummyVecEnv,
                           env_kwargs={"A_sp_list": A_sp_list, "b_list": b_list, "template_config": config_dict})
    return vec_env


if __name__ == "__main__":
    A_sp_list, b_list, config_dict = collect_mat_dataset()
    print(f"Collected {len(A_sp_list)} matrices and {len(b_list)} RHS vectors.")
    print(f"Sample matrix shape: {A_sp_list[0].shape}, sample RHS shape: {b_list[0].shape}")
    print(f"Sample config keys: {list(config_dict.keys())}")
    gym_env = make_env("config_env_one_step/ConfigOptSimple-v0", A_sp_list, b_list, config_dict)
    print("Environment created successfully.")
    obs, info = gym_env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    action = gym_env.action_space.sample()
    print(f"Sample action: {action}")
    obs, reward, terminated, truncated, info = gym_env.step([action])
    print(f"Observation after step: {obs}")
    print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")