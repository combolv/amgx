from config_env_make import collect_mat_dataset, make_env
from config_env_train import mlp_ppo_train, mlp_sac_train

if __name__ == "__main__":
    A_sp_list, b_list, config_dict = collect_mat_dataset()
    print(f"Collected {len(A_sp_list)} matrices and {len(b_list)} RHS vectors.")
    print(f"Sample matrix shape: {A_sp_list[0].shape}, sample RHS shape: {b_list[0].shape}")
    print(f"Sample config keys: {list(config_dict.keys())}")
    gym_env = make_env("config_env_one_step/ConfigOptSimple-v0", A_sp_list, b_list, config_dict)
    print("Environment created successfully.")
    obs = gym_env.reset()
   
    print(f"Initial observation: {obs}")
    action = gym_env.action_space.sample()
    print(f"Sample action: {action}")
    obs, reward, terminated, info = gym_env.step([action])
    print(f"Observation after step: {obs}")
    print(f"Reward: {reward}, Terminated: {terminated}, Info: {info}")
    input("Check the PID now?")  # Pause to allow checking the process status before training starts
    sac_training = mlp_ppo_train(gym_env, "sac_test_output", device="cpu", iter=10000)
