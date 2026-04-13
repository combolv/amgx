import stable_baselines3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold, CheckpointCallback
import gymnasium

def mlp_ppo_train(gym_env, saving_path, device, iter, gae_lambda=0.97, load_path="", lr=3e-4):
    model = PPO("MlpPolicy", gym_env, device=device, gae_lambda=gae_lambda, learning_rate=lr)

    model.set_logger(configure(saving_path, [ "csv", "json", "log", "tensorboard" ]))
    if load_path:
        model.load(load_path)
    call_backs = CallbackList([
        EvalCallback(gym_env, best_model_save_path=saving_path, log_path=saving_path, eval_freq=2,
            n_eval_episodes=2, deterministic=True, render=True, callback_after_eval=StopTrainingOnRewardThreshold(2000., 1)),
        CheckpointCallback(save_path=saving_path, save_replay_buffer=True, save_freq=2)
    ])
    model.learn(iter, callback=call_backs)
    return model

def mlp_sac_train(gym_env, saving_path, device, iter, tau=0.02, load_path="", lr=3e-4):
    model = SAC("MlpPolicy", gym_env, tau=tau, learning_rate=lr, device=device)
    model.set_logger(configure(saving_path, [ "csv", "json", "log", "tensorboard" ]))
    if load_path:
        model.load(load_path)
    call_backs = CallbackList([
        EvalCallback(gym_env, best_model_save_path=saving_path, log_path=saving_path, eval_freq=200,
            n_eval_episodes=2, deterministic=True, render=True, callback_after_eval=StopTrainingOnRewardThreshold(2000., 1)),
        CheckpointCallback(save_path=saving_path, save_replay_buffer=True, save_freq=200)
    ])
    model.learn(iter, callback=call_backs, progress_bar=True)
    return model

# def mlp_ppo_train(wrapped_env, saving_path, device, iter, gae_lambda=0.97, load_path="", lr=3e-4):
#     env = gymnasium.make("config_env_one_step/ConfigOptSimple-v0", wrapped_grady_simulator=wrapped_env, render_mode="text")
#     model = PPO("MlpPolicy", env, device=device, gae_lambda=gae_lambda, learning_rate=lr)
#     model.set_logger(configure(saving_path, [ "csv", "json", "log", "tensorboard" ]))
#     if load_path:
#         model.load(load_path)
#     call_backs = CallbackList([
#         EvalCallback(env, best_model_save_path=saving_path, log_path=saving_path, eval_freq=2,
#             n_eval_episodes=2, deterministic=True, render=True, callback_after_eval=StopTrainingOnRewardThreshold(2000., 1)),
#         CheckpointCallback(save_path=saving_path, save_replay_buffer=True, save_freq=2)
#     ])
#     model.learn(iter, callback=call_backs)
#     return model


# def mlp_sac_train(wrapped_env, saving_path, device, iter, tau=0.02, load_path="", lr=3e-4):
#     env = gymnasium.make("config_env_one_step/ConfigOptSimple-v0", wrapped_grady_simulator=wrapped_env, render_mode="text")
#     model = SAC("MlpPolicy", env, tau=tau, learning_rate=lr, device=device)
#     model.set_logger(configure(saving_path, [ "csv", "json", "log", "tensorboard" ]))
#     if load_path:
#         model.load(load_path)
#     call_backs = CallbackList([
#         EvalCallback(env, best_model_save_path=saving_path, log_path=saving_path, eval_freq=200,
#             n_eval_episodes=2, deterministic=True, render=True, callback_after_eval=StopTrainingOnRewardThreshold(2000., 1)),
#         CheckpointCallback(save_path=saving_path, save_replay_buffer=True, save_freq=200)
#     ])
#     model.learn(iter, callback=call_backs, progress_bar=True)
#     return model


def cart_pole(saving_path, device, iter):
    env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
    model = PPO("MlpPolicy", env, device=device)
    model.set_logger(configure(saving_path, [ "csv", "json", "log", "tensorboard" ]))
    call_backs = CallbackList([
        EvalCallback(env, best_model_save_path=saving_path, log_path=saving_path, eval_freq=200,
            n_eval_episodes=2, deterministic=True, render=True),
        CheckpointCallback(save_path=saving_path, save_replay_buffer=True, save_freq=200)
    ])
    model.learn(iter, callback=call_backs, progress_bar=True)
    return model


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    parser = ArgumentParser()
    parser.add_argument("-c", "--cuda", type=int, default=0)
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument("-n", "--iter", type=int, default=20000)
    parser.add_argument("-t", "--tactile", action="store_true")
    args = parser.parse_args()
    print("Warning: You are running config_env_train.py, which may not be expected.")

    exp_name = args.folder
    result_path = str(Path("output") / "CartPole" / exp_name)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    model = cart_pole(result_path, args.cuda, args.iter)