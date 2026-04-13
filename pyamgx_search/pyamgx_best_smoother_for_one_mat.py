import optuna
from pyamgx_sol import robust_wrapped_solver
from pyamgx_config_modifier import generate_all
import scipy.sparse as sp
import numpy as np
import argparse
import json
from pathlib import Path

if __name__ != "__main__":
    raise ImportError("This script is meant to be run as a standalone program, not imported as a module.")

robust_best_value_global = [1.1]  # Use a list to allow modification inside callback

parser = argparse.ArgumentParser()
parser.add_argument("--mat", type=str, default="/home/combo/env/ncg/generated/poisson_tetmesh/mat/000000.npz")
parser.add_argument("--rhs", type=str, default="/home/combo/env/ncg/generated/poisson_tetmesh/rhs/000000.npy")
parser.add_argument("--config", type=str, default="/home/combo/env/amgx/test_mat/test.json")
parser.add_argument("--output-dir", type=str, default="/home/combo/env/amgx/pyamgx_search/output")
parser.add_argument("--n-trials", type=int, default=10000)
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

A_sp = sp.load_npz(args.mat)
b = np.load(args.rhs)
with open(args.config, "r") as f:
    template_config = json.load(f)

def F(int_vars, float_vars):
    config_dict = generate_all(template_config, int_vars, float_vars)
    res = robust_wrapped_solver(A_sp, b, config_dict, repeat=1)
    if res["err_code"] != 0:
        return 1.0
    ret = np.mean(res["all_time_list"])
    if not 0 < ret < 1:
        return 1.0
    return ret

def objective(trial):
    int_vars = [
        trial.suggest_int("selector_int", 0, 6),
        trial.suggest_int("smoother_int", 0, 12),
        trial.suggest_int("coarsest_int", 0, 4),
        trial.suggest_int("amg_sweep_int", 0, 5),
        trial.suggest_int("amg_min_coarse_rows_int", 0, 3),
        trial.suggest_int("amg_cycle_int", 0, 2),
        trial.suggest_int("amg_interp_max_elements_int", 0, 3)
    ]
    float_vars = [
        trial.suggest_float("amg_strength_threshold_float", 0.0, 1.0),
        trial.suggest_float("smoother_relaxation_factor_float", 0.0, 1.0)
    ]
    return F(int_vars, float_vars)

def callback_on_new_best(study, trial):
    if study.best_trial.number == trial.number:
        if study.best_value < 0.9:
            print(f"New best value: {study.best_value:.6f} with params: {study.best_params}")
            # Save the best config to a JSON file
            best_config = generate_all(template_config, 
                                        [trial.params["selector_int"], trial.params["smoother_int"], trial.params["coarsest_int"], trial.params["amg_sweep_int"], trial.params["amg_min_coarse_rows_int"], trial.params["amg_cycle_int"], trial.params["amg_interp_max_elements_int"]],
                                        [trial.params["amg_strength_threshold_float"], trial.params["smoother_relaxation_factor_float"]])
            # Solve the new best more robustly and print the result
            res = robust_wrapped_solver(A_sp, b, best_config, repeat=5)
            # Save the best config and result to a JSON file
            output_dict = {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_config": best_config,
                "result": res
            }
            output_path = Path(args.output_dir) / f"best_config_{trial.number}.json"
            with open(output_path, "w") as f:
                json.dump(output_dict, f, indent=4)
            robust_time_mean = np.mean(res["all_time_list"])
            if robust_time_mean < robust_best_value_global[0]:
                robust_best_value_global[0] = robust_time_mean
                output_dict["robust_best_value"] = robust_time_mean
                output_dict["robust_opt_trial_number"] = trial.number
                best_output_path = Path(args.output_dir) / "best_config_overall.json"
                with open(best_output_path, "w") as f:
                    json.dump(output_dict, f, indent=4)


sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=args.n_trials, callbacks=[callback_on_new_best])
# Save the optimization history to a JSON file
history = []
for trial in study.trials:
    history.append({
        "trial_number": trial.number,
        "value": trial.value,
        "params": trial.params,
        "state": trial.state.name
    })
history_path = Path(args.output_dir) / "optimization_history.json"
with open(history_path, "w") as f:
    json.dump(history, f, indent=4)