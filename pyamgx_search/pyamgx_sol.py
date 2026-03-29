import pyamgx
import scipy.sparse as sp
import numpy as np
import json

from multiprocessing import Process, Queue
from pyamgx_util import solve_with_A_sp_b_and_config

def wrapped_solver(A_sp, b, config_dict, return_residuals=False, timeout=1):
    def get_err_return(error_code, message):
        return -1, 1, 1, [], [], error_code, message

    def _worker(A_sp, b, config_dict, queue):
        try:
            result = solve_with_A_sp_b_and_config(A_sp, b, config_dict, return_residuals=return_residuals)
            queue.put(("success", result))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            queue.put(("error", str(e)))

    queue = Queue()
    p = Process(target=_worker, args=(A_sp, b, config_dict, queue))

    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()   # force kill
        p.join()
        return get_err_return(4, "Timeout")
        

    if not queue.empty():
        result = queue.get()
        if result[0] == "error":
            return get_err_return(5, f"Error in solver process: {result[1]}")
        elif result[0] != "success":
            return get_err_return(6, f"Unexpected result from solver process: {result[0]}")
    else:
        return get_err_return(7, "No result from solver process")
    
    return result[1]

def robust_wrapped_solver(A_sp, b, config_dict, warmup_iter=1, repeat=5, refine=False, verbose=False):
    ret_dict = {
        "num_iters": None,
        "final_residual_norm": None,
        "final_rel_residual_norm": None,
        "residual_list": None,
        "setup_time_list": None,
        "all_time_list": None,
        "err_code": None,
        "err_msg": None
    }

    def refine_default_config(input_config):
        new_config = input_config.copy()
        new_config["solver"]["store_res_history"] = 1
        new_config["solver"]["print_solve_stats"] = 1
        new_config["solver"]["max_iters"] = 1000
        return new_config
    
    if refine:
        config_dict = refine_default_config(config_dict)

    def parse_solving_time(logs):
        setup_time = None
        all_time = None
        for log in logs:
            if "setup" in log:
                parts = log.split()
                for i, part in enumerate(parts):
                    if part == "setup:" and i + 1 < len(parts):
                        try:
                            setup_time = float(parts[i + 1].rstrip("s"))
                        except ValueError:
                            continue
            if "Total" in log:
                parts = log.split()
                for i, part in enumerate(parts):
                    if part == "Time:" and i + 1 < len(parts):
                        try:
                            all_time = float(parts[i + 1].rstrip("s"))
                        except ValueError:
                            continue
        return setup_time, all_time
    
    for _ in range(warmup_iter):
        res = wrapped_solver(A_sp, b, config_dict, return_residuals=True, timeout=2)
        if res[-2] != 0:
            if verbose:
                print(f"Warmup iteration failed with error code {res[-2]}: {res[-1]}")
            ret_dict["num_iters"] = res[0]
            ret_dict["final_residual_norm"] = res[1]
            ret_dict["final_rel_residual_norm"] = res[2]
            ret_dict["residual_list"] = res[3]
            ret_dict["err_code"] = res[5]
            ret_dict["err_msg"] = res[6]
            return ret_dict
        if verbose:
            print(f"Warmup iteration completed: num_iters={res[0]}, final_residual_norm={res[1]}, final_rel_residual_norm={res[2]}")
        ret_dict["num_iters"] = res[0]
        ret_dict["final_residual_norm"] = res[1]
        ret_dict["final_rel_residual_norm"] = res[2]
        ret_dict["residual_list"] = res[3]
    
    def generate_residual_unmonitoring_config(input_config):
        new_config = input_config.copy()
        new_config["solver"]["store_res_history"] = 0
        new_config["solver"]["print_solve_stats"] = 0
        return new_config
    
    config_no_res = generate_residual_unmonitoring_config(config_dict)
    setup_times = []
    all_times = []
    for i in range(repeat):
        res = wrapped_solver(A_sp, b, config_no_res, return_residuals=False, timeout=1)
        if res[-2] != 0:
            if verbose:
                print(f"Iteration {i} failed with error code {res[-2]}: {res[-1]}")
            return res
        logs = res[4]
        try:
            setup_time, all_time = parse_solving_time(logs)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            if verbose:
                print(f"Error parsing solving time: {e}")
            ret_dict["err_code"] = 8
            ret_dict["err_msg"] = f"Error parsing solving time: {e}"
            return ret_dict
            
        if setup_time is not None:
            setup_times.append(setup_time)
        else:
            if verbose:
                print(f"Warning: setup time not found in logs for iteration {i}")
            ret_dict["err_code"] = 9
            ret_dict["err_msg"] = f"Setup time not found in logs for iteration {i}"
            return ret_dict

        if all_time is not None:
            all_times.append(all_time)
        else:
            if verbose:
                print(f"Warning: total solving time not found in logs for iteration {i}")
            ret_dict["err_code"] = 10
            ret_dict["err_msg"] = f"Total solving time not found in logs for iteration {i}"
            return ret_dict
    
    ret_dict["setup_time_list"] = setup_times
    ret_dict["all_time_list"] = all_times
    ret_dict["err_code"] = 0
    ret_dict["err_msg"] = "Success"
    
    return ret_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, default="/home/combo/env/ncg/generated/poisson_tetmesh/mat/000000.npz")
    parser.add_argument("--rhs", type=str, default="/home/combo/env/ncg/generated/poisson_tetmesh/rhs/000000.npy")
    parser.add_argument("--config", type=str, default="/home/combo/env/amgx/test_mat/test.json")
    args = parser.parse_args()
    A_sp = sp.load_npz(args.mat)
    b = np.load(args.rhs)
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    num_iters, residual_norm, rel_residual_norm, residual_list, logs, err_code, err_msg = wrapped_solver(A_sp, b, config_dict)
    print(f"Number of iterations: {num_iters}")
    print(f"Final residual norm: {residual_norm}")
    print(f"Relative residual norm: {rel_residual_norm}")
    print(f"Error code: {err_code}, message: {err_msg}")

    ret_dict = robust_wrapped_solver(A_sp, b, config_dict)
    print(ret_dict)
