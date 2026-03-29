import pyamgx
import scipy.sparse as sp
import numpy as np

def solve_with_A_sp_b_and_config(A_sp, b, config_dict, return_residuals=False):
    logs = []

    def amgx_logger(msg):
        logs.append(msg)

    def get_err_return(error_code, message):
        return -1, 1, 1, [], logs, error_code, message

    pyamgx.register_print_callback(amgx_logger)
    pyamgx.initialize()
    cfg = pyamgx.Config()
    cfg.create_from_dict(config_dict)
    rsc = pyamgx.Resources().create_simple(cfg)
    A = pyamgx.Matrix().create(rsc)
    b_amgx = pyamgx.Vector().create(rsc)
    x = pyamgx.Vector().create(rsc)
    solver = pyamgx.Solver().create(rsc, cfg)
    A.upload_CSR(A_sp)
    b_amgx.upload(b)
    sol = np.zeros_like(b)
    x.upload(sol)
    solver.setup(A)
    solver.solve(b_amgx, x)
    final_sol = x.download()
    if return_residuals:
        num_iters = solver.iterations_number
        residuals = [solver.get_residual(i) for i in range(num_iters + 1)]
    else:
        num_iters = -1
        residuals = []
    A.destroy()
    x.destroy()
    b_amgx.destroy()
    solver.destroy()
    rsc.destroy()
    cfg.destroy()
    pyamgx.finalize()

    # Check the result
    if return_residuals:
        try:
            Ax_b = (A_sp @ final_sol) - b.ravel()
            residual_norm = np.linalg.norm(Ax_b)
            init_norm = np.linalg.norm(b)
            rel_residual_norm = residual_norm / init_norm if init_norm > 0 else float('inf')
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            return get_err_return(1, f"Error calculating residuals: {e}")
        
        if np.isnan(residual_norm) or np.isinf(residual_norm):
            return get_err_return(2, f"Invalid residual norm: {residual_norm}")
        
        if rel_residual_norm > 1e-4:
            return get_err_return(3, f"High relative residual norm: {rel_residual_norm:3g}")
    else:
        residual_norm = None
        rel_residual_norm = None
    
    return num_iters, residual_norm, rel_residual_norm, residuals, logs, 0, "Success"

if __name__ == "__main__":
    import json
    import time
    A_sp = sp.load_npz("/home/combo/env/ncg/generated/poisson_tetmesh/mat/000001.npz")
    b = np.load("/home/combo/env/ncg/generated/poisson_tetmesh/rhs/000001.npy")
    with open("/home/combo/env/amgx/test_mat/test.json", "r") as f:
        config_dict = json.load(f)
    start_time = time.time()
    num_iters, residual_norm, rel_residual_norm, residual_list, logs, err_code, err_msg = solve_with_A_sp_b_and_config(A_sp, b, config_dict)
    end_time = time.time()
    print(f"Number of iterations: {num_iters}")
    print(f"Final residual norm: {residual_norm}")
    print(f"Relative residual norm: {rel_residual_norm}")
    print(f"Execution time: {end_time - start_time}")
    print("AMGX logs:")
    for log in logs:
        print(log)