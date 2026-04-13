import random
from copy import deepcopy
def generate_selector(input_config_dict, input_size_int):
    '''
    input_size_int: -3, -2, -1, 0, 2, 4, 8
    When <= 0, use the classical AMG with aggressive coarsening = -input_size_int
    When > 0, use the aggregation AMG with size of aggregates = input_size_int
    '''
    config_dict = deepcopy(input_config_dict)
    if "selector" in config_dict["solver"]["preconditioner"]:
        config_dict["solver"]["preconditioner"].pop("selector")
    if input_size_int <= 0:
        config_dict["solver"]["preconditioner"]["algorithm"] = "CLASSICAL"
        config_dict["solver"]["preconditioner"]["aggressive_levels"] = -input_size_int
    else:
        config_dict["solver"]["preconditioner"]["algorithm"] = "AGGREGATION"
        config_dict["solver"]["preconditioner"]["selector"] = "SIZE_" + str(input_size_int)
    return config_dict

def generate_smoother_partial(input_config_dict, smoother_type_id, input_factor):
    '''
    smoother_type_id: -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4
    0 -> BLOCK_JACOBI
    1 -> JACOBI_L1
    2 -> CHEBYSHEV (poly order 2, lambda estimate mode 2)
    3 -> CHEBYSHEV (poly order 3, lambda estimate mode 2)
    4 -> CHEBYSHEV (poly order 4, lambda estimate mode 2)
    -1 -> MULTICOLOR_GS ("PARALLEL_GREEDY")
    -2 -> MULTICOLOR_GS ("MIN_MAX")
    -3 -> MULTICOLOR_GS ("ROUND_ROBIN")
    -4 -> MULTICOLOR_GS ("MULTI_HASH")
    -5 -> MULTICOLOR_DILU ("PARALLEL_GREEDY")
    -6 -> MULTICOLOR_DILU ("MIN_MAX")
    -7 -> MULTICOLOR_DILU ("ROUND_ROBIN")
    -8 -> MULTICOLOR_DILU ("MULTI_HASH")

    input_factor:
    For Jacobi and Chebyshev: relaxation factor, suggested search [0.33, 0.5, 0.66, 0.72, 0.8, 0.91, 1.0]
    For multi-color GS and DILU: max uncolored percentage, suggested search [0.05, 0.1, 0.15, 0.2]
    '''
    config_dict = deepcopy(input_config_dict)
    if "preconditioner" in config_dict["solver"]["preconditioner"]["smoother"]:
        config_dict["solver"]["preconditioner"]["smoother"].pop("preconditioner")
    if smoother_type_id in [0, 1]:
        config_dict["solver"]["preconditioner"]["smoother"]["solver"] = "BLOCK_JACOBI" if smoother_type_id == 0 else "JACOBI_L1"
        config_dict["solver"]["preconditioner"]["smoother"]["relaxation_factor"] = input_factor
    elif smoother_type_id in [2, 3, 4]:
        config_dict["solver"]["preconditioner"]["smoother"]["solver"] = "CHEBYSHEV"
        config_dict["solver"]["preconditioner"]["smoother"]["relaxation_factor"] = input_factor
        config_dict["solver"]["preconditioner"]["smoother"]["chebyshev_polynomial_order"] = smoother_type_id
        config_dict["solver"]["preconditioner"]["smoother"]["chebyshev_lambda_estimate_mode"] = 2
        config_dict["solver"]["preconditioner"]["smoother"]["preconditioner"] = {
            "solver": "JACOBI_L1",
            "max_iters": 1,
            "scope": "chebyshev_inner"
        }
    elif smoother_type_id in [-1, -2, -3, -4, -5, -6, -7, -8]:
        use_dilu = smoother_type_id in [-5, -6, -7, -8]
        agg_strategy = {
            -1: "PARALLEL_GREEDY", -5: "PARALLEL_GREEDY",
            -2: "MIN_MAX", -6: "MIN_MAX",
            -3: "ROUND_ROBIN", -7: "ROUND_ROBIN",
            -4: "MULTI_HASH", -8: "MULTI_HASH"
        }
        config_dict["solver"]["preconditioner"]["smoother"]["solver"] = "MULTICOLOR_DILU" if use_dilu else "MULTICOLOR_GS"
        config_dict["solver"]["preconditioner"]["max_uncolored_percentage"] = input_factor
        config_dict["solver"]["preconditioner"]["matrix_coloring_scheme"] = agg_strategy[smoother_type_id]
    else:
        raise ValueError(f"Invalid smoother_type_id: {smoother_type_id}")
    return config_dict

def generate_coarsest(input_config_dict, coarsest_max_iter):
    '''
    coarsest_max_iter: 0, 2, 4, 8, 16
    0: Use a direct solver on the coarsest level
    >0: Set the coarsest solver sweeps to the input value and use the NOSOLVER (iterative) on the coarsest level
    '''
    config_dict = deepcopy(input_config_dict)
    if coarsest_max_iter == 0:
        config_dict["solver"]["preconditioner"]["coarse_solver"] = "DENSE_LU_SOLVER"
    else:
        config_dict["solver"]["preconditioner"]["coarse_solver"] = "NOSOLVER"
        config_dict["solver"]["preconditioner"]["coarsest_sweeps"] = coarsest_max_iter
    return config_dict

def generate_amg_partial(input_config_dict, sweep_iter, min_coarse_rows, cycle_type_id, interp_max_elements, strength_threshold):
    config_dict = deepcopy(input_config_dict)
    config_dict["solver"]["preconditioner"]["presweeps"] = sweep_iter
    config_dict["solver"]["preconditioner"]["postsweeps"] = sweep_iter
    config_dict["solver"]["preconditioner"]["min_coarse_rows"] = min_coarse_rows
    config_dict["solver"]["preconditioner"]["cycle"] = "VWF"[cycle_type_id]
    config_dict["solver"]["preconditioner"]["interp_max_elements"] = interp_max_elements
    config_dict["solver"]["preconditioner"]["strength_threshold"] = strength_threshold
    return config_dict


def generate_smoother(input_config_dict, smoother_hyperclass, smoother_type_sub_int, input_factors):
    '''
    smoother_hyperclass: 0, 1, 2
    0 -> Jacobi;
    1 -> Chebyshev;
    2 -> Multi-color;

    Jacobi:
    - smoother_type_sub_int: 0, 1
        0 -> JACOBI_L1
        1 -> BLOCK_JACOBI
    - input_factors: relaxation factor, suggested search [0.33, 0.5, 0.66, 0.72, 0.8, 0.91, 1.0]

    Chebyshev:
    - smoother_type_sub_int:
        polynomial order (2, 3, 4)
    - input_factors:
        relaxation factor, suggested search [0.33, 0.5, 0.66, 0.72, 0.8, 0.91, 1.0]
        min lambda:
            < 1e-5: return to mode 2
            >= 1e-5: use the input as the min lambda
        max lambda:
            < min lambda + 1e-5: return to mode 2
            >= min lambda + 1e-5: use the input as the max lambda
        
    Multi-color:

    '''
    raise NotImplementedError("Full generator for smoothers not implemented yet")

def generate_all(input_config_dict, input_int_para, input_float_para):
    config_dict = deepcopy(input_config_dict)
    selector_int = [-3, -2, -1, 0, 2, 4, 8][input_int_para[0]]
    smoother_int = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4][input_int_para[1]]
    coarsest_int = [0, 2, 4, 8, 16][input_int_para[2]]
    amg_sweep_int = input_int_para[3] + 1
    amg_min_coarse_rows_int = [2, 4, 8, 16][input_int_para[4]]
    amg_cycle_int = input_int_para[5]
    amg_interp_max_elements_int = [2, 4, 6, 8][input_int_para[6]]
    amg_strength_threshold_float = input_float_para[0]
    smoother_relaxation_factor_float = input_float_para[1]
    config_dict = generate_selector(config_dict, selector_int)
    config_dict = generate_smoother_partial(config_dict, smoother_int, smoother_relaxation_factor_float)
    config_dict = generate_coarsest(config_dict, coarsest_int)
    config_dict = generate_amg_partial(config_dict, amg_sweep_int, amg_min_coarse_rows_int, amg_cycle_int, amg_interp_max_elements_int, amg_strength_threshold_float)
    return config_dict


def generate_random_test(input_config):
    input_int_para_max_ranges = [7, 13, 5, 6, 4, 3, 4]
    input_int_para = [random.randint(0, max_range - 1) for max_range in input_int_para_max_ranges]
    input_float_para = [random.uniform(0.1, 1.0) for _ in range(2)]
    return generate_all(input_config, input_int_para, input_float_para)


def get_input_para_ranges():
    input_int_para_max_ranges = [7, 13, 5, 6, 4, 3, 4]
    input_float_para_ranges = [(0.0, 1.0), (0.0, 1.0)]
    return input_int_para_max_ranges, input_float_para_ranges

if __name__ == "__main__":
    from tqdm import tqdm
    import json
    test_config_path = "/home/combo/env/amgx/test_mat/test.json"
    with open(test_config_path, "r") as f:
        input_config_dict = json.load(f)
    input_int_para = [0, 1, 0, 2, 2, 0, 2]
    input_float_para = [0.25, 0.8]
    new_config = generate_all(input_config_dict, input_int_para, input_float_para)
    print(json.dumps(new_config, indent=4))

    # Test the generated config by running the solver
    import scipy.sparse as sp
    import numpy as np
    from pyamgx_sol import wrapped_solver
    A_sp = sp.load_npz("/home/combo/env/ncg/generated/poisson_tetmesh/mat/000001.npz")
    b = np.load("/home/combo/env/ncg/generated/poisson_tetmesh/rhs/000001.npy")
    res = wrapped_solver(A_sp, b, new_config, return_residuals=True, timeout=5)
    print(res)

    # Test the random config generator
    for _ in tqdm(range(1024)):
        random_config = generate_random_test(input_config_dict)
        res = wrapped_solver(A_sp, b, random_config, return_residuals=True, timeout=1)
        print(res)
        if res[-2] != 0 and res[-2] != 4:  # Allow error code 4 (solver did not converge in time) since some random configs may be too slow
            print("Error with random config:", random_config)
            input("?")
