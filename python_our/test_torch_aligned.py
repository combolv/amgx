from read_our_vec import read_vec, read_mat, convert_scipy_coo_to_torch_sparse
import torch
import scipy as sp
import numpy as np

ms = {
    "A": [],
    "x": [],
    "xap": [], # x after postsmoothing
    "xac": [], # x after correction
    "xc": [],  # x coarse
    "r": [],   # residual to be restricted
    "bc": [],  # b coarse
    "RAP": [], # R x A x P
    "P": [],   # prolongator
    "R": [],   # restrictor
}

# Note that for A, P, R, RAP, only exists #level matrices.
path_prefix = "/home/combo/env/amgx/output_mat/"

def load_level_matrix(key, prefix, suffix, start_level=0):
    for i in range(start_level, 10):
        file_name = f"{prefix}{i}{suffix}"
        if key == "RAP":
            file_name = f"{prefix}{i}_l{i}{suffix}"
        try:
            mat = read_mat(file_name)
            ms[key].append(mat)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            continue
    if len(ms[key]) == 0:
        raise ValueError(f"No matrices loaded for key {key} with prefix {prefix} and suffix {suffix}")
    print(f"Loaded {len(ms[key])} matrices for key {key}")

def load_mat_all():
    # Load A, P, R, RAP
    A_prefix, A_suffix = path_prefix + "m_l", "_r0.mtx"
    P_prefix, P_suffix = path_prefix + "p", "_l0_r0.mtx"
    R_prefix, R_suffix = path_prefix + "r", "_l0_r0.mtx"
    RAP_prefix, RAP_suffix = path_prefix + "rap", "_r0.mtx"
    load_level_matrix("A", A_prefix, A_suffix, start_level=0)
    load_level_matrix("P", P_prefix, P_suffix, start_level=0)
    load_level_matrix("R", R_prefix, R_suffix, start_level=0)
    load_level_matrix("RAP", RAP_prefix, RAP_suffix, start_level=0)

# Now test if RAP = R * A * P
def test_RAP_all():
    R_torch = [convert_scipy_coo_to_torch_sparse(mat).coalesce() for mat in ms["R"]]
    A_torch = [convert_scipy_coo_to_torch_sparse(mat).coalesce() for mat in ms["A"]]
    P_torch = [convert_scipy_coo_to_torch_sparse(mat).coalesce() for mat in ms["P"]]
    RAP_torch = [convert_scipy_coo_to_torch_sparse(mat).coalesce() for mat in ms["RAP"]]
    for i, (R, A, P, RAP) in enumerate(zip(R_torch, A_torch, P_torch, RAP_torch)):
        RAP_computed = torch.sparse.mm(R, torch.sparse.mm(A, P)).coalesce()
        if not torch.allclose(RAP_computed.values(), RAP.values(), atol=1e-5):
            print(f"RAP test failed for level {i}")
        else:
            print(f"RAP test passed for level {i}")

load_mat_all()
# test_RAP_all()

def load_cg_iter_vector(key, prefix, suffix, start_index, num_level):
    for lvl_id in range(num_level):
        for id_count in range(start_index, 100):
            file_name = f"{prefix}{lvl_id}_c_{id_count}{suffix}"
            try:
                vec = read_vec(file_name)
                ms[key].append(vec)
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                continue
    if len(ms[key]) == 0:
        raise ValueError(f"No vectors loaded for key {key} with prefix {prefix} and suffix {suffix}")

def load_vec_all(start_index=0, num_level=10):
    x_prefix, x_suffix = path_prefix + "x_lvl_", ".txt"
    r_prefix, r_suffix = path_prefix + "r_lvl_", ".txt"
    xac_prefix, xac_suffix = path_prefix + "x_after_correction_lvl_", ".txt"
    xap_prefix, xap_suffix = path_prefix + "x_after_postsmooth_lvl_", ".txt"
    xc_prefix, xc_suffix = path_prefix + "xc_lvl_", ".txt"
    r_prefix, r_suffix = path_prefix + "r_lvl_", ".txt"
    bc_prefix, bc_suffix = path_prefix + "bc_lvl_", ".txt"
    load_cg_iter_vector("x", x_prefix, x_suffix, start_index=start_index, num_level=num_level)
    load_cg_iter_vector("r", r_prefix, r_suffix, start_index=start_index, num_level=num_level)
    load_cg_iter_vector("xac", xac_prefix, xac_suffix, start_index=start_index, num_level=num_level)
    load_cg_iter_vector("xap", xap_prefix, xap_suffix, start_index=start_index, num_level=num_level)
    load_cg_iter_vector("xc", xc_prefix, xc_suffix, start_index=start_index, num_level=num_level)
    load_cg_iter_vector("bc", bc_prefix, bc_suffix, start_index=start_index, num_level=num_level)
    print(f"Loaded vectors for keys: x({len(ms['x'])}), r({len(ms['r'])}), xac({len(ms['xac'])}), xap({len(ms['xap'])}), xc({len(ms['xc'])}), bc({len(ms['bc'])})")

def utils():
    # Write a V-cycle test for all levels.
    omega = 0.90
    # Some utils.
    def get_diag(U):
        return U.coalesce().values()[U.coalesce().indices()[0] == U.coalesce().indices()[1]]
    def get_offdiag_from_A_and_diag(A, diag_A):
        I = torch.sparse_coo_tensor(torch.arange(A.shape[0]).unsqueeze(0).repeat(2,1), torch.ones(A.shape[0], device=A.device), A.shape, device=A.device).coalesce()
        off_diag_A = A - torch.sparse_coo_tensor(I.indices(), diag_A, A.shape, device=A.device)
        return off_diag_A
    return omega, get_diag, get_offdiag_from_A_and_diag

def test_iterative_methods_level_0():
    omega, get_diag, get_offdiag_from_A_and_diag = utils()
    # 0. Prepare A and b.
    A0 = convert_scipy_coo_to_torch_sparse(ms["A"][0], device="cuda:0").coalesce()
    b = torch.ones(A0.shape[0], dtype=torch.float32, device="cuda:0")
    # 1. Initial guess x0 = 0 -> x after the Jacobi presmoothing.
    x = torch.zeros(A0.shape[1], dtype=torch.float32, device="cuda:0")
    diag_A = get_diag(A0)
    off_diag_A = get_offdiag_from_A_and_diag(A0, diag_A)
    x = omega * b / diag_A
    x_ref = torch.tensor(ms["x"][0], dtype=torch.float32, device="cuda:0")
    if not torch.allclose(x, x_ref, atol=1e-5):
        print("Jacobi presmoothing test failed at level 0")
    else:
        print("Jacobi presmoothing test passed at level 0")
    # 2. Compute residual r = b - Ax -> r after presmoothing.
    r = b - torch.sparse.mm(A0, x.unsqueeze(1)).squeeze(1)
    r_ref = torch.tensor(ms["r"][0], dtype=torch.float32, device="cuda:0")
    if not torch.allclose(r, r_ref, atol=1e-5):
        print("Residual after presmoothing test failed at level 0")
    else:
        print("Residual after presmoothing test passed at level 0")
    # 3. Restrict residual to coarse level: rc = R * r
    R0 = convert_scipy_coo_to_torch_sparse(ms["R"][0], device="cuda:0").coalesce()
    rc = torch.sparse.mm(R0, r.unsqueeze(1)).squeeze(1)
    rc_ref = torch.tensor(ms["bc"][0], dtype=torch.float32, device="cuda:0")
    if not torch.allclose(rc, rc_ref, atol=1e-5):
        print("Restriction of residual test failed at level 0")
    else:
        print("Restriction of residual test passed at level 0")

def test_iterative_methods_level_all():
    omega, get_diag, get_offdiag_from_A_and_diag = utils()
    num_level = 4
    # Use the V-cycle to test all levels.
    xs = []
    bs = []
    for lvl in range(num_level):
        # 0. Prepare A and b.
        A = convert_scipy_coo_to_torch_sparse(ms["A"][lvl], device="cuda:0").coalesce()
        diag_A = get_diag(A)
        off_diag_A = get_offdiag_from_A_and_diag(A, diag_A)
        if lvl == 0:
            b = torch.ones(A.shape[0], dtype=torch.float32, device="cuda:0")
        else:
            b = torch.tensor(ms["bc"][lvl - 1], dtype=torch.float32, device="cuda:0")
        bs.append(b)
        # In presmoothing, we simply have: x -> omega * b / diag(A)
        x = omega * b / diag_A
        x_ref = torch.tensor(ms["x"][lvl], dtype=torch.float32, device="cuda:0")
        if not torch.allclose(x, x_ref, atol=1e-5):
            print(f"Jacobi presmoothing test failed at level {lvl}")
        else:
            print(f"Jacobi presmoothing test passed at level {lvl}")
        # 2. Compute residual r = b - Ax -> r after presmoothing.
        r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        r_ref = torch.tensor(ms["r"][lvl], dtype=torch.float32, device="cuda:0")
        if not torch.allclose(r, r_ref, atol=1e-5):
            print(f"Residual after presmoothing test failed at level {lvl}")
        else:
            print(f"Residual after presmoothing test passed at level {lvl}")
        xs.append(x)
        if lvl < num_level - 1:
            # 3. Restrict residual to coarse level: rc = R * r
            R = convert_scipy_coo_to_torch_sparse(ms["R"][lvl], device="cuda:0").coalesce()
            rc = torch.sparse.mm(R, r.unsqueeze(1)).squeeze(1)
            rc_ref = torch.tensor(ms["bc"][lvl], dtype=torch.float32, device="cuda:0")
            if not torch.allclose(rc, rc_ref, atol=1e-5):
                print(f"Restriction of residual test failed at level {lvl}")
            else:
                print(f"Restriction of residual test passed at level {lvl}")
    # On the coarsest level, we simply solve Ax = b directly.
    Ac = convert_scipy_coo_to_torch_sparse(ms["A"][-1], device="cuda:0").coalesce()
    bc = torch.tensor(ms["bc"][-1], dtype=torch.float32, device="cuda:0")
    # It seems that a "densed lu solver" is used on the coarsest level in AMG.
    x_ref = torch.tensor(ms["xc"][ -1], dtype=torch.float64, device="cuda:0")
    # Now transform Ac, bc to float64 to improve accuracy.
    Ac = Ac.to(torch.float64)
    bc = bc.to(torch.float64)
    x = torch.linalg.solve(Ac.to_dense(), bc)
    # See if x is close to x_ref.
    diff_x_x_ref = torch.norm(x - x_ref) / torch.norm(x_ref)
    if diff_x_x_ref > 1e-5:
        print(f"Coarse level direct solve test failed at level {num_level - 1}, relative error: {diff_x_x_ref}")
    else:
        print(f"Coarse level direct solve test passed at level {num_level - 1}, relative error: {diff_x_x_ref}")
    x = x_ref.float()
    # Now do coarse grid correction and postsmoothing in reverse order.
    for lvl in reversed(range(num_level)):
        A = convert_scipy_coo_to_torch_sparse(ms["A"][lvl], device="cuda:0").coalesce()
        diag_A = get_diag(A)
        off_diag_A = get_offdiag_from_A_and_diag(A, diag_A)
        # Prolongate and correct: x = x + P * xc
        P = convert_scipy_coo_to_torch_sparse(ms["P"][lvl], device="cuda:0").coalesce()
        xc = torch.tensor(ms["xc"][lvl], dtype=torch.float32, device="cuda:0")
        x = xs[lvl] + torch.sparse.mm(P, xc.unsqueeze(1)).squeeze(1)
        x_ref = torch.tensor(ms["xac"][lvl], dtype=torch.float32, device="cuda:0")
        if not torch.allclose(x, x_ref, atol=1e-5):
            print(f"Prolongation and correction test failed at level {lvl}")
        else:
            print(f"Prolongation and correction test passed at level {lvl}")
        # Read b from bc of previous level.
        b = torch.tensor(ms["bc"][lvl - 1], dtype=torch.float32, device="cuda:0") if lvl > 0 else torch.ones(A.shape[0], dtype=torch.float32, device="cuda:0")
        # Postsmoothing: x -> omega * (b - off_diag_A * x) / diag_A + (1 - omega) * x
        x_new = omega * (b - torch.sparse.mm(off_diag_A, x.unsqueeze(1)).squeeze(1)) / diag_A + (1 - omega) * x
        x_ref = torch.tensor(ms["xap"][lvl], dtype=torch.float32, device="cuda:0")
        if not torch.allclose(x_new, x_ref, atol=1e-5):
            print(f"Jacobi postsmoothing test failed at level {lvl}")
        else:
            print(f"Jacobi postsmoothing test passed at level {lvl}")
        x = x_new

load_vec_all(start_index=0, num_level=len(ms["A"]))
# test_iterative_methods_level_all()

def test_torch_V():
    omega, get_diag, get_offdiag_from_A_and_diag = utils()
    num_level = 4
    # Use the V-cycle to test all levels.
    xs = []
    bs = []
    for lvl in range(num_level):
        # 0. Prepare A and b.
        A = convert_scipy_coo_to_torch_sparse(ms["A"][lvl], device="cuda:0").coalesce()
        diag_A = get_diag(A)
        off_diag_A = get_offdiag_from_A_and_diag(A, diag_A)
        if lvl == 0:
            b = torch.ones(A.shape[0], dtype=torch.float32, device="cuda:0")
            bs.append(b.clone())
        else:
            b = bs[-1]
        # In presmoothing, we simply have: x -> omega * b / diag(A)
        x = omega * b / diag_A
        # 2. Compute residual r = b - Ax -> r after presmoothing.
        r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        xs.append(x)
        # 3. Restrict residual to coarse level: rc = R * r
        R = convert_scipy_coo_to_torch_sparse(ms["R"][lvl], device="cuda:0").coalesce()
        rc = torch.sparse.mm(R, r.unsqueeze(1)).squeeze(1)
        bs.append(rc)
    # On the coarsest level, we simply solve Ax = b directly.
    Ac = convert_scipy_coo_to_torch_sparse(ms["A"][-1], device="cuda:0").coalesce()
    bc = bs[-1]
    # It seems that a "densed lu solver" is used on the coarsest level in AMG.
    # Now transform Ac, bc to float64 to improve accuracy.
    Ac = Ac.to(torch.float64)
    bc = bc.to(torch.float64)
    x = torch.linalg.solve(Ac.to_dense(), bc)
    x = x.float()
    # Now do coarse grid correction and postsmoothing in reverse order.
    for lvl in reversed(range(num_level)):
        A = convert_scipy_coo_to_torch_sparse(ms["A"][lvl], device="cuda:0").coalesce()
        diag_A = get_diag(A)
        off_diag_A = get_offdiag_from_A_and_diag(A, diag_A)
        # Prolongate and correct: x = x + P * xc
        P = convert_scipy_coo_to_torch_sparse(ms["P"][lvl], device="cuda:0").coalesce()
        xc = x
        x = xs[lvl] + torch.sparse.mm(P, xc.unsqueeze(1)).squeeze(1)
        # Read b from bc of previous level.
        b = bs[lvl]
        # Postsmoothing: x -> omega * (b - off_diag_A * x) / diag_A + (1 - omega) * x
        x_new = omega * (b - torch.sparse.mm(off_diag_A, x.unsqueeze(1)).squeeze(1)) / diag_A + (1 - omega) * x
        x = x_new
    print("Final result of V-cycle:")
    print(x)
    final_xap_ref = torch.tensor(ms["xap"][0], dtype=torch.float32, device="cuda:0")
    print(x[:32])
    print(final_xap_ref[:32])
    print("Difference norm:", torch.norm(x - final_xap_ref) / torch.norm(final_xap_ref))
    print(x[:32] - final_xap_ref[:32])
    input("?")

test_torch_V()
