import torch
import torch.nn as nn
import torch.nn.functional as F
# Load the real data from the files.
from test_torch_aligned import *

# Important: in all implementation of GCN here, we should NOT have batches!!!!
# The reason is that, when batching in multigrid, we often run **1** A with multiple b's.
# The following operations are all designed for single A, multiple b's.

# Also we assume A is always a sparse COO matrix. (CSR format is not well supported in PyTorch sparse operations.)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        '''
        A single GCN layer using COO sparse matrix multiplication.
        Parameters:
          - in_features:  input feature dimension
          - out_features: output feature dimension
        '''
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, A, x):
        # H = sigma(A X W)
        return F.leaky_relu(self.linear((A @ x)))


class GCNLevel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        '''
        A multi-layer GCN for a single level.
        Parameters:
          - input_dim:  input feature dimension
          - hidden_dim: hidden feature dimension
        '''
        super().__init__()
        self.pre0 = GCNLayer(input_dim, hidden_dim)
        # self.pre1 = GCNLayer(hidden_dim, hidden_dim)
        self.post0 = GCNLayer(hidden_dim * 3, hidden_dim)
        self.post1 = GCNLayer(hidden_dim, hidden_dim)

    def pre(self, A, x):
        x = self.pre0(A, x)
        # x = self.pre1(A, x)
        return x
    
    def post(self, A, x):
        x = self.post0(A, x)
        x = self.post1(A, x)
        return x

    def forward(self, A, x):
        # Note: this function is only used at the finest level.
        return self.post1(A, x)


class LinearEncoder(nn.Module):
    def __init__(self, hidden_dim):
        '''
        A linear encoder that encodes the node features from the diagonal of A.
        Parameters:
          - hidden_dim:  hidden feature dimension
        '''
        super().__init__()
        self.nn = GCNLayer(4, hidden_dim)
    
    def forward(self, A):
        # Get the diagonal of A.
        A_diag = A.coalesce().values()[A.coalesce().indices()[0] == A.coalesce().indices()[1]]
        ret = []
        for pw in [-1, 0, 1, 2]:
            ret.append(A_diag.pow(pw).unsqueeze(-1))
        return self.nn(A, torch.cat(ret, dim=-1))


class LinearDecoder(nn.Module):
    def __init__(self, input_dim):
        '''
        A linear decoder that decodes the edge features back to a single value.
        Parameters:
          - input_dim:  input feature dimension
        '''
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, A, x):
        # A: sparse COO matrix of shape [num_nodes, num_nodes]
        A = A.coalesce()
        indices, values = A.indices(), A.values()
        x = self.linear(x) * 1e-1
        # Use + to ensure out_ij = out_ji.
        out = x[indices[0], :] + x[indices[1], :]  # shape [num_edges, 2]
        updated_value = out[:, 0].exp() * values + out[:, 1]
        # Return a new sparse matrix with the same indices but updated values.
        return torch.sparse_coo_tensor(indices, updated_value, A.size(), device=A.device)


class UGCN(nn.Module):
    def __init__(self, max_level=4, finest_hidden_dim=4):
        super().__init__()
        self.max_level = max_level
        self.base_encoder = LinearEncoder(finest_hidden_dim)
        self.encoder = nn.ModuleList([LinearEncoder(finest_hidden_dim * 2 ** l) for l in range(max_level)])
        self.levels = nn.ModuleList([GCNLevel(finest_hidden_dim * 2 ** l, finest_hidden_dim * 2 ** l) for l in range(max_level)])
        self.decoder = nn.ModuleList([LinearDecoder(finest_hidden_dim * 2 ** l) for l in range(max_level)])

    def lvl(self, model, level_idx):
        if level_idx >= self.max_level:
            return model[-1]
        return model[level_idx]
    
    def encode(self, A, level_idx, last_level_input=None):
        if level_idx == 0:
            return self.base_encoder(A)
        assert last_level_input is not None, "last_level_input should be provided for level_idx > 0"
        # Otherwise, we concat the last level input if provided. (Here is why hidden_dim should be doubled each level.)
        x = self.lvl(self.encoder, level_idx - 1)(A)
        x = torch.cat([x, last_level_input], dim=-1)
        return x
    
    def forward(self, A_list, P_list, R_list):
        # Inputs:
        #   - A_list: list of adjacency matrices at each level, from fine to coarse.
        #   - P_list: list of prolongation matrices from fine to coarse.
        #   - R_list: list of restriction matrices from fine to coarse.
        # Outputs:
        #   - A'_list: list of updated adjacency matrices at each level, from fine to coarse.

        # For example:
        # A0,  A1 = R0 A0 P0,  A2 = R1 A1 P1, A3 = R2 A2 P2, A4 = R3 A3 P3, A5 = R4 A4 P4
        # We should output: A0, A1', A2', A3', A4', A5; (A0, A5 are unchanged)
        # Therefore, for simplicity, we still assume len(A_list) == len(P_list) + 1 == len(R_list) + 1.
        num_levels = len(A_list)
        feat = []
        for l in range(num_levels):
            A = A_list[l]
            x = self.encode(A, l, self.lvl(R_list, l - 1) @ feat[-1] if l > 0 else None)
            feat.append(self.lvl(self.levels, l).pre(A, x))
        ret = []
        # Now feat[l] is the output feature at level l after pre-GCN.
        for l in reversed(range(num_levels)):
            A = A_list[l]
            if l == num_levels - 1:
                x = self.lvl(self.levels, l)(A, feat[l])
            else:
                print(feat[l].shape, P_list[l].shape, x.shape)
                x = torch.cat([feat[l], self.lvl(P_list, l) @ x], dim=-1)
                print("At level", l, "x shape:", x.shape)
                x = self.lvl(self.levels, l).post(A, x)
            A_updated = self.lvl(self.decoder, l)(A, x)
            ret.append(A_updated)
        return list(reversed(ret))


class VCycle(nn.Module):
    def __init__(self, omega_init=0.9, max_num_level=4):
        '''
        A V-cycle implementation for multigrid method.
        Parameters:
          - omega_init: initial relaxation parameter for weighted Jacobi.
          - max_num_level: maximum number of levels.
        '''
        super().__init__()
        self.omega = nn.Parameter(torch.tensor([omega_init] * max_num_level, dtype=torch.float32))
    
    @staticmethod
    def get_diag(U):
        return U.coalesce().values()[U.coalesce().indices()[0] == U.coalesce().indices()[1]]
    
    @staticmethod
    def get_offdiag_from_A_and_diag(A, diag_A):
        I = torch.sparse_coo_tensor(torch.arange(A.shape[0]).unsqueeze(0).repeat(2,1), torch.ones(A.shape[0], device=A.device), A.shape, device=A.device).coalesce()
        off_diag_A = A - torch.sparse_coo_tensor(I.indices(), diag_A, A.shape, device=A.device)
        return off_diag_A
    
    @staticmethod    
    def DU(A):
        D = VCycle.get_diag(A)
        U = VCycle.get_offdiag_from_A_and_diag(A, D)
        return D, U

    def get_omega(self, level_idx):
        if level_idx >= self.omega.numel():
            return self.omega[-1]
        return self.omega[level_idx]


    def forward(self, ms_dict, batched_b):
        num_level = len(ms_dict["A"]) - 1
        # Use the V-cycle to test all levels.
        xs = []
        bs = []
        for lvl in range(num_level):
            # 0. Prepare A and b.
            A = ms_dict["A"][lvl].coalesce()
            D, U = self.DU(A)
            if lvl == 0:
                b = batched_b
                # b = torch.ones(A.shape[0], dtype=torch.float32, device="cuda:0")
                bs.append(b.clone())
            else:
                b = bs[-1]
            # In presmoothing, we simply have: x -> omega * b / diag(A)
            x = self.get_omega(lvl) * b / D.unsqueeze(1)
            # 2. Compute residual r = b - Ax -> r after presmoothing.
            r = b - torch.sparse.mm(A, x)
            xs.append(x)
            # 3. Restrict residual to coarse level: rc = R * r
            R = ms_dict["R"][lvl].coalesce()
            rc = torch.sparse.mm(R, r)
            bs.append(rc)
        # On the coarsest level, we simply solve Ax = b directly.
        Ac = ms_dict["A"][-1].coalesce()
        bc = bs[-1]
        # It seems that a "densed lu solver" is used on the coarsest level in AMG.
        # Now transform Ac, bc to float64 to improve accuracy.
        Ac = Ac.to(torch.float64)
        bc = bc.to(torch.float64)
        x = torch.linalg.solve(Ac.to_dense(), bc)
        x = x.float()
        # Now do coarse grid correction and postsmoothing in reverse order.
        for lvl in reversed(range(num_level)):
            A = ms_dict["A"][lvl].coalesce()
            D, U = self.DU(A)
            # Prolongate and correct: x = x + P * xc
            P = ms_dict["P"][lvl].coalesce()
            x = xs[lvl] + torch.sparse.mm(P, x)
            # Read b from bc of previous level.
            b = bs[lvl]
            # Postsmoothing: x -> omega * (b - off_diag_A * x) / diag_A + (1 - omega) * x
            x_new = self.get_omega(lvl) * (b - torch.sparse.mm(U, x)) / D.unsqueeze(1) + (1 - self.get_omega(lvl)) * x
            x = x_new
        return x


class ResL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A, x, b):
        r = b - torch.sparse.mm(A, x)
        return (r ** 2).mean()


class UMGPCG(nn.Module):
    def __init__(self, max_level=4, finest_hidden_dim=4, omega_init=0.9):
        super().__init__()
        self.nn = UGCN(max_level, finest_hidden_dim)
        self.cycle = VCycle(omega_init, max_level)
        self.criterion = ResL2Loss()

    def forward(self, ms_dict, batched_b):
        # Extract the A, P, R from ms_dict.
        A_list = ms_dict["A"][1:-1]
        P_list = ms_dict["P"][1:-1]
        R_list = ms_dict["R"][1:-1]
        mod_A_list = self.nn(A_list, P_list, R_list)
        ms_dict["A"] = [ms_dict["A"][0]] + mod_A_list + [ms_dict["A"][-1]]
        x = self.cycle(ms_dict, batched_b)
        loss = self.criterion(ms_dict["A"][0], x, batched_b)
        return loss


def test_gcn():
    # Example usage
    # Define a simple graph in CSR format
    indices = torch.tensor([[0, 0, 1, 2],
                            [1, 2, 0, 1]], dtype=torch.long)
    values = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    size = torch.Size([3, 3])
    csr_sparse_mat = torch.sparse_coo_tensor(indices, values, size).to_sparse_csr()

    # Create a GCN layer
    gcn_layer = GCNLayer(in_features=4, out_features=2)

    # Input feature matrix (batch_size=1 for simplicity)
    x = torch.randn(7, 3, 4)  # [batch_size, num_nodes, in_features]
    x.requires_grad_()

    # Forward pass
    output = gcn_layer(csr_sparse_mat, x)
    print(output.shape)

    # Backward pass
    output.sum().backward()
    # print(output)
    # print(x.grad)

    # Print all parameters.
    for name, param in gcn_layer.named_parameters():
        print(f"Parameter {name}: {param.size()}, with value:\n{param}, grad:\n{param.grad}")


def test_u():
    # Example usage of UGCN
    indices = torch.tensor([[0, 1, 2, 0, 1, 2, 1],
                            [1, 0, 1, 0, 1, 2, 2]], dtype=torch.long)
    values = torch.tensor([-1.5, -1.5, -1.0, 1.5, 2.0, 1.0, -1.0], dtype=torch.float32)
    size = torch.Size([3, 3])
    A0 = torch.sparse_coo_tensor(indices, values, size)
    # Let R be a restriction matrix that maps 3 nodes to 2 nodes.
    R0_indices = torch.tensor([[0, 0, 1],
                               [0, 1, 2]], dtype=torch.long)
    R0_values = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    R0_size = torch.Size([2, 3])
    R0 = torch.sparse_coo_tensor(R0_indices, R0_values, R0_size)
    P0 = R0.transpose(0, 1)
    A1 = R0 @ A0 @ P0
    A_list = [A0.coalesce(), A1.coalesce()]
    P_list = [P0.coalesce()]
    R_list = [R0.coalesce()]

    ugcn = UGCN(max_level=2, finest_hidden_dim=4)
    updated_A_list = ugcn(A_list, P_list, R_list)
    for i, A in enumerate(updated_A_list):
        print(f"Updated A at level {i}:")
        print(A)


def u_speed():
    import time
    R_torch = [convert_scipy_coo_to_torch_sparse(mat).to(0).coalesce() for mat in ms["R"]]
    A_torch = [convert_scipy_coo_to_torch_sparse(mat).to(0).coalesce() for mat in ms["A"]]
    P_torch = [convert_scipy_coo_to_torch_sparse(mat).to(0).coalesce() for mat in ms["P"]]
    ugcn = UGCN().to(0)
    torch.compile(ugcn)
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            out = ugcn(A_torch[1:-1], P_torch[1:-1], R_torch[1:-1])
        # Timing
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            out = ugcn(A_torch[1:-1], P_torch[1:-1], R_torch[1:-1])
        torch.cuda.synchronize()
        end = time.time()
    print(f"Avg time per forward: {(end - start) / 10:.6f} seconds")
    for i, A in enumerate(out):
        print(f"Updated A at level {i}:")
        print(A.shape, len(A.values()))


def test_forward(batch_size=32):
    model = UMGPCG().to("cuda:0")
    num_node = ms["A"][0].shape[0]
    # Test if #batch_size right-hand sides work.
    batched_b = torch.randn(num_node, batch_size, device="cuda:0")
    for name in 'RAP':
        for i, mat in enumerate(ms[name]):
            ms[name][i] = convert_scipy_coo_to_torch_sparse(mat).to("cuda:0").coalesce()
    loss = model(ms, batched_b)
    loss.backward()
    print("Memory used: ", torch.cuda.memory_allocated())
    input("Please check the GPU memory usage, then press Enter to continue...")


if __name__ == "__main__":
    test_forward(1)