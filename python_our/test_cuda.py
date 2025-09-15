import torch
import numpy as np
N = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

# Suppose these come from your GNN (node->diag, edge->offdiag)
diag_vals = torch.randn(N, device=device, requires_grad=True)        # shape [N]
edge_index = torch.tensor([[0,1,2,3],
                           [1,2,3,4]], device=device)                # E edges (i->j)
edge_vals = torch.randn(edge_index.shape[1], device=device, requires_grad=True)  # [E]

# Build sparse COO indices and values
diag_idx = torch.arange(N, device=device)
indices = torch.cat([
    torch.stack([diag_idx, diag_idx], dim=0),     # (2, N) for diagonal
    edge_index                                   # (2, E) for off-diagonals
], dim=1)

values = torch.cat([diag_vals, edge_vals], dim=0)  # (N+E,)

A = torch.sparse_coo_tensor(indices, values, (N, N), requires_grad=True, device=device).coalesce()
# Test what if A is in CSC format: not yet supported in PyTorch
# A = A.to_sparse_csc()

# Gradients:
# b = torch.randn(N, device=device)         # no need for grad on b if you don't want it
# y = A @ b
# loss = (y * y).sum()                      # ||Ab||^2 is usually nicer for grads
# loss.backward()
loss2 = A @ A - 2 * A + torch.sparse_coo_tensor(torch.eye(N, device=device).nonzero().t(), torch.ones(N, device=device), (N,N), device=device)
# Get the Frobenius norm squared.
loss = loss2.coalesce().values().pow(2).sum()
loss.backward()

print(diag_vals.grad) 
print(edge_vals.grad)