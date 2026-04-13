import scipy as sp
import numpy as np
import torch

def read_vec(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # First line is size
    N = int(lines[0].strip())
    len_lines = len(lines)
    data = []
    for i in range(N):
        if i + 1 >= len_lines:
            raise ValueError(f"File {filename} has fewer lines than expected size {N}")
        data.append(float(lines[i + 1].strip()))
    vec = np.array(data)
    assert len(vec) == N
    return np.array(vec, dtype=np.float64)

def read_mat(filename):
    # Raise error if it is not a .mtx file
    if not filename.endswith('.mtx'):
        raise ValueError("Only .mtx files are supported")
    # Use scipy to read the matrix
    mat = sp.io.mmread(filename).tocoo()
    return mat

def convert_scipy_coo_to_torch_sparse(mat, device='cpu'):
    # mat is a scipy coo_matrix
    row = torch.tensor(mat.row, dtype=torch.long, device=device)
    col = torch.tensor(mat.col, dtype=torch.long, device=device)
    data = torch.tensor(mat.data, dtype=torch.float32, device=device)
    indices = torch.stack([row, col], dim=0)  # shape (2, nnz)
    shape = mat.shape
    sparse_tensor = torch.sparse_coo_tensor(indices, data, size=shape, device=device)
    return sparse_tensor

if __name__ == "__main__":
    raise NotImplementedError("The following is autogen code. Please integrate it manually.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Read matrix and vector from files
    A = read_our_mat('A.mtx')
    b = read_our_vec('b.vec')

    # Convert to torch sparse tensor
    A_torch = convert_scipy_coo_to_torch_sparse(A, device=device).coalesce()
    b_torch = torch.tensor(b, dtype=torch.float32, device=device)

    # Make sure dimensions match
    assert A_torch.shape[1] == b_torch.shape[0], "Matrix and vector size mismatch"

    # Perform matrix-vector multiplication
    y = torch.sparse.mm(A_torch, b_torch.unsqueeze(1)).squeeze(1)  # y = Ab

    print("Result of Ab:")
    print(y)

    # Optionally, compute a loss and backpropagate if needed
    loss = (y * y).sum()  # Example loss: ||Ab||^2
    loss.backward()

    print("Gradient w.r.t. b:")
    print(b_torch.grad)  # Should be None unless b_torch.requires_grad=True