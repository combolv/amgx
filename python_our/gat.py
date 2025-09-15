import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayerCSR(nn.Module):
    """
    One message-passing round of multi-head GAT on a CSR graph (no deps).
    Parameters (learnable):
      - W:   [H_in, H_out, n_head]
      - a:   [n_head, 2*H_out]   (per-head attention vector for [Vi || Vj])
    Inputs:
      - H:        [N, H_in]            (node features; bf16/fp16/fp32)
      - rowptr:   [N+1]                (CSR row offsets; int32/int64)
      - colind:   [E]                  (CSR column indices; int32/int64)
    Returns:
      - out:      [N, n_head, H_out]   (or [N, n_head*H_out] if concat=True)
      - (optional) alpha_per_edge: [E, n_head] if return_alpha=True
    """
    def __init__(self, H_in, H_out, n_head, negative_slope=0.2, attn_dropout=0.0, concat=False):
        super().__init__()
        self.H_in, self.H_out, self.n_head = H_in, H_out, n_head
        self.negative_slope = negative_slope
        self.attn_dropout = attn_dropout
        self.concat = concat

        # Parameters
        self.W = nn.Parameter(torch.empty(H_in, H_out, n_head))
        self.a = nn.Parameter(torch.empty(n_head, 2 * H_out))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.view(self.H_in, self.H_out * self.n_head))
        nn.init.xavier_uniform_(self.a)

    @staticmethod
    def _edge_src_from_rowptr(rowptr: torch.Tensor) -> torch.Tensor:
        # rowptr: [N+1] → edge_src: [E] with value i repeated deg(i) times
        deg = rowptr[1:] - rowptr[:-1]
        return torch.repeat_interleave(torch.arange(deg.numel(), device=rowptr.device, dtype=rowptr.dtype), deg)

    def forward(self, H, rowptr, colind, return_alpha: bool = False):
        assert H.dim() == 2
        N = H.size(0)
        assert rowptr.numel() == N + 1, "rowptr must have length N+1"
        assert colind.dim() == 1

        device = H.device
        dtype_logits = torch.float32  # stable softmax
        dtype_feats = torch.float32   # accumulators in fp32

        # 1) Linear projection per head: V = H @ W  → [N, n_head, H_out]
        # H: [N, H_in], W: [H_in, H_out, n_head]
        V = torch.einsum('ni,ioh->nho', H.to(dtype_feats), self.W.to(dtype_feats))

        # 2) Build edge list (COO) from CSR
        edge_src = self._edge_src_from_rowptr(rowptr.to(device))              # [E]
        edge_dst = colind.to(device)                                          # [E]
        E = edge_dst.numel()

        # 3) Gather per-edge transformed node features (per head)
        Vi = V.index_select(0, edge_src)                                      # [E, n_head, H_out]
        Vj = V.index_select(0, edge_dst)                                      # [E, n_head, H_out]

        # 4) Attention logits per edge/head: LeakyReLU( a_h^T [Vi || Vj] )
        a = self.a.to(dtype_logits)                                           # [n_head, 2*H_out]
        cat_ij = torch.cat([Vi, Vj], dim=-1)                                  # [E, n_head, 2*H_out]
        e = (cat_ij.to(dtype_logits) * a.unsqueeze(0)).sum(dim=-1)            # [E, n_head]
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # 5) Segmented softmax over neighbors of each src node
        # 5a) row-wise max
        m = torch.full((N, self.n_head), -torch.inf, device=device, dtype=dtype_logits)
        # m[i,h] = max_{edges with src=i} e[edge,h]
        m.index_reduce_(0, edge_src, e, reduce='amax', include_self=True)     # [N, n_head]
        m_edge = m.index_select(0, edge_src)                                  # [E, n_head]

        # 5b) exp/logits and row-wise sum
        w = torch.exp(e - m_edge)                                             # [E, n_head]
        S = torch.zeros((N, self.n_head), device=device, dtype=dtype_logits)
        S.index_add_(0, edge_src, w)                                          # [N, n_head]
        S_edge_inv = (1.0 / (S.index_select(0, edge_src) + 1e-9))             # [E, n_head]
        alpha = w * S_edge_inv                                                # [E, n_head]

        if self.training and self.attn_dropout > 0.0:
            alpha = F.dropout(alpha, p=self.attn_dropout, training=True)

        # 6) Aggregate: out[i,h,:] = sum_{j in N(i)} alpha_{ij,h} * Vj[h,:]
        U = torch.zeros((N, self.n_head, self.H_out), device=device, dtype=dtype_feats)
        U.index_add_(0, edge_src, alpha.unsqueeze(-1) * Vj)                   # [N, n_head, H_out]
        out = U

        # 7) (optional) concatenate heads → [N, n_head*H_out]
        if self.concat:
            out = out.transpose(1, 2).reshape(N, self.n_head * self.H_out)

        if return_alpha:
            return out, alpha  # alpha is per-edge per-head
        return out


# ---------------------------- Minimal sanity check ----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N, H_in, H_out, n_head = 8, 6, 4, 3
    # Toy CSR graph
    rowptr = torch.tensor([0, 2, 5, 5, 7, 9, 10, 12, 12], device=device, dtype=torch.int64)  # N+1
    colind = torch.tensor([1,2, 0,2,3,                    3,4,  1,5,  6,  0,4], device=device, dtype=torch.int64)

    H = torch.randn(N, H_in, device=device, dtype=torch.bfloat16)

    gat = GATLayerCSR(H_in, H_out, n_head, negative_slope=0.2, attn_dropout=0.0, concat=False).to(device)
    out, alpha = gat(H, rowptr, colind, return_alpha=True)

    print("out:", out.shape)     # [N, n_head, H_out]
    print("alpha:", alpha.shape) # [E, n_head]
