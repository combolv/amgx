// gat_refined.cu
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdio.h>

namespace gat_fast
{

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// ========================= Utils =========================
__device__ __forceinline__ float lrelu(float x, float neg_slope) {
    return x >= 0.f ? x : neg_slope * x;
}

// ========================= Kernel 1: V = H @ W (multi-head) =========================
// H: [N, Hin] (bf16), W: [Hin, F, Hh] (bf16), V: [Hh, N, F] (fp32)
__global__ void bf16_proj_heads_kernel(
    const __nv_bfloat16* __restrict__ H,     // [N, Hin]
    const __nv_bfloat16* __restrict__ W,     // [Hin, F, Hh]
    float* __restrict__ V,                   // [Hh, N, F]
    int N, int Hin, int F, int Hh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // node
    if (i >= N) return;

    for (int h = 0; h < Hh; ++h) {
        for (int f = 0; f < F; ++f) {
            float acc = 0.f;
            // dot(H[i,:], W[:,f,h])
            #pragma unroll 4
            for (int k = 0; k < Hin; ++k) {
                float hi = __bfloat162float(H[i*Hin + k]);
                float wk = __bfloat162float(W[(k*F + f)*Hh + h]); // W[k,f,h]
                acc += hi * wk;
            }
            V[(h*N + i)*F + f] = acc; // V[h,i,f]
        }
    }
}

// ========================= Kernel 2a: thread-per-row, per-head =========================
// Good when F is small (<= 16) and N is large.
// Processes a single (row i, head h) per thread.
// Layouts: V[h,i,f], out[h,i,f], a[h, 2F] split into aL (=a[h,0..F-1]) and aR (=a[h,F..2F-1]).
template<int F_STATIC>
__global__ void gat_softmax_agg_thread_kernel_Fx(
    const int* __restrict__ rowptr,    // [N+1]
    const int* __restrict__ colind,    // [E]
    const float* __restrict__ V,       // [Hh, N, F]
    const float* __restrict__ a,       // [Hh, 2F]
    float* __restrict__ out,           // [Hh, N, F]
    int N, int F, int Hh,
    float neg_slope)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int h = blockIdx.y;                             // head
    if (i >= N || h >= Hh) return;

    const float* Vh = V + (h * N) * F;     // base for head h
    const float* a_h = a + h * (2 * F);    // a for head h
    const float* aL = a_h;                 // size F
    const float* aR = a_h + F;             // size F

    // Precompute phi_i = aL · V[i,:]
    float phi_i = 0.f;
    #pragma unroll
    for (int f = 0; f < F_STATIC; ++f)
        phi_i += aL[f] * Vh[i*F + f];

    int start = rowptr[i];
    int end   = rowptr[i+1];

    if (start == end) {
        // No neighbors: output zeros
        for (int f = 0; f < F_STATIC; ++f)
            out[(h*N + i)*F + f] = 0.f;
        return;
    }

    // Pass A: row max over e_ij = lrelu(phi_i + psi_j)
    float m = -CUDART_INF_F;
    for (int e = start; e < end; ++e) {
        int j = colind[e];
        // psi_j = aR · V[j,:]
        float psi_j = 0.f;
        #pragma unroll
        for (int f = 0; f < F_STATIC; ++f)
            psi_j += aR[f] * Vh[j*F + f];
        float logit = lrelu(phi_i + psi_j, neg_slope);
        m = fmaxf(m, logit);
    }

    // Pass B: sum weights and accumulate U = Σ w V[j,:]
    float S = 0.f;
    float acc[F_STATIC];
    #pragma unroll
    for (int f = 0; f < F_STATIC; ++f) acc[f] = 0.f;

    for (int e = start; e < end; ++e) {
        int j = colind[e];
        float psi_j = 0.f;
        #pragma unroll
        for (int f = 0; f < F_STATIC; ++f)
            psi_j += aR[f] * Vh[j*F + f];
        float logit = lrelu(phi_i + psi_j, neg_slope);
        float w = __expf(logit - m);
        S += w;
        #pragma unroll
        for (int f = 0; f < F_STATIC; ++f)
            acc[f] += w * Vh[j*F + f];
    }

    float invS = 1.f / (S + 1e-9f);
    #pragma unroll
    for (int f = 0; f < F_STATIC; ++f)
        out[(h*N + i)*F + f] = acc[f] * invS;
}

// Variant for runtime F (still for small F, up to 16)
__global__ void gat_softmax_agg_thread_kernel_varF(
    const int* __restrict__ rowptr,    // [N+1]
    const int* __restrict__ colind,    // [E]
    const float* __restrict__ V,       // [Hh, N, F]
    const float* __restrict__ a,       // [Hh, 2F]
    float* __restrict__ out,           // [Hh, N, F]
    int N, int F, int Hh,
    float neg_slope)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int h = blockIdx.y;                             // head
    if (i >= N || h >= Hh) return;

    const float* Vh = V + (h * N) * F;     // base for head h
    const float* a_h = a + h * (2 * F);    // a for head h
    const float* aL = a_h;
    const float* aR = a_h + F;

    // Precompute phi_i = aL · V[i,:]
    float phi_i = 0.f;
    for (int f = 0; f < F; ++f)
        phi_i += aL[f] * Vh[i*F + f];

    int start = rowptr[i];
    int end   = rowptr[i+1];

    if (start == end) {
        for (int f = 0; f < F; ++f)
            out[(h*N + i)*F + f] = 0.f;
        return;
    }

    float m = -CUDART_INF_F;
    for (int e = start; e < end; ++e) {
        int j = colind[e];
        float psi_j = 0.f;
        for (int f = 0; f < F; ++f)
            psi_j += aR[f] * Vh[j*F + f];
        float logit = lrelu(phi_i + psi_j, neg_slope);
        m = fmaxf(m, logit);
    }

    float S = 0.f;
    // small stack array (up to ~32 safely)
    float acc_local[32];
    for (int f = 0; f < F; ++f) acc_local[f] = 0.f;

    for (int e = start; e < end; ++e) {
        int j = colind[e];
        float psi_j = 0.f;
        for (int f = 0; f < F; ++f)
            psi_j += aR[f] * Vh[j*F + f];
        float logit = lrelu(phi_i + psi_j, neg_slope);
        float w = __expf(logit - m);
        S += w;
        for (int f = 0; f < F; ++f)
            acc_local[f] += w * Vh[j*F + f];
    }

    float invS = 1.f / (S + 1e-9f);
    for (int f = 0; f < F; ++f)
        out[(h*N + i)*F + f] = acc_local[f] * invS;
}

// ========================= Kernel 2b: warp-per-row, per-head =========================
// One warp handles (row i, head h). Better when F is moderate/large.
// Here each lane processes a feature tile f = lane, lane+32, ...
__inline__ __device__ float warp_allreduce_max(float v) {
    #pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, mask));
    return v;
}
__inline__ __device__ float warp_allreduce_sum(float v) {
    #pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(FULL_MASK, v, mask);
    return v;
}

__global__ void gat_softmax_agg_warp_kernel(
    const int* __restrict__ rowptr,    // [N+1]
    const int* __restrict__ colind,    // [E]
    const float* __restrict__ V,       // [Hh, N, F]
    const float* __restrict__ a,       // [Hh, 2F]
    float* __restrict__ out,           // [Hh, N, F]
    int N, int F, int Hh,
    float neg_slope)
{
    int i = blockIdx.x;          // row
    int h = blockIdx.y;          // head
    int lane = threadIdx.x & (WARP_SIZE - 1);
    if (i >= N || h >= Hh) return;

    const float* Vh = V + (h * N) * F;     // V[h,i,f]
    const float* a_h = a + h * (2 * F);
    const float* aL = a_h;
    const float* aR = a_h + F;

    int start = rowptr[i];
    int end   = rowptr[i+1];

    if (start == end) {
        for (int f = lane; f < F; f += WARP_SIZE)
            out[(h*N + i)*F + f] = 0.f;
        return;
    }

    // Precompute phi_i = aL · V[i,:]
    float phi_i = 0.f;
    for (int f = lane; f < F; f += WARP_SIZE)
        phi_i += aL[f] * Vh[i*F + f];
    phi_i = warp_allreduce_sum(phi_i); // scalar across warp

    // Pass A: row max
    float local_max = -CUDART_INF_F;
    for (int e = start + lane; e < end; e += WARP_SIZE) {
        int j = colind[e];
        // psi_j = aR · V[j,:]
        float psi_j = 0.f;
        for (int f = 0; f < F; ++f)
            psi_j += aR[f] * Vh[j*F + f];
        float logit = lrelu(phi_i + psi_j, neg_slope);
        local_max = fmaxf(local_max, logit);
    }
    float m = warp_allreduce_max(local_max);

    // Pass B: sum weights and accumulate feature tiles
    // We accumulate features per lane (f = lane, lane+32, ...)
    // Need a scalar S across warp; we compute lane-local S and reduce.
    float S_local = 0.f;
    // Initialize out slice to zero for each lane's tiles
    for (int f = lane; f < F; f += WARP_SIZE)
        out[(h*N + i)*F + f] = 0.f;

    for (int e = start; e < end; ++e) {
        int j = colind[e];
        float psi_j = 0.f;
        for (int f = 0; f < F; ++f)
            psi_j += aR[f] * Vh[j*F + f];
        float logit = lrelu(phi_i + psi_j, neg_slope);
        float w = __expf(logit - m);
        S_local += w;

        // accumulate weighted V[j,:] for lane's feature tiles
        for (int f = lane; f < F; f += WARP_SIZE) {
            float prev = out[(h*N + i)*F + f];
            out[(h*N + i)*F + f] = prev + w * Vh[j*F + f];
        }
    }

    float S = warp_allreduce_sum(S_local);
    float invS = 1.f / (S + 1e-9f);

    // Normalize
    for (int f = lane; f < F; f += WARP_SIZE) {
        out[(h*N + i)*F + f] *= invS;
    }
}

// ========================= Host launcher =========================
static inline void launch_thread_kernel_F(
    int F, dim3 grid, dim3 block,
    const int* rowptr, const int* colind,
    const float* V, const float* a, float* out,
    int N, int Hh, float neg_slope)
{
    if (F == 1) {
        gat_softmax_agg_thread_kernel_Fx<1><<<grid, block>>>(rowptr, colind, V, a, out, N, F, Hh, neg_slope);
    } else if (F == 2) {
        gat_softmax_agg_thread_kernel_Fx<2><<<grid, block>>>(rowptr, colind, V, a, out, N, F, Hh, neg_slope);
    } else if (F == 3) {
        gat_softmax_agg_thread_kernel_Fx<3><<<grid, block>>>(rowptr, colind, V, a, out, N, F, Hh, neg_slope);
    } else if (F == 4) {
        gat_softmax_agg_thread_kernel_Fx<4><<<grid, block>>>(rowptr, colind, V, a, out, N, F, Hh, neg_slope);
    } else if (F == 8) {
        gat_softmax_agg_thread_kernel_Fx<8><<<grid, block>>>(rowptr, colind, V, a, out, N, F, Hh, neg_slope);
    } else if (F == 16) {
        gat_softmax_agg_thread_kernel_Fx<16><<<grid, block>>>(rowptr, colind, V, a, out, N, F, Hh, neg_slope);
    } else {
        gat_softmax_agg_thread_kernel_varF<<<grid, block>>>(rowptr, colind, V, a, out, N, F, Hh, neg_slope);
    }
}

// Heuristic:
//  - If F <= 16 → thread-per-row
//  - Else       → warp-per-row
extern "C"
void gat_forward(
    const __nv_bfloat16* H,       // [N,Hin]
    const __nv_bfloat16* W,       // [Hin,F,Hh]
    const int* rowptr,            // [N+1]
    const int* colind,            // [E]
    const float* a,               // [Hh, 2F]
    float* V,                     // [Hh, N, F]  (scratch)
    float* out,                   // [Hh, N, F]
    int N, int Hin, int F, int Hh,
    float negative_slope)
{
    // 1) V = H @ W (bf16 → fp32)
    dim3 block1(128);
    dim3 grid1((N + block1.x - 1) / block1.x);
    bf16_proj_heads_kernel<<<grid1, block1>>>(H, W, V, N, Hin, F, Hh);

    // 2) Attention softmax + aggregation
    if (F <= 16) {
        // Thread-per-row per head
        dim3 blockT(256);
        dim3 gridT((N + blockT.x - 1) / blockT.x, Hh);
        launch_thread_kernel_F(F, gridT, blockT, rowptr, colind, V, a, out, N, Hh, negative_slope);
    } else {
        // Warp-per-row per head
        dim3 gridW(N, Hh);
        dim3 blockW(WARP_SIZE);
        gat_softmax_agg_warp_kernel<<<gridW, blockW>>>(rowptr, colind, V, a, out, N, F, Hh, negative_slope);
    }
}

    
} // namespace gat_fast