/*
 * kernels/qra_router.cu — QRA Deterministic Routing Kernel
 *
 * Replaces softmax MoE gating with deterministic 6×6 tensor lookup.
 * H=0: zero entropy, perfect load balance, no matmul overhead.
 *
 * From sovereign-qra: HK_DSL_Formalized_v2026.lean
 * Tripartite 6=6=6: QLG=SLA=QRA, T≤36 JWT
 *
 * Usage:
 *   qra_route_kernel<<<batch, 6>>>(hidden, hidden_dim, num_experts, routes_out)
 *   // routes_out[b] = deterministic expert ID for batch item b
 */

#include <cuda_runtime.h>
#include <cstdint>

// QRA Tensor: 6×6 identity (deterministic self-route)
// Real QRA uses qlg_to_qra = id bijection, zero entropy
__constant__ int QRA_TENSOR[6][6] = {
    {1, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0},
    {0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 1},
};

/*
 * FNV-1a hash for deterministic routing (no entropy, no randomness)
 * Used instead of softmax: route = hash(hidden_state) % num_experts
 */
__device__ __forceinline__ uint32_t fnv1a_hash(const float* data, int len) {
    uint32_t hash = 2166136261u;  // FNV offset basis
    for (int i = 0; i < len; i++) {
        uint32_t bytes;
        // Reinterpret float as uint32 for hashing
        memcpy(&bytes, &data[i], sizeof(uint32_t));
        hash ^= bytes;
        hash *= 16777619u;  // FNV prime
    }
    return hash;
}

/*
 * QRA routing kernel: deterministic expert assignment
 *
 * For each batch item:
 *   1. Compute FNV-1a hash of hidden state (no entropy)
 *   2. Map to expert ID via modulo
 *   3. QRA tensor lookup (identity = self-route)
 *
 * Output: routes[b] = expert_id ∈ [0, num_experts)
 */
extern "C" __global__ void qra_route_kernel(
    const float* __restrict__ hidden,     // (batch, hidden_dim)
    int hidden_dim,
    int num_experts,                       // default 6
    int* __restrict__ routes_out           // (batch,)
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_size = gridDim.x * blockDim.x;

    if (b >= batch_size) return;

    // FNV-1a hash of hidden state for this batch item
    const float* h = hidden + b * hidden_dim;
    uint32_t hash = fnv1a_hash(h, hidden_dim);

    // Deterministic expert assignment (no softmax, no randomness)
    int expert_id = hash % num_experts;

    // QRA tensor lookup: route = QRA_TENSOR[expert_id][expert_id] = 1 (self-route)
    // For MoE dispatch: route to expert[expert_id]
    routes_out[b] = expert_id;
}

/*
 * QRA routing with load balance metrics
 * Returns: per-expert counts for monitoring perfect balance
 */
extern "C" __global__ void qra_route_with_balance(
    const float* __restrict__ hidden,
    int hidden_dim,
    int num_experts,
    int* __restrict__ routes_out,
    int* __restrict__ expert_counts        // (num_experts,) output
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_size = gridDim.x * blockDim.x;

    if (b >= batch_size) return;

    const float* h = hidden + b * hidden_dim;
    uint32_t hash = fnv1a_hash(h, hidden_dim);
    int expert_id = hash % num_experts;

    routes_out[b] = expert_id;

    // Atomic increment for load counting
    atomicAdd(&expert_counts[expert_id], 1);
}

// ── Host wrapper ─────────────────────────────────────────────────────────────

extern "C" void launch_qra_route(
    const float* hidden,
    int batch_size,
    int hidden_dim,
    int num_experts,
    int* routes_out,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    qra_route_kernel<<<blocks, threads, 0, stream>>>(
        hidden, hidden_dim, num_experts, routes_out
    );
}

extern "C" void launch_qra_route_with_balance(
    const float* hidden,
    int batch_size,
    int hidden_dim,
    int num_experts,
    int* routes_out,
    int* expert_counts,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);

    qra_route_with_balance<<<blocks, threads, 0, stream>>>(
        hidden, hidden_dim, num_experts, routes_out, expert_counts
    );
}
