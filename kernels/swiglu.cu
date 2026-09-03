/*
 * kernels/swiglu.cu — SwiGLU Activation CUDA Kernel
 *
 * SwiGLU(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
 *
 * Fused: gate + up projection + SiLU activation + elementwise multiply + down projection.
 * MiMo-4B: d_model=2048, d_ff=5504.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

/*
 * SiLU activation: silu(x) = x * sigmoid(x)
 */
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ __half silu_fp16(__half x) {
    float xf = __half2float(x);
    return __float2half(silu(xf));
}

/*
 * Fused SwiGLU kernel (float32)
 * For each token: y = down_proj(silu(gate_proj(x)) * up_proj(x))
 *
 * This kernel fuses the activation + elementwise multiply.
 * gate_proj and up_proj are applied separately (as matmul).
 *
 * x: (tokens, d_model)
 * gate: (tokens, d_ff) — after gate_proj matmul
 * up: (tokens, d_ff) — after up_proj matmul
 * down_weight: (d_model, d_ff) — down projection weight
 * output: (tokens, d_model)
 */
extern "C" __global__ void swiglu_fused_kernel(
    const float* __restrict__ gate,   // (tokens, d_ff) — after gate_proj
    const float* __restrict__ up,     // (tokens, d_ff) — after up_proj
    const float* __restrict__ down_weight,  // (d_model, d_ff)
    float* __restrict__ output,       // (tokens, d_model)
    int tokens,
    int d_model,
    int d_ff
) {
    int token_idx = blockIdx.x;
    int d = threadIdx.x;

    if (d >= d_model) return;

    const float* gate_row = gate + token_idx * d_ff;
    const float* up_row = up + token_idx * d_ff;
    float* out_row = output + token_idx * d_model;

    // Accumulate down_proj(d * d_ff : (d+1) * d_ff)
    float sum = 0.0f;
    for (int j = 0; j < d_ff; j++) {
        float g = gate_row[j];
        float u = up_row[j];
        // Fused: silu(gate) * up
        float activated = silu(g) * u;
        // Down projection: out[d] += activated * down_weight[d, j]
        sum += activated * down_weight[d * d_ff + j];
    }
    out_row[d] = sum;
}

/*
 * SiLU + elementwise multiply kernel (fused activation)
 * gate: (tokens, d_ff) → silu(gate)
 * up: (tokens, d_ff)
 * output: (tokens, d_ff) = silu(gate) * up
 */
extern "C" __global__ void swiglu_activate_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int tokens,
    int d_ff
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * d_ff;

    if (idx >= total) return;

    float g = gate[idx];
    float u = up[idx];
    output[idx] = silu(g) * u;
}

/*
 * SiLU + elementwise multiply kernel (fp16)
 */
extern "C" __global__ void swiglu_activate_kernel_fp16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ output,
    int tokens,
    int d_ff
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * d_ff;

    if (idx >= total) return;

    __half g = gate[idx];
    __half u = up[idx];
    output[idx] = silu_fp16(g) * u;
}

// ── Host wrapper ─────────────────────────────────────────────────────────────

extern "C" void launch_swiglu_activate(
    const float* gate,
    const float* up,
    float* output,
    int tokens,
    int d_ff,
    cudaStream_t stream
) {
    int total = tokens * d_ff;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    swiglu_activate_kernel<<<blocks, threads, 0, stream>>>(
        gate, up, output, tokens, d_ff
    );
}

extern "C" void launch_swiglu_activate_fp16(
    const __half* gate,
    const __half* up,
    __half* output,
    int tokens,
    int d_ff,
    cudaStream_t stream
) {
    int total = tokens * d_ff;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    swiglu_activate_kernel_fp16<<<blocks, threads, 0, stream>>>(
        gate, up, output, tokens, d_ff
    );
}
