/*
 * kernels/rmsnorm.cu — Fused RMSNorm CUDA Kernel
 *
 * Root Mean Square Layer Normalization:
 *   y = x / sqrt(mean(x^2) + eps) * weight
 *
 * Fused: single-pass with Welford online variance, shared memory reduction.
 * MiMo-4B uses eps=1e-5.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE 32

/*
 * Fused RMSNorm kernel (float32)
 * Each block handles one token across d_model dimensions.
 * Uses shared memory for reduction.
 */
extern "C" __global__ void rmsnorm_kernel(
    const float* __restrict__ x,        // (tokens, d_model)
    const float* __restrict__ weight,   // (d_model,)
    float* __restrict__ y,              // (tokens, d_model)
    float eps,
    int d_model
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* x_row = x + token_idx * d_model;
    float* y_row = y + token_idx * d_model;

    extern __shared__ float sdata[];

    // Parallel reduction: sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float val = x_row[i];
        local_sum += val * val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Compute rms_inv = 1 / sqrt(sum_sq / d_model + eps)
    float rms_inv = rsqrtf(sdata[0] / d_model + eps);
    __syncthreads();

    // Normalize and scale
    for (int i = tid; i < d_model; i += blockDim.x) {
        y_row[i] = x_row[i] * rms_inv * weight[i];
    }
}

/*
 * Fused RMSNorm kernel (float16)
 * Uses __half2 for 2x throughput on Ampere+.
 */
extern "C" __global__ void rmsnorm_kernel_fp16(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ y,
    float eps,
    int d_model
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    const __half* x_row = x + token_idx * d_model;
    __half* y_row = y + token_idx * d_model;

    extern __shared__ float sdata[];

    // Parallel reduction: sum of squares (fp32 accumulation)
    float local_sum = 0.0f;
    int d_model2 = d_model / 2;
    const __half2* x_row2 = reinterpret_cast<const __half2*>(x_row);

    for (int i = tid; i < d_model2; i += blockDim.x) {
        __half2 v = x_row2[i];
        float2 vf = __half22float2(v);
        local_sum += vf.x * vf.x + vf.y * vf.y;
    }
    // Handle odd dimension
    if (tid == 0 && (d_model & 1)) {
        float v = __half2float(x_row[d_model - 1]);
        local_sum += v * v;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float rms_inv = rsqrtf(sdata[0] / d_model + eps);
    __synthreads();

    // Normalize and scale (fp16)
    __half2* y_row2 = reinterpret_cast<__half2*>(y_row);
    const __half2* w2 = reinterpret_cast<const __half2*>(weight);

    for (int i = tid; i < d_model2; i += blockDim.x) {
        __half2 v = x_row2[i];
        __half2 w = w2[i];
        float2 vf = __half22float2(v);
        float2 wf = __half22float2(w);
        __half2 result = __float22half2_rn(make_float2(
            vf.x * rms_inv * wf.x,
            vf.y * rms_inv * wf.y
        ));
        y_row2[i] = result;
    }
    if (tid == 0 && (d_model & 1)) {
        float v = __half2float(x_row[d_model - 1]);
        float w = __half2float(weight[d_model - 1]);
        y_row[d_model - 1] = __float2half(v * rms_inv * w);
    }
}

// ── Host wrapper ─────────────────────────────────────────────────────────────

extern "C" void launch_rmsnorm(
    const float* x,
    const float* weight,
    float* y,
    float eps,
    int tokens,
    int d_model,
    cudaStream_t stream
) {
    int threads = 256;
    if (d_model <= 256) threads = 128;
    if (d_model <= 128) threads = 64;

    int shared_mem = threads * sizeof(float);

    rmsnorm_kernel<<<tokens, threads, shared_mem, stream>>>(
        x, weight, y, eps, d_model
    );
}

extern "C" void launch_rmsnorm_fp16(
    const __half* x,
    const __half* weight,
    __half* y,
    float eps,
    int tokens,
    int d_model,
    cudaStream_t stream
) {
    int threads = 256;
    int shared_mem = threads * sizeof(float);

    rmsnorm_kernel_fp16<<<tokens, threads, shared_mem, stream>>>(
        x, weight, y, eps, d_model
    );
}
