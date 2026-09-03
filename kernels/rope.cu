/*
 * kernels/rope.cu — Rotary Position Embedding (RoPE) CUDA Kernel
 *
 * Apply rotary embeddings to Q and K tensors:
 *   q_rot = q * cos(freq) + rotate_half(q) * sin(freq)
 *   k_rot = k * cos(freq) + rotate_half(k) * sin(freq)
 *
 * Precomputed freqs_cis: complex64 (cos + i*sin).
 * MiMo-4B: theta=10000, dim=128.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

/*
 * Rotate half: [x0, x1, x2, x3, ...] → [-x1, x0, -x3, x2, ...]
 */
__device__ __forceinline__ void rotate_half_pair(
    float x0, float x1,
    float& out0, float& out1
) {
    out0 = -x1;
    out1 = x0;
}

/*
 * RoPE kernel (float32)
 * Applies rotary position embedding to Q or K tensor.
 *
 * x: (batch, n_heads, seq_len, head_dim)
 * freqs_cis: (seq_len, head_dim/2) — complex64 (cos, sin pairs)
 * output: (batch, n_heads, seq_len, head_dim)
 */
extern "C" __global__ void rope_kernel(
    const float* __restrict__ x,
    const float* __restrict__ freqs_cis,  // (seq_len, head_dim) — interleaved cos,sin
    float* __restrict__ output,
    int batch_size,
    int n_heads,
    int seq_len,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n_heads * seq_len * head_dim;

    if (idx >= total) return;

    // Decode indices
    int head_dim_half = head_dim / 2;
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % n_heads;
    int b = idx / (head_dim * seq_len * n_heads);

    float val = x[idx];

    // Pair index: d < head_dim/2 → pair with d + head_dim/2
    int pair_d = (d < head_dim_half) ? d + head_dim_half : d - head_dim_half;
    int pair_idx = b * (n_heads * seq_len * head_dim) +
                   h * (seq_len * head_dim) +
                   s * head_dim +
                   pair_d;
    float pair_val = x[pair_idx];

    // freqs_cis layout: (seq_len, head_dim) — interleaved [cos0, sin0, cos1, sin1, ...]
    int freq_idx = s * head_dim + d;
    float freq_val = freqs_cis[freq_idx];

    // For d < head_dim_half: cos/sin pair
    // For d >= head_dim_half: same cos/sin as pair
    int cos_sin_idx = s * head_dim + (d % head_dim_half);
    float cos_val = freqs_cis[cos_sin_idx];
    float sin_val = freqs_cis[cos_sin_idx + head_dim_half];

    // Apply rotation
    float result;
    if (d < head_dim_half) {
        result = val * cos_val - pair_val * sin_val;
    } else {
        result = val * cos_val + pair_val * sin_val;
    }

    output[idx] = result;
}

/*
 * RoPE precomputation kernel
 * freqs_cis[t, d] = exp(i * t * theta^(-2d/head_dim))
 * Output: interleaved [cos(t,d), sin(t,d), cos(t,d+1), sin(t,d+1), ...]
 */
extern "C" __global__ void precompute_rope_freqs(
    float* __restrict__ freqs_cis,  // (seq_len, head_dim)
    int seq_len,
    int head_dim,
    float theta
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;

    if (t >= seq_len || d >= head_dim / 2) return;

    float freq = 1.0f / powf(theta, (2.0f * d) / head_dim);
    float angle = t * freq;

    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Interleaved layout: [cos0, cos1, ..., cos_{D/2-1}, sin0, sin1, ..., sin_{D/2-1}]
    freqs_cis[t * head_dim + d] = cos_val;
    freqs_cis[t * head_dim + d + head_dim / 2] = sin_val;
}

// ── Host wrapper ─────────────────────────────────────────────────────────────

extern "C" void launch_rope(
    const float* x,
    const float* freqs_cis,
    float* output,
    int batch_size,
    int n_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    int total = batch_size * n_heads * seq_len * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    rope_kernel<<<blocks, threads, 0, stream>>>(
        x, freqs_cis, output, batch_size, n_heads, seq_len, head_dim
    );
}

extern "C" void launch_precompute_rope_freqs(
    float* freqs_cis,
    int seq_len,
    int head_dim,
    float theta,
    cudaStream_t stream
) {
    dim3 threads(32, 8);
    dim3 blocks(
        (seq_len + threads.x - 1) / threads.x,
        (head_dim / 2 + threads.y - 1) / threads.y
    );

    precompute_rope_freqs<<<blocks, threads, 0, stream>>>(
        freqs_cis, seq_len, head_dim, theta
    );
}
