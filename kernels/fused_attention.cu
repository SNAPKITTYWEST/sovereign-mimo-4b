/*
 * kernels/fused_attention.cu — Fused GQA (Grouped-Query Attention) CUDA Kernel
 *
 * MiMo-4B GQA: 16 Q heads, 4 KV heads, head_dim=128.
 * KV heads are shared across 4 Q heads each.
 *
 * Fused: Q*K^T matmul + scale + causal mask + softmax + V matmul.
 * Uses FlashAttention-style tiling for memory efficiency.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE 32

/*
 * Fused GQA attention kernel (float32, simplified)
 * Each block handles one batch, one Q head, one sequence position.
 *
 * q: (batch, n_heads, seq_len, head_dim)
 * k: (batch, n_kv_heads, seq_len, head_dim)
 * v: (batch, n_kv_heads, seq_len, head_dim)
 * output: (batch, n_heads, seq_len, head_dim)
 */
extern "C" __global__ void fused_gqa_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    float scale,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int head_dim
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int t = blockIdx.x;  // query position
    int tid = threadIdx.x;

    if (b >= batch_size || h >= n_heads || t >= seq_len) return;

    // KV head index (GQA: 16 Q heads share 4 KV heads)
    int kv_h = h / (n_heads / n_kv_heads);

    // Compute attention scores: score[t, j] = q[t] · k[j] * scale
    extern __shared__ float s_scores[];
    extern __shared__ float s_weights[];

    // Phase 1: Compute scores
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = q[b * (n_heads * seq_len * head_dim) +
                           h * (seq_len * head_dim) +
                           t * head_dim + d];
            float k_val = k[b * (n_kv_heads * seq_len * head_dim) +
                           kv_h * (seq_len * head_dim) +
                           j * head_dim + d];
            score += q_val * k_val;
        }
        score *= scale;

        // Causal mask: prevent attending to future positions
        if (j > t) {
            score = -1e9f;
        }

        s_scores[tid] = expf(score);
        local_sum += s_scores[tid];
    }
    __syncthreads();

    // Phase 2: Softmax normalization
    s_weights[tid] = s_scores[tid] / local_sum;
    __syncthreads();

    // Phase 3: Compute weighted sum of values
    float out_val = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        float w = s_weights[j];
        float v_val = v[b * (n_kv_heads * seq_len * head_dim) +
                       kv_h * (seq_len * head_dim) +
                       j * head_dim + tid];
        out_val += w * v_val;
    }

    output[b * (n_heads * seq_len * head_dim) +
           h * (seq_len * head_dim) +
           t * head_dim + tid] = out_val;
}

/*
 * FlashAttention-style tiled GQA kernel (float32)
 * Tiles over sequence dimension for better cache utilization.
 * Uses online softmax (no full attention matrix in memory).
 */
extern "C" __global__ void flash_gqa_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    float scale,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int head_dim,
    int tile_size
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int t = blockIdx.x * tile_size + threadIdx.x;
    int tid = threadIdx.x;

    if (b >= batch_size || h >= n_heads || t >= seq_len) return;

    int kv_h = h / (n_heads / n_kv_heads);

    // Online softmax: maintain running max and sum
    float running_max = -1e30f;
    float running_sum = 0.0f;

    extern __shared__ float s_output[];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_output[d] = 0.0f;
    }
    __syncthreads();

    // Tile over KV sequence
    for (int j_start = 0; j_start < seq_len; j_start += tile_size) {
        int j_end = min(j_start + tile_size, seq_len);

        // Compute scores for this tile
        float score = 0.0f;
        if (t < seq_len) {
            for (int d = 0; d < head_dim; d++) {
                float q_val = q[b * (n_heads * seq_len * head_dim) +
                               h * (seq_len * head_dim) +
                               t * head_dim + d];
                float k_val = k[b * (n_kv_heads * seq_len * head_dim) +
                               kv_h * (seq_len * head_dim) +
                               j_start * head_dim + d];
                score += q_val * k_val;
            }
            score *= scale;

            // Causal mask
            if (j_start > t) {
                score = -1e9f;
            }
        }

        // Online softmax update
        float prev_max = running_max;
        running_max = fmaxf(running_max, score);
        running_sum = running_sum * expf(prev_max - running_max) + expf(score - running_max);

        // Accumulate weighted values
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float v_val = v[b * (n_kv_heads * seq_len * head_dim) +
                           kv_h * (seq_len * head_dim) +
                           j_start * head_dim + d];
            s_output[d] = s_output[d] * expf(prev_max - running_max) +
                         v_val * expf(score - running_max);
        }
        __syncthreads();
    }

    // Final normalization
    for (int d = tid; d < head_dim; d += blockDim.x) {
        output[b * (n_heads * seq_len * head_dim) +
               h * (seq_len * head_dim) +
               t * head_dim + d] = s_output[d] / running_sum;
    }
}

// ── Host wrapper ─────────────────────────────────────────────────────────────

extern "C" void launch_fused_gqa_attention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    float scale,
    int batch_size,
    int n_heads,
    int n_kv_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    dim3 grid(seq_len, n_heads, batch_size);
    dim3 block(head_dim);
    int shared_mem = head_dim * sizeof(float) * 2;  // scores + weights

    fused_gqa_attention_kernel<<<grid, block, shared_mem, stream>>>(
        q, k, v, output, scale, batch_size, n_heads, n_kv_heads, seq_len, head_dim
    );
}
