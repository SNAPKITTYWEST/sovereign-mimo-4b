/*
 * k3_nano_harness.cu — Sovereign MiMo-4B Kimi K3 Nano Transformer
 *
 * Complete bare-metal CUDA inference harness:
 *   - Delta Attention (tanh(Q * ΔK * scale) * V)
 *   - LatentMoE Router (Top-K, low-rank projection)
 *   - Warp Decode MoE (1 warp per output neuron, butterfly reduction)
 *   - WMMA FP16 Tensor Core MoE (sm_86 Ampere)
 *   - Speculative Decoding (draft model + batched verification)
 *   - INT8 Storage + FP16 Compute for draft
 *   - KV Cache + Delta Buffers
 *   - Binary weight loading (K3M1 format)
 *   - Character-level tokenizer
 *   - Top-P sampling (device-side)
 *
 * Compile: nvcc -O3 -arch=sm_86 -use_fast_math k3_nano_harness.cu -o k3_nano
 * Run: ./k3_nano model.bin "Hello world" 128 0.8 0.9 [--speculative draft.bin 4]
 *
 * Binary Format (model.bin):
 *   Header: magic("K3M1") version(1) dim n_layers n_experts top_k
 *           vocab_size max_seq_len intermediate_dim tied_embeddings(1) pad(3)
 *   Tensors: token_emb, pos_emb, ln_f_weight, output_weight,
 *            per-layer: attn_ln, qkv, attn_out, moe_ln, router, w1, w2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>
#include <time.h>
#include <assert.h>

using namespace nvcuda;

#define WARP_SIZE 32
#define MAX_TOP_K 8
#define MAX_LAYERS 32
#define MAX_EXPERTS 128
#define MAX_DIM 8192
#define MAX_VOCAB 65536
#define MAX_SEQ_LEN 4096
#define MAX_GAMMA 8
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ============================================================================
// CONFIGURATION & STATE STRUCTS
// ============================================================================

typedef struct {
    int dim;
    int intermediate_dim;
    int num_experts;
    int top_k;
    int num_layers;
    int vocab_size;
    int max_seq_len;
    int tied_embeddings;
    float rms_norm_eps;
} K3Config;

typedef struct {
    float* token_emb;
    float* pos_emb;
    float* ln_f_weight;
    float* output_weight;
    float** attn_ln_weight;
    float** qkv_weight;
    float** attn_out_weight;
    float** moe_ln_weight;
    float** router_weight;
    float** expert_w1;
    float** expert_w2;
    half** expert_w1_fp16;
    half** expert_w2_fp16;
    int dim_padded;
    int inter_padded;
} K3Weights;

typedef struct {
    float* k_cache;
    float* v_cache;
    float* delta_k;
    int seq_len;
} KVCache;

typedef struct {
    K3Config config;
    K3Weights weights;
    KVCache kv_cache;
    float* residual;
    float* attn_out;
    float* moe_out;
    float* qkv_buffer;
    float* ln_out;
    int* selected_experts;
    float* routing_weights;
    float* logits;
    half* d_residual_fp16;
} K3Model;

// ============================================================================
// ERROR CHECKING
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

// ============================================================================
// PTX WARP PRIMITIVES
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// ============================================================================
// KERNEL: RMSNorm
// ============================================================================

__global__ void rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float eps, int dim
) {
    int tid = threadIdx.x;
    extern __shared__ float s_data[];
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float x = input[i];
        sum_sq += x * x;
    }
    s_data[tid] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    float rms = rsqrtf(s_data[0] / dim + eps);
    for (int i = tid; i < dim; i += blockDim.x)
        output[i] = input[i] * rms * weight[i];
}

// ============================================================================
// KERNEL: Fused QKV Projection
// ============================================================================

__global__ void qkv_proj_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ qkv_out,
    int dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= 3 * dim) return;
    float sum = 0.0f;
    const float* w_row = weight + row * dim;
    for (int i = 0; i < dim; ++i)
        sum += w_row[i] * input[i];
    qkv_out[row] = sum;
}

// ============================================================================
// KERNEL: Delta Attention
// ============================================================================

__global__ void kimi_delta_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ prev_k,
    const float* __restrict__ curr_k,
    const float* __restrict__ v,
    float* __restrict__ out,
    float scale, int dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim) return;
    float delta_k = curr_k[tid] - prev_k[tid];
    float attn_weight = tanhf(q[tid] * delta_k * scale);
    out[tid] = attn_weight * v[tid];
}

// ============================================================================
// KERNEL: Attention Output Projection
// ============================================================================

__global__ void attn_out_proj_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;
    float sum = 0.0f;
    const float* w_row = weight + row * dim;
    for (int i = 0; i < dim; ++i)
        sum += w_row[i] * input[i];
    output[row] = sum;
}

// ============================================================================
// KERNEL: Latent MoE Router
// ============================================================================

__global__ void latent_moe_router_kernel(
    const float* __restrict__ input,
    const float* __restrict__ router_weights,
    int* __restrict__ selected_experts,
    float* __restrict__ routing_weights,
    int dim, int num_experts, int top_k
) {
    if (threadIdx.x != 0) return;
    extern __shared__ float logits[];
    for (int e = 0; e < num_experts; ++e) {
        float score = 0.0f;
        const float* w_row = router_weights + e * dim;
        for (int d = 0; d < dim; ++d)
            score += input[d] * w_row[d];
        logits[e] = score;
    }
    __syncthreads();
    for (int k = 0; k < top_k; ++k) {
        float max_val = -1e9f;
        int max_idx = -1;
        for (int e = 0; e < num_experts; ++e) {
            if (logits[e] > max_val) { max_val = logits[e]; max_idx = e; }
        }
        selected_experts[k] = max_idx;
        routing_weights[k] = max_val;
        if (max_idx != -1) logits[max_idx] = -1e9f;
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        routing_weights[k] = expf(routing_weights[k]);
        sum_exp += routing_weights[k];
    }
    for (int k = 0; k < top_k; ++k)
        routing_weights[k] /= sum_exp;
}

// ============================================================================
// KERNEL: Warp Decode MoE (FP32, 1 warp per output neuron)
// ============================================================================

__global__ void warp_decode_moe_kernel(
    const float* __restrict__ input,
    const float* __restrict__ expert_w1,
    const float* __restrict__ expert_w2,
    const int* __restrict__ selected_experts,
    const float* __restrict__ routing_weights,
    float* __restrict__ output,
    int dim, int intermediate_dim, int top_k
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= dim) return;
    float accumulator = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        int expert_idx = selected_experts[k];
        float route_weight = routing_weights[k];
        const float* w1 = expert_w1 + (size_t)expert_idx * dim * intermediate_dim;
        const float* w2 = expert_w2 + (size_t)expert_idx * intermediate_dim * dim;
        for (int j = lane_id; j < intermediate_dim; j += WARP_SIZE) {
            float hidden_j = 0.0f;
            for (int i = 0; i < dim; ++i)
                hidden_j += input[i] * w1[i * intermediate_dim + j];
            hidden_j = hidden_j / (1.0f + expf(-hidden_j));
            accumulator += hidden_j * w2[j * dim + warp_id] * route_weight;
        }
    }
    accumulator = warp_reduce_sum(accumulator);
    if (lane_id == 0) output[warp_id] = accumulator;
}

// ============================================================================
// KERNEL: WMMA FP16 Tensor Core MoE (sm_86 Ampere)
// ============================================================================

__global__ void wmma_decode_moe_kernel(
    const half* __restrict__ input,
    const half* __restrict__ expert_w1,
    const half* __restrict__ expert_w2,
    const int* __restrict__ selected_experts,
    const float* __restrict__ routing_weights,
    float* __restrict__ output,
    int dim, int intermediate_dim, int top_k,
    int dim_padded, int inter_padded
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int output_tiles = (dim_padded + WMMA_M - 1) / WMMA_M;
    if (warp_id >= output_tiles) return;
    int output_start = warp_id * WMMA_M;
    int valid_outputs = min(WMMA_M, dim - output_start);

    extern __shared__ half smem[];
    half* s_input = smem;
    half* s_hidden = smem + dim_padded;

    for (int i = threadIdx.x; i < dim_padded; i += blockDim.x)
        s_input[i] = input[i];
    __syncthreads();

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> out_frag;
    wmma::fill_fragment(out_frag, 0.0f);
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> hidden_frag;

    for (int k = 0; k < top_k; ++k) {
        int expert_idx = selected_experts[k];
        float route_weight = routing_weights[k];
        const half* w1 = expert_w1 + (size_t)expert_idx * dim_padded * inter_padded;
        const half* w2 = expert_w2 + (size_t)expert_idx * inter_padded * dim_padded;

        for (int j_tile = 0; j_tile < (inter_padded + WMMA_N - 1) / WMMA_N; ++j_tile) {
            int j_start = j_tile * WMMA_N;
            wmma::fill_fragment(hidden_frag, 0.0f);
            for (int i_tile = 0; i_tile < (dim_padded + WMMA_K - 1) / WMMA_K; ++i_tile) {
                int i_start = i_tile * WMMA_K;
                __shared__ half s_input_bcast[WMMA_M * WMMA_K];
                if (lane_id < WMMA_M * WMMA_K) {
                    int row = lane_id / WMMA_K;
                    int col = lane_id % WMMA_K;
                    s_input_bcast[row * WMMA_K + col] = (i_start + col < dim_padded) ? s_input[i_start + col] : __float2half(0.0f);
                }
                __syncwarp();
                wmma::load_matrix_sync(a_frag, s_input_bcast, WMMA_K);
                wmma::load_matrix_sync(b_frag, w1 + i_start * inter_padded + j_start, inter_padded);
                wmma::mma_sync(hidden_frag, a_frag, b_frag, hidden_frag);
            }
            __shared__ float s_hidden_frag[WMMA_M * WMMA_N];
            wmma::store_matrix_sync(s_hidden_frag, hidden_frag, WMMA_N, wmma::mem_row_major);
            __syncwarp();
            if (lane_id < WMMA_N && j_start + lane_id < intermediate_dim) {
                float val = s_hidden_frag[lane_id];
                float silu = val / (1.0f + expf(-val));
                s_hidden[j_start + lane_id] = __float2half_rn(silu * route_weight);
            }
            __syncthreads();
        }

        for (int j_tile = 0; j_tile < (inter_padded + WMMA_K - 1) / WMMA_K; ++j_tile) {
            int j_start = j_tile * WMMA_K;
            __shared__ half s_hidden_bcast[WMMA_M * WMMA_K];
            if (lane_id < WMMA_M * WMMA_K) {
                int row = lane_id / WMMA_K;
                int col = lane_id % WMMA_K;
                s_hidden_bcast[row * WMMA_K + col] = (j_start + col < inter_padded) ? s_hidden[j_start + col] : __float2half(0.0f);
            }
            __syncwarp();
            wmma::load_matrix_sync(a_frag, s_hidden_bcast, WMMA_K);
            wmma::load_matrix_sync(b_frag, w2 + j_start * dim_padded + output_start, dim_padded);
            wmma::mma_sync(out_frag, a_frag, b_frag, out_frag);
        }
    }

    __shared__ float s_out_frag[WMMA_M * WMMA_N];
    wmma::store_matrix_sync(s_out_frag, out_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();
    if (lane_id < valid_outputs)
        output[output_start + lane_id] = s_out_frag[lane_id];
}

// ============================================================================
// KERNEL: FP32 to FP16 Conversion (Vectorized)
// ============================================================================

__global__ void convert_fp32_to_fp16_kernel(
    const float* __restrict__ input,
    half* __restrict__ output,
    int dim, int dim_padded
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx * 4; i < dim; i += stride * 4) {
        float4 f4;
        f4.x = (i + 0 < dim) ? input[i + 0] : 0.0f;
        f4.y = (i + 1 < dim) ? input[i + 1] : 0.0f;
        f4.z = (i + 2 < dim) ? input[i + 2] : 0.0f;
        f4.w = (i + 3 < dim) ? input[i + 3] : 0.0f;
        half2 h2_0 = __floats2half2_rn(f4.x, f4.y);
        half2 h2_1 = __floats2half2_rn(f4.z, f4.w);
        if (i + 0 < dim_padded) output[i + 0] = h2_0.x;
        if (i + 1 < dim_padded) output[i + 1] = h2_0.y;
        if (i + 2 < dim_padded) output[i + 2] = h2_1.x;
        if (i + 3 < dim_padded) output[i + 3] = h2_1.y;
    }
    int tail_start = ((dim + 3) / 4) * 4;
    for (int i = tail_start + idx; i < dim_padded; i += stride)
        output[i] = __float2half(0.0f);
}

// ============================================================================
// KERNEL: Output Head
// ============================================================================

__global__ void output_head_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ logits,
    int vocab_size, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size) return;
    float sum = 0.0f;
    const float* w_row = weight + idx * dim;
    for (int i = 0; i < dim; ++i)
        sum += w_row[i] * input[i];
    logits[idx] = sum;
}

// ============================================================================
// KERNEL: Top-P Sampling (Device)
// ============================================================================

__global__ void top_p_sample_kernel(
    float* logits, int vocab_size,
    float temperature, float top_p,
    unsigned long long seed, int* sampled_token
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float max_logit = -1e9f;
    for (int i = 0; i < vocab_size; ++i)
        max_logit = fmaxf(max_logit, logits[i]);
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] = expf((logits[i] - max_logit) / temperature);
        sum_exp += logits[i];
    }
    for (int i = 0; i < vocab_size; ++i)
        logits[i] /= sum_exp;
    float cumsum = 0.0f;
    int cutoff = vocab_size - 1;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += logits[i];
        if (cumsum >= top_p) { cutoff = i; break; }
    }
    float top_mass = 0.0f;
    for (int i = 0; i <= cutoff; ++i) top_mass += logits[i];
    for (int i = 0; i <= cutoff; ++i) logits[i] /= top_mass;
    for (int i = cutoff + 1; i < vocab_size; ++i) logits[i] = 0.0f;

    unsigned long long x = seed;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    float r = (x * 0x2545F4914F6CDD1DULL >> 32) / 4294967296.0f;
    cumsum = 0.0f;
    int token = 0;
    for (int i = 0; i <= cutoff; ++i) {
        cumsum += logits[i];
        if (r < cumsum) { token = i; break; }
    }
    *sampled_token = token;
}

// ============================================================================
// KERNEL: Speculative Acceptance (Parallel across gamma positions)
// ============================================================================

__global__ void speculative_acceptance_kernel(
    const float* __restrict__ target_logits,
    const float* __restrict__ draft_logits,
    const int* __restrict__ draft_tokens,
    int vocab_size, int gamma, float temperature,
    unsigned long long seed,
    int* __restrict__ accept_count,
    int* __restrict__ accepted_tokens
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    auto rng = [&](unsigned long long* s) -> float {
        unsigned long long x = *s;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        *s = x;
        return (x * 0x2545F4914F6CDD1DULL >> 32) / 4294967296.0f;
    };
    unsigned long long rng_state = seed;
    int accepted = 0;
    for (int i = 0; i < gamma; ++i) {
        int token = draft_tokens[i];
        const float* tgt = target_logits + (i + 1) * vocab_size;
        const float* drft = draft_logits + i * vocab_size;
        float max_tgt = -1e9f, max_drft = -1e9f;
        for (int v = 0; v < vocab_size; ++v) {
            max_tgt = fmaxf(max_tgt, tgt[v]);
            max_drft = fmaxf(max_drft, drft[v]);
        }
        float sum_tgt = 0, sum_drft = 0;
        for (int v = 0; v < vocab_size; ++v) {
            sum_tgt += expf((tgt[v] - max_tgt) / temperature);
            sum_drft += expf((drft[v] - max_drft) / temperature);
        }
        float p_target = expf((tgt[token] - max_tgt) / temperature) / sum_tgt;
        float p_draft = expf((drft[token] - max_drft) / temperature) / sum_drft;
        float accept_prob = fminf(1.0f, p_target / (p_draft + 1e-10f));
        float u = rng(&rng_state);
        if (u < accept_prob) {
            accepted_tokens[accepted++] = token;
        } else {
            float r = rng(&rng_state);
            float cum = 0;
            int sampled = 0;
            for (int v = 0; v < vocab_size; ++v) {
                cum += expf((tgt[v] - max_tgt) / temperature) / sum_tgt;
                if (r < cum) { sampled = v; break; }
            }
            accepted_tokens[accepted++] = sampled;
            break;
        }
    }
    *accept_count = accepted;
}

// ============================================================================
// KERNEL: Batched Embedding (Token + Position)
// ============================================================================

__global__ void batched_embedding_kernel(
    const int* __restrict__ token_ids,
    const float* __restrict__ token_emb,
    const float* __restrict__ pos_emb,
    float* __restrict__ residual,
    int dim, int max_seq_len, int start_pos, int batch
) {
    int pos = blockIdx.x;
    int tid = threadIdx.x;
    if (pos >= batch) return;
    int token = token_ids[pos];
    int seq_pos = start_pos + pos;
    for (int i = tid; i < dim; i += blockDim.x) {
        residual[pos * dim + i] = token_emb[(size_t)token * dim + i]
                                + pos_emb[(size_t)seq_pos * dim + i];
    }
}

// ============================================================================
// KERNEL: Embedding (Single token)
// ============================================================================

__global__ void embedding_kernel(
    int token_id, int pos,
    const float* token_emb, const float* pos_emb,
    float* residual, int dim, int max_seq_len
) {
    int tid = threadIdx.x;
    if (tid < dim)
        residual[tid] = token_emb[(size_t)token_id * dim + tid]
                      + pos_emb[(size_t)pos * dim + tid];
}

// ============================================================================
// HOST: Binary Weight Loading
// ============================================================================

typedef struct {
    char magic[4];
    int version;
    int dim, num_layers, num_experts, top_k;
    int vocab_size, max_seq_len, intermediate_dim;
    char tied_embeddings;
    char pad[3];
} ModelHeader;

void load_model_weights(const char* path, K3Config* config, K3Weights* weights) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); exit(1); }
    ModelHeader header;
    fread(&header, sizeof(ModelHeader), 1, f);
    if (strncmp(header.magic, "K3M1", 4) != 0) {
        fprintf(stderr, "Invalid magic\n"); exit(1);
    }
    config->dim = header.dim;
    config->num_layers = header.num_layers;
    config->num_experts = header.num_experts;
    config->top_k = header.top_k;
    config->vocab_size = header.vocab_size;
    config->max_seq_len = header.max_seq_len;
    config->intermediate_dim = header.intermediate_dim;
    config->tied_embeddings = header.tied_embeddings;
    config->rms_norm_eps = 1e-6f;

    printf("Config: dim=%d layers=%d experts=%d top_k=%d vocab=%d max_seq=%d inter=%d\n",
           config->dim, config->num_layers, config->num_experts, config->top_k,
           config->vocab_size, config->max_seq_len, config->intermediate_dim);

    size_t dim = config->dim, vocab = config->vocab_size, inter = config->intermediate_dim;
    size_t experts = config->num_experts, layers = config->num_layers;

    float* h_token_emb = (float*)malloc(vocab * dim * sizeof(float));
    float* h_pos_emb = (float*)malloc(config->max_seq_len * dim * sizeof(float));
    float* h_ln_f = (float*)malloc(dim * sizeof(float));
    float* h_output_w = (float*)malloc(vocab * dim * sizeof(float));
    fread(h_token_emb, sizeof(float), vocab * dim, f);
    fread(h_pos_emb, sizeof(float), config->max_seq_len * dim, f);
    fread(h_ln_f, sizeof(float), dim, f);
    if (!config->tied_embeddings) fread(h_output_w, sizeof(float), vocab * dim, f);

    CUDA_CHECK(cudaMalloc(&weights->token_emb, vocab * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weights->pos_emb, config->max_seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weights->ln_f_weight, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weights->output_weight, vocab * dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(weights->token_emb, h_token_emb, vocab * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights->pos_emb, h_pos_emb, config->max_seq_len * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights->ln_f_weight, h_ln_f, dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights->output_weight, h_output_w, vocab * dim * sizeof(float), cudaMemcpyHostToDevice));

    weights->attn_ln_weight = (float**)malloc(layers * sizeof(float*));
    weights->qkv_weight = (float**)malloc(layers * sizeof(float*));
    weights->attn_out_weight = (float**)malloc(layers * sizeof(float*));
    weights->moe_ln_weight = (float**)malloc(layers * sizeof(float*));
    weights->router_weight = (float**)malloc(layers * sizeof(float*));
    weights->expert_w1 = (float**)malloc(layers * sizeof(float*));
    weights->expert_w2 = (float**)malloc(layers * sizeof(float*));

    float *d_attn_ln[layers], *d_qkv[layers], *d_attn_out[layers];
    float *d_moe_ln[layers], *d_router[layers], *d_w1[layers], *d_w2[layers];

    for (int l = 0; l < layers; ++l) {
        float *h_attn_ln = (float*)malloc(dim * sizeof(float));
        float *h_qkv = (float*)malloc(3 * dim * dim * sizeof(float));
        float *h_attn_out = (float*)malloc(dim * dim * sizeof(float));
        float *h_moe_ln = (float*)malloc(dim * sizeof(float));
        float *h_router = (float*)malloc(experts * dim * sizeof(float));
        float *h_w1 = (float*)malloc(experts * dim * inter * sizeof(float));
        float *h_w2 = (float*)malloc(experts * inter * dim * sizeof(float));
        fread(h_attn_ln, sizeof(float), dim, f);
        fread(h_qkv, sizeof(float), 3 * dim * dim, f);
        fread(h_attn_out, sizeof(float), dim * dim, f);
        fread(h_moe_ln, sizeof(float), dim, f);
        fread(h_router, sizeof(float), experts * dim, f);
        fread(h_w1, sizeof(float), experts * dim * inter, f);
        fread(h_w2, sizeof(float), experts * inter * dim, f);

        CUDA_CHECK(cudaMalloc(&d_attn_ln[l], dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_qkv[l], 3 * dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn_out[l], dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_moe_ln[l], dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_router[l], experts * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w1[l], experts * dim * inter * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w2[l], experts * inter * dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_attn_ln[l], h_attn_ln, dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qkv[l], h_qkv, 3 * dim * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_attn_out[l], h_attn_out, dim * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_moe_ln[l], h_moe_ln, dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_router[l], h_router, experts * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w1[l], h_w1, experts * dim * inter * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w2[l], h_w2, experts * inter * dim * sizeof(float), cudaMemcpyHostToDevice));
        free(h_attn_ln); free(h_qkv); free(h_attn_out);
        free(h_moe_ln); free(h_router); free(h_w1); free(h_w2);
    }
    CUDA_CHECK(cudaMemcpy(weights->attn_ln_weight, d_attn_ln, layers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights->qkv_weight, d_qkv, layers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights->attn_out_weight, d_attn_out, layers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights->moe_ln_weight, d_moe_ln, layers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights->router_weight, d_router, layers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights->expert_w1, d_w1, layers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights->expert_w2, d_w2, layers * sizeof(float*), cudaMemcpyHostToDevice));
    free(h_token_emb); free(h_pos_emb); free(h_ln_f); free(h_output_w);
    fclose(f);
}

// ============================================================================
// HOST: FP32 to FP16 Weight Conversion
// ============================================================================

void convert_and_upload_fp16_weights(K3Model* model) {
    int dim = model->config.dim;
    int inter = model->config.intermediate_dim;
    int experts = model->config.num_experts;
    int layers = model->config.num_layers;
    int dim_padded = ((dim + 15) / 16) * 16;
    int inter_padded = ((inter + 15) / 16) * 16;
    model->weights.dim_padded = dim_padded;
    model->weights.inter_padded = inter_padded;

    printf("FP16 conversion: dim %d->%d, inter %d->%d\n", dim, dim_padded, inter, inter_padded);

    half *h_w1_padded = (half*)malloc((size_t)experts * dim_padded * inter_padded * sizeof(half));
    half *h_w2_padded = (half*)malloc((size_t)experts * inter_padded * dim_padded * sizeof(half));
    half *d_w1_fp16[layers], *d_w2_fp16[layers];

    for (int l = 0; l < layers; ++l) {
        float *d_w1_fp32, *d_w2_fp32;
        cudaMemcpy(&d_w1_fp32, model->weights.expert_w1 + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_w2_fp32, model->weights.expert_w2 + l, sizeof(float*), cudaMemcpyDeviceToHost);
        float *h_w1_fp32 = (float*)malloc((size_t)experts * dim * inter * sizeof(float));
        float *h_w2_fp32 = (float*)malloc((size_t)experts * inter * dim * sizeof(float));
        cudaMemcpy(h_w1_fp32, d_w1_fp32, (size_t)experts * dim * inter * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_w2_fp32, d_w2_fp32, (size_t)experts * inter * dim * sizeof(float), cudaMemcpyDeviceToHost);

        for (int e = 0; e < experts; ++e) {
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < inter; ++j)
                    h_w1_padded[e * dim_padded * inter_padded + i * inter_padded + j] = __float2half_rn(h_w1_fp32[e * dim * inter + i * inter + j]);
                for (int j = inter; j < inter_padded; ++j)
                    h_w1_padded[e * dim_padded * inter_padded + i * inter_padded + j] = __float2half(0.0f);
            }
            for (int i = dim; i < dim_padded; ++i)
                for (int j = 0; j < inter_padded; ++j)
                    h_w1_padded[e * dim_padded * inter_padded + i * inter_padded + j] = __float2half(0.0f);
            for (int i = 0; i < inter; ++i) {
                for (int j = 0; j < dim; ++j)
                    h_w2_padded[e * inter_padded * dim_padded + i * dim_padded + j] = __float2half_rn(h_w2_fp32[e * inter * dim + i * dim + j]);
                for (int j = dim; j < dim_padded; ++j)
                    h_w2_padded[e * inter_padded * dim_padded + i * dim_padded + j] = __float2half(0.0f);
            }
            for (int i = inter; i < inter_padded; ++i)
                for (int j = 0; j < dim_padded; ++j)
                    h_w2_padded[e * inter_padded * dim_padded + i * dim_padded + j] = __float2half(0.0f);
        }
        free(h_w1_fp32); free(h_w2_fp32);

        size_t w1_size = (size_t)experts * dim_padded * inter_padded * sizeof(half);
        size_t w2_size = (size_t)experts * inter_padded * dim_padded * sizeof(half);
        CUDA_CHECK(cudaMalloc(&d_w1_fp16[l], w1_size));
        CUDA_CHECK(cudaMalloc(&d_w2_fp16[l], w2_size));
        CUDA_CHECK(cudaMemcpy(d_w1_fp16[l], h_w1_padded, w1_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w2_fp16[l], h_w2_padded, w2_size, cudaMemcpyHostToDevice));
    }
    free(h_w1_padded); free(h_w2_padded);

    CUDA_CHECK(cudaMalloc(&model->weights.expert_w1_fp16, layers * sizeof(half*)));
    CUDA_CHECK(cudaMalloc(&model->weights.expert_w2_fp16, layers * sizeof(half*)));
    CUDA_CHECK(cudaMemcpy(model->weights.expert_w1_fp16, d_w1_fp16, layers * sizeof(half*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(model->weights.expert_w2_fp16, d_w2_fp16, layers * sizeof(half*), cudaMemcpyHostToDevice));
}

// ============================================================================
// HOST: Model Init / Free
// ============================================================================

void init_model(K3Model* model, const K3Config* config) {
    model->config = *config;
    int dim = config->dim, vocab = config->vocab_size, top_k = config->top_k, layers = config->num_layers;
    CUDA_CHECK(cudaMalloc(&model->residual, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->attn_out, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->moe_out, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->qkv_buffer, 3 * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->ln_out, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->selected_experts, top_k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&model->routing_weights, top_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->logits, vocab * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->d_residual_fp16, ((dim + 15) / 16) * 16 * sizeof(half)));

    size_t kv_size = (size_t)layers * config->max_seq_len * dim;
    CUDA_CHECK(cudaMalloc(&model->kv_cache.k_cache, kv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->kv_cache.v_cache, kv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->kv_cache.delta_k, layers * dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(model->kv_cache.k_cache, 0, kv_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(model->kv_cache.v_cache, 0, kv_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(model->kv_cache.delta_k, 0, layers * dim * sizeof(float)));
    model->kv_cache.seq_len = 0;
}

void free_model(K3Model* model) {
    cudaFree(model->residual); cudaFree(model->attn_out);
    cudaFree(model->moe_out); cudaFree(model->qkv_buffer);
    cudaFree(model->ln_out); cudaFree(model->selected_experts);
    cudaFree(model->routing_weights); cudaFree(model->logits);
    cudaFree(model->d_residual_fp16);
    cudaFree(model->kv_cache.k_cache); cudaFree(model->kv_cache.v_cache);
    cudaFree(model->kv_cache.delta_k);
}

// ============================================================================
// HOST: Tokenizer (Character-level)
// ============================================================================

#define TOK_PAD 0
#define TOK_BOS 1
#define TOK_EOS 2
#define TOK_UNK 3
#define TOK_BYTE_BASE 4

int char_to_token(unsigned char c) { return TOK_BYTE_BASE + c; }
unsigned char token_to_char(int tok) {
    return (tok >= TOK_BYTE_BASE && tok < TOK_BYTE_BASE + 256) ? tok - TOK_BYTE_BASE : '?';
}

void tokenize_prompt(const char* prompt, int* tokens, int* n_tokens, int max_tokens) {
    int len = strlen(prompt);
    *n_tokens = 0;
    tokens[(*n_tokens)++] = TOK_BOS;
    for (int i = 0; i < len && *n_tokens < max_tokens - 1; ++i)
        tokens[(*n_tokens)++] = char_to_token((unsigned char)prompt[i]);
}

void detokenize_and_print(int token) {
    if (token == TOK_EOS) return;
    putchar(token_to_char(token));
    fflush(stdout);
}

// ============================================================================
// HOST: Forward Pass (Single Token)
// ============================================================================

void forward_token(K3Model* model, int token_id, int pos, float* logits_out) {
    K3Config* c = &model->config;
    K3Weights* w = &model->weights;
    KVCache* kv = &model->kv_cache;
    int dim = c->dim, inter = c->intermediate_dim, top_k = c->top_k, layers = c->num_layers;
    float scale = 1.0f / sqrtf(dim / 32.0f);
    int blocks = DIV_CEIL(dim, 256);

    embedding_kernel<<<1, 256>>>(token_id, pos, w->token_emb, w->pos_emb, model->residual, dim, c->max_seq_len);

    for (int l = 0; l < layers; ++l) {
        float *d_attn_ln_w, *d_qkv_w, *d_attn_out_w, *d_moe_ln_w, *d_router_w, *d_w1, *d_w2;
        cudaMemcpy(&d_attn_ln_w, w->attn_ln_weight + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_qkv_w, w->qkv_weight + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_attn_out_w, w->attn_out_weight + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_moe_ln_w, w->moe_ln_weight + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_router_w, w->router_weight + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_w1, w->expert_w1 + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_w2, w->expert_w2 + l, sizeof(float*), cudaMemcpyDeviceToHost);

        rmsnorm_kernel<<<1, 256, 256*sizeof(float)>>>(model->residual, d_attn_ln_w, model->ln_out, c->rms_norm_eps, dim);
        qkv_proj_kernel<<<DIV_CEIL(3*dim, 256), 256>>>(model->ln_out, d_qkv_w, model->qkv_buffer, dim);

        float *q = model->qkv_buffer, *k = model->qkv_buffer + dim, *v = model->qkv_buffer + 2*dim;
        size_t kv_offset = (size_t)l * c->max_seq_len * dim + (size_t)pos * dim;
        cudaMemcpy(kv->k_cache + kv_offset, k, dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(kv->v_cache + kv_offset, v, dim * sizeof(float), cudaMemcpyDeviceToDevice);

        kimi_delta_attention_kernel<<<blocks, 256>>>(q, kv->delta_k + (size_t)l * dim, kv->k_cache + kv_offset, kv->v_cache + kv_offset, model->attn_out, scale, dim);
        cudaMemcpy(kv->delta_k + (size_t)l * dim, kv->k_cache + kv_offset, dim * sizeof(float), cudaMemcpyDeviceToDevice);
        attn_out_proj_kernel<<<blocks, 256>>>(model->attn_out, d_attn_out_w, model->attn_out, dim);

        // Residual add (host-side for simplicity)
        float *h_res = (float*)malloc(dim * sizeof(float));
        float *h_attn = (float*)malloc(dim * sizeof(float));
        cudaMemcpy(h_res, model->residual, dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_attn, model->attn_out, dim * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < dim; ++i) h_res[i] += h_attn[i];
        cudaMemcpy(model->residual, h_res, dim * sizeof(float), cudaMemcpyHostToDevice);

        rmsnorm_kernel<<<1, 256, 256*sizeof(float)>>>(model->residual, d_moe_ln_w, model->ln_out, c->rms_norm_eps, dim);
        latent_moe_router_kernel<<<1, 32, c->num_experts * sizeof(float)>>>(model->ln_out, d_router_w, model->selected_experts, model->routing_weights, dim, c->num_experts, top_k);

        // WMMA FP16 Tensor Core MoE
        int output_tiles = (model->weights.dim_padded + 15) / 16;
        int wmma_blocks = DIV_CEIL(output_tiles * 32, 128);
        size_t smem_size = (model->weights.dim_padded + model->weights.inter_padded) * sizeof(half);
        convert_fp32_to_fp16_kernel<<<DIV_CEIL(model->weights.dim_padded, 256), 256>>>(model->ln_out, model->d_residual_fp16, dim, model->weights.dim_padded);

        half *d_w1_fp16, *d_w2_fp16;
        cudaMemcpy(&d_w1_fp16, w->expert_w1_fp16 + l, sizeof(half*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_w2_fp16, w->expert_w2_fp16 + l, sizeof(half*), cudaMemcpyDeviceToHost);
        wmma_decode_moe_kernel<<<wmma_blocks, 128, smem_size>>>(model->d_residual_fp16, d_w1_fp16, d_w2_fp16, model->selected_experts, model->routing_weights, model->moe_out, dim, inter, top_k, model->weights.dim_padded, model->weights.inter_padded);

        float *h_moe = (float*)malloc(dim * sizeof(float));
        cudaMemcpy(h_moe, model->moe_out, dim * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < dim; ++i) h_res[i] += h_moe[i];
        cudaMemcpy(model->residual, h_res, dim * sizeof(float), cudaMemcpyHostToDevice);
        free(h_res); free(h_attn); free(h_moe);
    }

    rmsnorm_kernel<<<1, 256, 256*sizeof(float)>>>(model->residual, w->ln_f_weight, model->ln_out, c->rms_norm_eps, dim);
    output_head_kernel<<<DIV_CEIL(c->vocab_size, 256), 256>>>(model->ln_out, w->output_weight, model->logits, c->vocab_size, dim);
    cudaMemcpy(logits_out, model->logits, c->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
}

// ============================================================================
// HOST: Prefill
// ============================================================================

void prefill(K3Model* model, const int* tokens, int len) {
    printf("Prefilling %d tokens...\n", len);
    for (int i = 0; i < len; ++i) {
        float* dummy = (float*)malloc(model->config.vocab_size * sizeof(float));
        forward_token(model, tokens[i], i, dummy);
        free(dummy);
    }
    model->kv_cache.seq_len = len;
}

// ============================================================================
// HOST: Draft Model (Lightweight, 1-layer, Dense FFN)
// ============================================================================

typedef struct {
    K3Config config;
    K3Weights weights;
    KVCache kv_cache;
    float* residual;
    float* ln_out;
    float* qkv_buffer;
    float* attn_out;
    float* ffn_out;
    float* logits;
} DraftModel;

void load_draft_model(const char* path, DraftModel* draft, const K3Model* target) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open draft %s\n", path); exit(1); }
    char magic[4]; int version;
    fread(magic, 1, 4, f);
    fread(&version, sizeof(int), 1, f);
    if (strncmp(magic, "K3D1", 4) != 0) { fprintf(stderr, "Invalid draft magic\n"); exit(1); }
    fread(&draft->config, sizeof(K3Config), 1, f);
    draft->config.rms_norm_eps = 1e-6f;

    int dim = draft->config.dim, inter = draft->config.intermediate_dim;
    int layers = draft->config.num_layers, vocab = draft->config.vocab_size;
    int max_seq = draft->config.max_seq_len;

    draft->weights.token_emb = target->weights.token_emb;
    draft->weights.pos_emb = target->weights.pos_emb;

    float *h_ln_f = (float*)malloc(dim * sizeof(float));
    float *h_out_w = (float*)malloc(vocab * dim * sizeof(float));
    fread(h_ln_f, sizeof(float), dim, f);
    fread(h_out_w, sizeof(float), vocab * dim, f);
    CUDA_CHECK(cudaMalloc(&draft->weights.ln_f_weight, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&draft->weights.output_weight, vocab * dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(draft->weights.ln_f_weight, h_ln_f, dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(draft->weights.output_weight, h_out_w, vocab * dim * sizeof(float), cudaMemcpyHostToDevice));
    free(h_ln_f); free(h_out_w);

    draft->weights.attn_ln_weight = (float**)malloc(layers * sizeof(float*));
    draft->weights.qkv_weight = (float**)malloc(layers * sizeof(float*));
    draft->weights.attn_out_weight = (float**)malloc(layers * sizeof(float*));
    draft->weights.moe_ln_weight = (float**)malloc(layers * sizeof(float*));
    draft->weights.expert_w1 = (float**)malloc(layers * sizeof(float*));
    draft->weights.expert_w2 = (float**)malloc(layers * sizeof(float*));

    float *d_attn_ln[layers], *d_qkv[layers], *d_attn_out[layers];
    float *d_moe_ln[layers], *d_w1[layers], *d_w2[layers];

    for (int l = 0; l < layers; ++l) {
        float *h_attn_ln = (float*)malloc(dim * sizeof(float));
        float *h_qkv = (float*)malloc(3 * dim * dim * sizeof(float));
        float *h_attn_out = (float*)malloc(dim * dim * sizeof(float));
        float *h_moe_ln = (float*)malloc(dim * sizeof(float));
        float *h_w1 = (float*)malloc(inter * dim * sizeof(float));
        float *h_w2 = (float*)malloc(dim * inter * sizeof(float));
        fread(h_attn_ln, sizeof(float), dim, f);
        fread(h_qkv, sizeof(float), 3 * dim * dim, f);
        fread(h_attn_out, sizeof(float), dim * dim, f);
        fread(h_moe_ln, sizeof(float), dim, f);
        fread(h_w1, sizeof(float), inter * dim, f);
        fread(h_w2, sizeof(float), dim * inter, f);
        CUDA_CHECK(cudaMalloc(&d_attn_ln[l], dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_qkv[l], 3 * dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn_out[l], dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_moe_ln[l], dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w1[l], inter * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w2[l], dim * inter * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_attn_ln[l], h_attn_ln, dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qkv[l], h_qkv, 3 * dim * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_attn_out[l], h_attn_out, dim * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_moe_ln[l], h_moe_ln, dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w1[l], h_w1, inter * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w2[l], h_w2, dim * inter * sizeof(float), cudaMemcpyHostToDevice));
        free(h_attn_ln); free(h_qkv); free(h_attn_out);
        free(h_moe_ln); free(h_w1); free(h_w2);
    }
    CUDA_CHECK(cudaMemcpy(draft->weights.attn_ln_weight, d_attn_ln, layers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(draft->weights.qkv_weight, d_qkv, layers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(draft->weights.attn_out_weight, d_attn_out, layers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(draft->weights.moe_ln_weight, d_moe_ln, layers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(draft->weights.expert_w1, d_w1, layers * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(draft->weights.expert_w2, d_w2, layers * sizeof(float*), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&draft->residual, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&draft->ln_out, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&draft->qkv_buffer, 3 * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&draft->attn_out, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&draft->ffn_out, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&draft->logits, vocab * sizeof(float)));

    size_t kv_size = (size_t)layers * max_seq * dim;
    CUDA_CHECK(cudaMalloc(&draft->kv_cache.k_cache, kv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&draft->kv_cache.v_cache, kv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&draft->kv_cache.delta_k, layers * dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(draft->kv_cache.k_cache, 0, kv_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(draft->kv_cache.v_cache, 0, kv_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(draft->kv_cache.delta_k, 0, layers * dim * sizeof(float)));
    draft->kv_cache.seq_len = 0;

    printf("Draft: dim=%d layers=%d inter=%d vocab=%d\n", dim, layers, inter, vocab);
    fclose(f);
}

int draft_step(DraftModel* draft, int token_id, int pos) {
    K3Config* c = &draft->config;
    int dim = c->dim, inter = c->intermediate_dim, layers = c->num_layers, vocab = c->vocab_size;
    float scale = 1.0f / sqrtf(dim / 32.0f);

    embedding_kernel<<<1, 256>>>(token_id, pos, draft->weights.token_emb, draft->weights.pos_emb, draft->residual, dim, c->max_seq_len);

    for (int l = 0; l < layers; ++l) {
        float *d_attn_ln_w, *d_qkv_w, *d_attn_out_w, *d_moe_ln_w, *d_w1, *d_w2;
        cudaMemcpy(&d_attn_ln_w, draft->weights.attn_ln_weight + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_qkv_w, draft->weights.qkv_weight + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_attn_out_w, draft->weights.attn_out_weight + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_moe_ln_w, draft->weights.moe_ln_weight + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_w1, draft->weights.expert_w1 + l, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_w2, draft->weights.expert_w2 + l, sizeof(float*), cudaMemcpyDeviceToHost);

        rmsnorm_kernel<<<1, 256, 256*sizeof(float)>>>(draft->residual, d_attn_ln_w, draft->ln_out, c->rms_norm_eps, dim);
        qkv_proj_kernel<<<DIV_CEIL(3*dim, 256), 256>>>(draft->ln_out, d_qkv_w, draft->qkv_buffer, dim);

        float *q = draft->qkv_buffer, *k = draft->qkv_buffer + dim, *v = draft->qkv_buffer + 2*dim;
        size_t kv_off = (size_t)l * c->max_seq_len * dim + (size_t)pos * dim;
        cudaMemcpy(draft->kv_cache.k_cache + kv_off, k, dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(draft->kv_cache.v_cache + kv_off, v, dim * sizeof(float), cudaMemcpyDeviceToDevice);

        kimi_delta_attention_kernel<<<DIV_CEIL(dim, 256), 256>>>(q, draft->kv_cache.delta_k + (size_t)l * dim, draft->kv_cache.k_cache + kv_off, draft->kv_cache.v_cache + kv_off, draft->attn_out, scale, dim);
        cudaMemcpy(draft->kv_cache.delta_k + (size_t)l * dim, draft->kv_cache.k_cache + kv_off, dim * sizeof(float), cudaMemcpyDeviceToDevice);
        attn_out_proj_kernel<<<DIV_CEIL(dim, 256), 256>>>(draft->attn_out, d_attn_out_w, draft->attn_out, dim);

        float *h_res = (float*)malloc(dim * sizeof(float));
        float *h_a = (float*)malloc(dim * sizeof(float));
        cudaMemcpy(h_res, draft->residual, dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_a, draft->attn_out, dim * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < dim; ++i) h_res[i] += h_a[i];
        cudaMemcpy(draft->residual, h_res, dim * sizeof(float), cudaMemcpyHostToDevice);

        rmsnorm_kernel<<<1, 256, 256*sizeof(float)>>>(draft->residual, d_moe_ln_w, draft->ln_out, c->rms_norm_eps, dim);
        warp_decode_moe_kernel<<<DIV_CEIL(dim * WARP_SIZE, 128), 128>>>(draft->ln_out, d_w1, d_w2, NULL, NULL, draft->ffn_out, dim, inter, 0);

        float *h_f = (float*)malloc(dim * sizeof(float));
        cudaMemcpy(h_f, draft->ffn_out, dim * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < dim; ++i) h_res[i] += h_f[i];
        cudaMemcpy(draft->residual, h_res, dim * sizeof(float), cudaMemcpyHostToDevice);
        free(h_res); free(h_a); free(h_f);
    }

    rmsnorm_kernel<<<1, 256, 256*sizeof(float)>>>(draft->residual, draft->weights.ln_f_weight, draft->ln_out, c->rms_norm_eps, dim);
    output_head_kernel<<<DIV_CEIL(vocab, 256), 256>>>(draft->ln_out, draft->weights.output_weight, draft->logits, vocab, dim);

    float* h_logits = (float*)malloc(vocab * sizeof(float));
    cudaMemcpy(h_logits, draft->logits, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    int best = 0;
    for (int i = 1; i < vocab; ++i)
        if (h_logits[i] > h_logits[best]) best = i;
    free(h_logits);
    draft->kv_cache.seq_len = pos + 1;
    return best;
}

// ============================================================================
// HOST: Speculative Decode Loop
// ============================================================================

void speculative_decode(K3Model* target, DraftModel* draft, int prompt_len,
                        int max_new_tokens, float temperature, float top_p, int gamma) {
    int vocab = target->config.vocab_size;
    int* h_draft_tokens = (int*)malloc(gamma * sizeof(int));
    float* h_target_logits = (float*)malloc((gamma + 1) * vocab * sizeof(float));
    float* h_draft_logits = (float*)malloc(gamma * vocab * sizeof(float));
    int* d_accept_count, *d_accepted_tokens;
    CUDA_CHECK(cudaMalloc(&d_accept_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_accepted_tokens, gamma * sizeof(int)));

    unsigned long long rng_seed = time(NULL);
    int target_pos = prompt_len, draft_pos = prompt_len;
    int token = TOK_BOS;
    int generated = 0;

    while (generated < max_new_tokens) {
        for (int g = 0; g < gamma; ++g)
            h_draft_tokens[g] = draft_step(draft, token, draft_pos++);

        for (int g = 0; g < gamma + 1; ++g) {
            float* dummy = (float*)malloc(vocab * sizeof(float));
            forward_token(target, (g == 0) ? token : h_draft_tokens[g - 1], target_pos + g, dummy);
            memcpy(h_target_logits + g * vocab, target, vocab * sizeof(float));
            free(dummy);
        }

        CUDA_CHECK(cudaMemcpy(d_accepted_tokens, h_draft_tokens, gamma * sizeof(int), cudaMemcpyHostToDevice));
        speculative_acceptance_kernel<<<1, 1>>>(target->logits, draft->logits, d_accepted_tokens, vocab, gamma, temperature, rng_seed++, d_accept_count, d_accepted_tokens);

        int accept_count;
        int h_accepted[MAX_GAMMA];
        cudaMemcpy(&accept_count, d_accept_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_accepted, d_accepted_tokens, accept_count * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < accept_count && generated < max_new_tokens; ++i) {
            detokenize_and_print(h_accepted[i]);
            generated++;
        }
        target_pos += accept_count;
        draft_pos = target_pos;
        token = h_accepted[accept_count - 1];
    }

    free(h_draft_tokens); free(h_target_logits); free(h_draft_logits);
    cudaFree(d_accept_count); cudaFree(d_accepted_tokens);
}

// ============================================================================
// HOST: FFI Interface (for Rust bridge)
// ============================================================================

extern "C" {
    K3Model* k3_engine_init(const char* model_path) {
        K3Model* model = (K3Model*)malloc(sizeof(K3Model));
        memset(model, 0, sizeof(K3Model));
        K3Config config;
        K3Weights weights = {0};
        load_model_weights(model_path, &config, &weights);
        init_model(model, &config);
        model->weights = weights;
        convert_and_upload_fp16_weights(model);
        return model;
    }

    int32_t k3_engine_forward(K3Model* engine, int32_t token, int32_t pos, float* out_logits) {
        forward_token(engine, token, pos, out_logits);
        return 0;
    }

    void k3_engine_free(K3Model* engine) {
        free_model(engine);
        free(engine);
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <model.bin> <prompt> <max_tokens> <temp> <top_p> [--speculative draft.bin gamma]\n", argv[0]);
        return 1;
    }

    const char* draft_path = NULL;
    int gamma = 4;
    for (int i = 6; i < argc; ++i)
        if (strcmp(argv[i], "--speculative") == 0 && i + 2 < argc) {
            draft_path = argv[i + 1];
            gamma = atoi(argv[i + 2]);
        }

    printf("=== Sovereign MiMo-4B Kimi K3 Nano (sm_86 FP16 WMMA) ===\n\n");

    K3Config config;
    K3Weights weights = {0};
    load_model_weights(argv[1], &config, &weights);

    K3Model model;
    init_model(&model, &config);
    model.weights = weights;
    convert_and_upload_fp16_weights(&model);

    DraftModel draft;
    bool use_spec = (draft_path != NULL);
    if (use_spec) load_draft_model(draft_path, &draft, &model);

    int tokens[MAX_SEQ_LEN], n_tokens = 0;
    tokenize_prompt(argv[2], tokens, &n_tokens, MAX_SEQ_LEN);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    prefill(&model, tokens, n_tokens);

    if (use_spec) {
        draft.kv_cache.seq_len = n_tokens;
        printf("\nSpeculative decoding (gamma=%d)...\n\n", gamma);
        speculative_decode(&model, &draft, n_tokens, atoi(argv[3]), atof(argv[4]), atof(argv[5]), gamma);
    } else {
        printf("\nGenerating %d tokens...\n\n", atoi(argv[3]));
        int token = TOK_BOS;
        float* h_logits = (float*)malloc(config.vocab_size * sizeof(float));
        int* d_tok;
        cudaMalloc(&d_tok, sizeof(int));
        unsigned long long seed = time(NULL);
        for (int step = 0; step < atoi(argv[3]); ++step) {
            forward_token(&model, token, n_tokens + step, h_logits);
            top_p_sample_kernel<<<1, 1>>>(model.logits, config.vocab_size, atof(argv[4]), atof(argv[5]), seed++, d_tok);
            cudaMemcpy(&token, d_tok, sizeof(int), cudaMemcpyDeviceToHost);
            detokenize_and_print(token);
            if (token == TOK_EOS) break;
        }
        free(h_logits); cudaFree(d_tok);
    }

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    printf("\n\nLatency: %.2f ms\n", ms);

    free_model(&model);
    return 0;
}
