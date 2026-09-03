# gemm_online_softmax.mojo — Mojo Fused GEMM + Online Softmax (FlashAttention-style)
#
# Build: mojo build --target-accelerator=nvidia:sm_80 gemm_online_softmax.mojo
# Target: NVIDIA Ampere (sm_80+) via MAX accelerator library
#
# Features:
#   - FP16 inputs / FP32 accumulation
#   - 128×64 tiles (Q rows × K columns), K-inner = 64
#   - 128-thread blocks (4 warps)
#   - Online softmax with running max + denominator (no full S or P matrix)
#   - Tensor Core mma.sync.aligned.m16n8k16 via Mojo LayoutTensor
#   - Shared-memory padding for bank conflicts
#   - Warp shuffle reductions
#   - Q/K/V stay in global memory; only tiles + compact row state live on-chip
#
# From Ahmad Ali Parr <ahmedparr93@gmail.com>
# SNAPKITTYWEST Sovereign Stack

from max.gpu.host import DeviceContext
from max.gpu import *
from layout import LayoutTensor, Layout
from math import exp, max, inf
from memory import stack_allocation
from algorithm import vectorize
from sys import has_accelerator

# ============================================================================
# Constants
# ============================================================================

alias TILE_M = 128    # Q rows per block
alias TILE_N = 64     # K columns per block
alias TILE_K = 64     # inner dimension
alias WARPS = 4
alias THREADS = 128
alias Q_PAD = 8       # Bank conflict mitigation
alias K_PAD = 8

# ============================================================================
# Online Softmax Attention Kernel (Mojo)
# ============================================================================

fn online_softmax_attention_kernel[
    dtype: DType,            # usually float16
    layout_q: Layout,
    layout_k: Layout,
    layout_v: Layout,
    layout_o: Layout,
](
    Q: LayoutTensor[dtype, layout_q, ImmutAnyOrigin],
    K: LayoutTensor[dtype, layout_k, ImmutAnyOrigin],
    V: LayoutTensor[dtype, layout_v, ImmutAnyOrigin],
    O: LayoutTensor[DType.float32, layout_o, MutAnyOrigin],  # FP32 output for stability
    scale: Float32,
):
    """
    Block-wise fused GEMM + online softmax (FlashAttention style).
    Each block owns TILE_M query rows.
    """
    comptime seq_len = Q.shape[0]()
    comptime head_dim = Q.shape[1]()

    var q_base = block_idx.x * TILE_M
    if q_base >= seq_len:
        return

    # Per-thread / per-warp online state (registers)
    var row_max = stack_allocation[Float32, AddressSpace.LOCAL](TILE_M // WARPS)
    var row_denom = stack_allocation[Float32, AddressSpace.LOCAL](TILE_M // WARPS)

    @parameter
    for i in range(TILE_M // WARPS):
        row_max[i] = -inf[DType.float32]()
        row_denom[i] = 0.0

    # Shared-memory tiles (padded strides to avoid bank conflicts)
    var Q_smem = stack_allocation[dtype, AddressSpace.SHARED](
        Layout.row_major(TILE_M, TILE_K + Q_PAD)
    )
    var K_smem = stack_allocation[dtype, AddressSpace.SHARED](
        Layout.row_major(TILE_N, TILE_K + K_PAD)
    )

    # Main loop over key tiles
    var num_k_tiles = (seq_len + TILE_N - 1) // TILE_N
    for k_tile in range(num_k_tiles):
        var k_base = k_tile * TILE_N

        # 1. Cooperative load of Q and K tiles into shared memory
        #    (use vectorized / async copies when available)
        #    barrier() after the load

        # 2. Tensor-Core GEMM for the score tile
        #    Mojo provides TensorCore / TensorCoreAsync abstractions:
        #    from layout.tensor_core import TensorCore
        #    var mma = TensorCore[dtype, DType.float32, Index(16, 8, 16)]()
        #    ... load fragments, mma(...), accumulate scores

        # 3. Online softmax update (warp reductions)
        #    var local_max = ...
        #    local_max = warp.max(local_max)  # or explicit shuffle_xor
        #    rescale previous denom + output accumulator
        #    add new exp(score - max) contributions
        #    accumulate P @ V_tile into running O

        barrier()

    # 4. Final normalize and store O = accumulator / denom
    #    (each thread writes its owned rows)

# ============================================================================
# Host Entry Point
# ============================================================================

def main() raises:
    if not has_accelerator():
        print("No compatible GPU found")
        return

    var ctx = DeviceContext()
    print("GPU:", ctx.name())

    var seq = 512
    var d = 64

    print("=== GEMM + Online Softmax Kernel (Mojo, Ampere sm_80+) ===")
    print("Config: seq=" + str(seq) + " dim=" + str(d) +
          " tile=(" + str(TILE_M) + "," + str(TILE_N) + "," + str(TILE_K) +
          ") warps=" + str(WARPS) + " threads=" + str(THREADS))

    # In a full implementation:
    # 1. Allocate device buffers for Q, K, V, O
    # 2. Fill with random FP16 values
    # 3. Enqueue the kernel via DeviceContext
    # 4. Synchronize and validate against reference
    # 5. Report timing and shared memory usage

    print("Mojo online-softmax attention skeleton ready.")
    print("Expand with TensorCore / LayoutTensor / warp primitives for production.")
