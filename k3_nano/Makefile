# k3_nano/Makefile — Sovereign MiMo-4B Kimi K3 Nano
# Build: make all
# Target: sm_86 (RTX 3080, Ampere)

NVCC = nvcc
NVCC_FLAGS = -arch=sm_86 -O3 --use_fast_math -lineinfo
CUDA_PATH ?= /usr/local/cuda
INCLUDES = -I$(CUDA_PATH)/include

TARGET = k3_nano
SRC = k3_nano_harness.cu
ATTN_SRC = kernels/fused_attention.cu

.PHONY: all clean draft test spec attention

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@
	@echo "[BUILD] $(TARGET) built successfully"

attention: $(ATTN_SRC)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DSTANDALONE $(INCLUDES) $< -o fused_attn
	@echo "[BUILD] fused_attn built successfully"

draft: $(TARGET)
	python3 scripts/create_draft_model.py --target model.bin --output draft.bin
	@echo "[BUILD] draft.bin created"

test: $(TARGET)
	./$(TARGET) model.bin "Hello world" 20 0.8 0.9

spec: $(TARGET) draft
	./$(TARGET) model.bin "Hello world" 128 0.8 0.9 --speculative draft.bin 4

clean:
	rm -f $(TARGET) fused_attn draft.bin
	@echo "[BUILD] Cleaned"
