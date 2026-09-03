# kernels/Makefile — Sovereign MiMo-4B CUDA Kernels
# Build: make all
# Target: sm_86 (RTX 3080, Ampere)

NVCC = nvcc
NVCC_FLAGS = -arch=sm_86 -O3 --use_fast_math -lineinfo
CUDA_PATH ?= /usr/local/cuda

INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -lcudart

SRCS = qra_router.cu rmsnorm.cu rope.cu swiglu.cu fused_attention.cu
OBJS = $(SRCS:.cu=.o)
LIBS = libmimo_kernels.a

.PHONY: all clean

all: $(LIBS)

$(LIBS): $(OBJS)
	ar rcs $@ $^
	@echo "[BUILD] $(LIBS) built successfully"

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIBS)
	@echo "[BUILD] Cleaned"

# Module build (for Python torch.utils.cpp_extension)
torch:
	python -c "from torch.utils.cpp_extension import load; \
	kernels = load('mimo_kernels', \
		sources=['qra_router.cu','rmsnorm.cu','rope.cu','swiglu.cu','fused_attention.cu'], \
		verbose=True); \
	print('Torch JIT compiled:', kernels)"
