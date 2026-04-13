# compilers
NVCC     = nvcc

# flags
NVFLAGS  = -O3 -fmad=false -std=c++17

# targets
GPU_TARGET = fdtd_cuda_cpp 

# sources
GPU_SRC = simplified_FDTD_cuda.cu

# default target
all: $(GPU_TARGET)

# build gpu
$(GPU_TARGET): $(GPU_SRC)
	$(NVCC) $(NVFLAGS) $(GPU_SRC) -o $(GPU_TARGET)

# clean everything
clean:
	rm -f $(GPU_TARGET)
