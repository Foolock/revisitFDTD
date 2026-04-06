# compilers
CXX      = g++
NVCC     = nvcc

# flags
CXXFLAGS = -O3 -std=c++17
NVFLAGS  = -O3 -fmad=false -std=c++17

# targets
SEQ_TARGET = fdtd_seq
GPU_TARGET = fdtd_gpu

# sources
SEQ_SRC = simplified_FDTD.cpp
GPU_SRC = simplified_FDTD_cuda.cu

# output dirs
SEQ_FIG_DIR = seq_figures_cpp
SEQ_BIN_DIR = seq_fields_cpp
GPU_FIG_DIR = gpu_figures_cpp
GPU_BIN_DIR = gpu_fields_cpp

# default target
all: $(SEQ_TARGET) $(GPU_TARGET)

# build seq
$(SEQ_TARGET): $(SEQ_SRC)
	$(CXX) $(CXXFLAGS) $(SEQ_SRC) -o $(SEQ_TARGET)

# build gpu
$(GPU_TARGET): $(GPU_SRC)
	$(NVCC) $(NVFLAGS) $(GPU_SRC) -o $(GPU_TARGET)

# run seq with optional args: make run_seq ARGS="64 64 64 20"
run_seq: $(SEQ_TARGET)
	rm -rf $(SEQ_FIG_DIR) $(SEQ_BIN_DIR)
	./$(SEQ_TARGET) $(ARGS)

# run gpu with optional args: make run_gpu ARGS="64 64 64 20"
run_gpu: $(GPU_TARGET)
	rm -rf $(GPU_FIG_DIR) $(GPU_BIN_DIR)
	./$(GPU_TARGET) $(ARGS)

# clean everything
clean:
	rm -f $(SEQ_TARGET) $(GPU_TARGET)
	rm -rf $(SEQ_FIG_DIR) $(SEQ_BIN_DIR) $(GPU_FIG_DIR) $(GPU_BIN_DIR)
