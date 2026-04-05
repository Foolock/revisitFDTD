# compiler
CXX = g++
CXXFLAGS = -O3 -std=c++17

# target
TARGET = fdtd_seq
SRC = simplified_FDTD.cpp

# output dirs
FIG_DIR = seq_figures_cpp
BIN_DIR = seq_fields_cpp

# default target
all: $(TARGET)

# build
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# run with optional args: make run ARGS="64 64 64 20"
run: $(TARGET)
	rm -rf $(FIG_DIR) $(BIN_DIR)
	./$(TARGET) $(ARGS)

# clean everything
clean:
	rm -f $(TARGET)
	rm -rf $(FIG_DIR) $(BIN_DIR)
