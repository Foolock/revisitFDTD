#!/usr/bin/env bash

set -euo pipefail

NUM_TIMESTEPS=100
REPEATS=3
OUTPUT_CSV="fdtd_benchmark_results.csv"

# Sizes: 32 -> 512, doubling each time
SIZES=(32 64 128 256 512)

# CSV header
echo "pixels,throughput_numpy_indexing,throughput_numpy_vectorized,throughput_numpy_cuda,throughput_cpp_seq,throughput_cpp_cuda" > "$OUTPUT_CSV"

run_and_extract_throughput() {
  local cmd="$1"
  local value

  # Run command, capture output, print it to screen, then extract throughput number
  value=$(
    eval "$cmd" | tee /dev/stderr | \
    grep -E "throughput|performance" | \
    tail -n 1 | \
    awk '{print $(NF-1)}'
  )

  echo "$value"
}

average_three_runs() {
  local cmd="$1"
  local sum=0
  local val

  for ((r=1; r<=REPEATS; r++)); do
    echo "  Run $r: $cmd" >&2
    val=$(run_and_extract_throughput "$cmd")
    sum=$(awk -v s="$sum" -v v="$val" 'BEGIN {printf "%.10f", s+v}')
  done

  awk -v s="$sum" -v n="$REPEATS" 'BEGIN {printf "%.10f", s/n}'
}

for N in "${SIZES[@]}"; do
  PIXELS=$((N * N * N))
  echo "==================================================" >&2
  echo "Benchmarking N=$N, pixels=$PIXELS" >&2
  echo "==================================================" >&2

  THROUGHPUT_NUMPY_INDEXING=$(average_three_runs \
    "python3 simplified_FDTD_numpy_benchmark.py $N $N $N $NUM_TIMESTEPS indexing")

  THROUGHPUT_NUMPY_VECTORIZED=$(average_three_runs \
    "python3 simplified_FDTD_numpy_benchmark.py $N $N $N $NUM_TIMESTEPS vectorized")

  THROUGHPUT_NUMPY_CUDA=$(average_three_runs \
    "python3 simplified_FDTD_numpy_benchmark.py $N $N $N $NUM_TIMESTEPS cuda")

  THROUGHPUT_CPP_SEQ=$(average_three_runs \
    "./fdtd_seq $N $N $N $NUM_TIMESTEPS")

  THROUGHPUT_CPP_CUDA=$(average_three_runs \
    "./fdtd_gpu $N $N $N $NUM_TIMESTEPS")

  echo "${PIXELS},${THROUGHPUT_NUMPY_INDEXING},${THROUGHPUT_NUMPY_VECTORIZED},${THROUGHPUT_NUMPY_CUDA},${THROUGHPUT_CPP_SEQ},${THROUGHPUT_CPP_CUDA}" >> "$OUTPUT_CSV"

  echo "Saved row for N=$N" >&2
done

echo "Done. Results written to $OUTPUT_CSV"
