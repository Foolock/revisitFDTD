./fdtd_seq 32 32 32 100
./fdtd_gpu 32 32 32 100
python3 simplified_FDTD_numpy_with_gpu_compare.py 32 32 32 100 vectorized numpy_figures
