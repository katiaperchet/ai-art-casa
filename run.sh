#!/usr/bin/bash

# Set up LD_LIBRARY_PATH for CUDA
export LD_LIBRARY_PATH=/home/millenium-falcon/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/home/millenium-falcon/.local/lib/python3.10/site-packages/nvidia/cublas/lib:/home/millenium-falcon/.local/lib/python3.10/site-packages/nvidia/cudnn/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
jupyter-notebook