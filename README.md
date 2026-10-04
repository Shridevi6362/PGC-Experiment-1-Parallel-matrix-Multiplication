# PGC-Experiment-1-Parallel-matrix-Multiplication
# Parallel Matrix Multiplication Benchmarks

A comprehensive laboratory experiment implementing a $4000 \times 4000$ dense matrix multiplication using four distinct computing paradigms: Sequential CPU, Shared-Memory Multithreading (OpenMP), Distributed-Memory Cluster Processing (MPI), and Massively Parallel GPU Acceleration (CUDA).

---

## Overview & Architecture

This repository contains full source code, setup instructions, and comparative benchmark analysis for computing $C = A \times B$ where $A, B \in \mathbb{R}^{4000 \times 4000}$. Every element in input matrices $A$ and $B$ is initialized to `1.0`. The mathematical baseline ensures that every element of the resulting output matrix $C$ evaluates to `4000.00`, enabling unified verification across all compute platforms.

## Benchmark Results & Performance Comparison
The performance metrics below were recorded for a $4000 \times 4000$ double/single-precision matrix multiplication. All four implementations successfully verified numerical correctness with $C[0][0] = 4000.00$.   

### Implementation Overview

| Implementation | Compute Model | Hardware / Compute Resources | Execution Time | Verification |
| :--- | :--- | :--- | :---: | :---: |
| **Sequential** | Single CPU execution | 1 CPU Core | 244.120000 s | 4000.00 |
| **OpenMP** | Shared memory | 8 CPU Threads | 30.830434 s | 4000.00 |
| **MPI** | Distributed memory | 4 Nodes / Virtual Machines | 92.979510 s | 4000.00 |
| **CUDA** | GPU Parallelism | NVIDIA RTX 4500 Ada | 0.165004 s | 4000.00 |

Speedup AnalysisCalculated using the standard speedup formula:

$$\text{Speedup} = \frac{T_{\text{Sequential}}}{T_{\text{Parallel}}}$$

### Performance & Speedup Comparison

| Implementation | Execution Time | Speedup Factor |
| :--- | :---: | :---: |
| **Sequential Baseline** | 244.120000 s | 1.00× |
| **OpenMP (8 Threads)** | 30.830434 s | 7.92× |
| **MPI (4 Processes)** | 92.979510 s | 2.63× |
| **CUDA (GPU Total)** | 0.165004 s | 1479.48× |

## System Prerequisites OS:
Windows 10/11 with WSL2 (Ubuntu) or native Linux environment.   CPU Tools: gcc, make, build-essential package.   MPI Cluster: Open MPI (openmpi-bin, libopenmpi-dev), OpenSSH Server (openssh-server).   CUDA GPU: NVIDIA CUDA Toolkit (nvcc), compatible GPU driver.   

## Part A: Sequential Baseline
Single-core execution serving as the baseline for speedup calculations.   

Compilation & Execution
cd sequential
gcc -O2 matrix_sequential.c -o matrix_sequential
./matrix_sequential

## Part B: OpenMP Shared-Memory Parallelism
Parallelizes the outer loop of the matrix multiplication using OpenMP directives across multiple threads on a shared-memory CPU.  

Environment Setup & Execution
cd openmp
nproc
export OMP_NUM_THREADS=8
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
./matrix_openmp

## Part C: MPI Distributed Cluster
Distributes matrix rows across 4 independent cluster nodes (1 Master, 3 Workers) using scatter/gather routines over network sockets

Network Cluster Topography
### Cluster Node Configuration

| Node Hostname | IP Address | Cluster Role | Workload Allocation |
| :--- | :--- | :--- | :--- |
| **master** | 192.168.125.128 | Rank 0 (Master) | 1000 Rows |
| **worker1** | 192.168.125.129 | Rank 1 (Worker) | 1000 Rows |
| **worker2** | 192.168.125.130 | Rank 2 (Worker) | 1000 Rows |
| **worker3** | 192.168.125.131 | Rank 3 (Worker) | 1000 Rows |

## Configuration & Execution
1.Configure hosts file on the Master node:
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1

2.Enable passwordless SSH from master to all worker nodes (ssh-copy-id worker1, etc.). 

3.Compile, distribute binary, and launch distributed run:   

cd mpi
mpicc -O2 matrix_mpi.c -o matrix_mpi

Synchronize executable to worker nodes

scp matrix_mpi worker1:~/matrix_mpi

scp matrix_mpi worker2:~/matrix_mpi

scp matrix_mpi worker3:~/matrix_mpi

Run across cluster

mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'

## Part D: CUDA GPU Acceleration
Offloads the computational grid onto GPU hardware, distributing execution across $16,000,000$ logical threads ($250 \times 250$ blocks, $16 \times 16$ threads per block).  
Grid Execution Parameters

1.Matrix Dimension($N \times N$): $4000 \times 4000$ 

2.Block Dimensions: $16 \times 16$ ($256$ threads/block)  

3.Grid Dimensions: $250 \times 250$ ($62,500$ blocks)   

4.Total Threads: $16,000,000$ threads   

##Compilation & Execution

cd cuda

## Verify GPU detection and nvcc compiler

nvidia-smi
nvcc --version

## Compile and run CUDA program

nvcc -O2 matrix_cuda.cu -o matrix_cuda
./matrix_cuda

### Troubleshooting Guide

| Issue / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| **`gcc: command not found`** | Missing build environment in WSL/Ubuntu | Run `sudo apt update && sudo apt install build-essential -y`. |
| **OpenMP running on 1 thread** | Missing `-fopenmp` flag or unset thread variable | Ensure compilation includes `-fopenmp` and run `export OMP_NUM_THREADS=8`. |
| **SSH password prompts in MPI** | Missing host public keys on worker nodes | Run `ssh-keygen -t rsa` and copy keys via `ssh-copy-id workerX`. |
| **`mpirun` host unreachable** | Misconfigured network interfaces or local firewall | Verify host IP addresses using `hostname -I` and confirm `ping <worker-ip>` works[cite: 1]. |
| **`nvcc: command not found`**[cite: 1] | CUDA toolkit not in environment `PATH`[cite: 1] | Install CUDA toolkit and export `PATH`/`LD_LIBRARY_PATH` environment variables[cite: 1]. |

