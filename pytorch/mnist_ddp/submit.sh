#!/bin/bash

#SBATCH --account=pawsey0001-gpu
#SBATCH --partition=gpu-dev
#SBATCH --job-name=mnist_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:20:00


module load pytorch/2.2.0-rocm5.7.3
# The following module swap is needed due to a bug in the PyTorch module.
module load singularity/3.11.4-nohost

# Number of PyTorch processes per node to be spawned by torchrun.
# One for each GCD.
NUM_PYTORCH_PROCESSES=8
# Setting the number of threads to be generated for each PyTorch process.
export OMP_NUM_THREADS=8

# The computw node executing the batch script.
export RDZV_HOST=$(hostname)
export RDZV_PORT=29400                   

# Give all the available resources to torchrun which will then distribute them internally
# to the processes it creates. Processes are NOT created by srun! For what srun is concerned,
# only one task is created, the torchrun process.

srun -c 64 singularity exec  $SINGULARITY_CONTAINER torchrun --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NUM_PYTORCH_PROCESSES\
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    mnist.py --epochs 100

