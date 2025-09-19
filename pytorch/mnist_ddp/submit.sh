#!/bin/bash

#SBATCH --account=pawsey0001-gpu
#SBATCH --partition=gpu-dev
#SBATCH --job-name=mnist_ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:20:00


module load pytorch/2.7.1-rocm6.3.3
# The following might be needed because later libfabric versions are not supported
# export SINGULARITYENV_LD_PRELOAD=/opt/cray/libfabric/1.15.2.0/lib64/libfabric.so:$SINGULARITYENV_LD_PRELOAD

# Number of PyTorch processes per node to be spawned by torchrun.
# One for each GCD.
NUM_PYTORCH_PROCESSES=8
# Setting the number of threads to be generated for each PyTorch process.
export OMP_NUM_THREADS=8

# The compute node executing the batch script.
export RDZV_HOST=$(hostname)
export RDZV_PORT=29400                   

# Give all the available resources to torchrun which will then distribute them internally
# to the processes it creates. Processes are NOT created by srun! For what srun is concerned,
# only one task is created, the torchrun process.
#export MIOPEN_USER_DB_PATH="$MYSCRATCH/my-miopen-cache"
#export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

srun -c 64 singularity exec  $SINGULARITY_CONTAINER torchrun --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NUM_PYTORCH_PROCESSES\
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    mnist.py --epochs 100

