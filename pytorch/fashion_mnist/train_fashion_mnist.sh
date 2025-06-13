#!/bin/bash

#SBATCH --account=courses01-gpu
#SBATCH --partition=gpu-dev
#SBATCH --job-name=fashion_mnist_trainings
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:20:00


module load pytorch/2.2.0-rocm5.7.3
srun -c 8 python3 main.py --data $MYSCRATCH/ml-data
