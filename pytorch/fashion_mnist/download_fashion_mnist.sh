#!/bin/bash

#SBATCH --partition=copy
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --account=courses01

data_dir=$MYSCRATCH/ml-data/FashionMNIST/raw
mkdir -p $data_dir
cd $data_dir 
srun -n 1 wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz &
srun -n 1 wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz &
srun -n 1 wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz &
srun -n 1 wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz &

wait

gunzip *.gz

