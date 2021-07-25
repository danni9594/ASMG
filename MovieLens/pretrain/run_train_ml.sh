#!/bin/sh
#SBATCH -o ml_pretrain.out
#SBATCH -p RTXq
#module load cuda90/toolkit
srun python2 train_ml.py
