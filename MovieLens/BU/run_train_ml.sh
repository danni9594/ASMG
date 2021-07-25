#!/bin/sh
#SBATCH -o ml_bu_7.out
#SBATCH -p RTXq
#module load cuda90/toolkit
srun python2 train_ml.py
