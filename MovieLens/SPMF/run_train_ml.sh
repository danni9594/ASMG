#!/bin/sh
#SBATCH -o ml_spmf.out
#SBATCH -p K20q
#module load cuda90/toolkit
srun python2 train_ml.py
