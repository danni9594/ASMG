#!/bin/sh
#SBATCH -o ml_incctr.out
#SBATCH -p DGXq
#module load cuda90/toolkit
srun python2 train_ml.py
