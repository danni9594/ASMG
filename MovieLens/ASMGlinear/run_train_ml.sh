#!/bin/sh
#SBATCH -o ml_asmglinear.out
#SBATCH -p PP1004q
#module load cuda90/toolkit
srun python2 train_ml.py
