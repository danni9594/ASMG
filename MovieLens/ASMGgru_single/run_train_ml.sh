#!/bin/sh
#SBATCH -o ml_asmggru_single.out
#SBATCH -p NV100q
#module load cuda90/toolkit
srun python2 train_ml.py
