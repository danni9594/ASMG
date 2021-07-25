#!/bin/sh
#SBATCH -o ml_asmggru_full.out
#SBATCH -p K20q
#module load cuda90/toolkit
srun python2 train_ml.py
