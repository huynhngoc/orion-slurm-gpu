#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=data
#SBATCH --mem=16G
#SBATCH --partition=COURSE        # CHANGED - partition for the tutorial session
#SBATCH --output=outputs/data-%A.out
#SBATCH --error=outputs/data-%A.out

# If you would like to use more please adjust this.

# Load conda module
module load Miniconda3

# create the environment
conda create --name $USER_gpu python=3.7 numpy scipy pandas scikit-learn matplotlib ipython numba cudatoolkit
