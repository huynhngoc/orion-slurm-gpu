#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=conda_install
#SBATCH --mem=4G
#SBATCH --partition=smallmem
#SBATCH --output=outputs/conda-%A.out
#SBATCH --error=outputs/conda-%A.out

# If you would like to use more please adjust this.

# Load conda module
module load Miniconda3

# create the environment
conda create --name test_gpu -c conda python=3.7 numpy ipython numba cudatoolkit
