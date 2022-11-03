#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=cpu_tf
#SBATCH --mem=16G
#SBATCH --partition=smallmem
#SBATCH --output=outputs/cpu-%A.out
#SBATCH --error=outputs/cpu-%A.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

# Run experiment
singularity exec --nv tensorflow_gpu.sif python scripts/run_tensorflow_xs.py
