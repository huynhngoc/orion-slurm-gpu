#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=data
#SBATCH --mem=4G
#SBATCH --partition=smallmem
#SBATCH --output=outputs/data-%A.out
#SBATCH --error=outputs/data-%A.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

singularity exec --nv tensorflow_gpu.sif python scripts/setup_data.py
