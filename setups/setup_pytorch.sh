#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=singularity
#SBATCH --mem=4G
#SBATCH --partition=smallmem
#SBATCH --output=outputs/singularity-%A.out
#SBATCH --error=outputs/singularity-%A.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

if [ ! "pytorch_gpu.sif" ]
  then
  echo "Deleting old files"
  rm -f pytorch_gpu.sif
  fi
singularity build --fakeroot pytorch_gpu.sif singularity/Singularity.PyTorch
