#!/bin/bash
#SBATCH --ntasks=1                # 1 core(CPU)
#SBATCH --nodes=1                 # Use 1 node
#SBATCH --job-name=numba          # sensible name for the job
#SBATCH --mem=2G                  # Default memory per CPU is 3GB.
#SBATCH --partition=COURSE        # CHANGED - to use nodes containing the GPUs in the tutorial session
#SBATCH --gres=gpu:1              # Must have - to use the GPU in the node
#SBATCH --output=outputs/numba-%A.out
#SBATCH --error=outputs/numba-%A.out


# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module

# Load the CUDA driver
module load CUDA/10.1.243-GCC-8.3.0
# Load Conda
module load Miniconda3
# enable conda activate
eval "$(conda shell.bash hook)"

# activate conda then run the python script
conda activate test_gpu

python scripts/run_numba.py

# deactivate when finished
conda deactivate
