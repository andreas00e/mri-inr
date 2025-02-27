#!/usr/bin/bash

#SBATCH -J "mri_inr"   # job name
#SBATCH --time=5-00:00:00   # walltime
#SBATCH --output=/vol/aimspace/projects/practical_SoSe24/mri_inr/logs/train_%A.out  # Standard output of the script (Can be absolute or relative path)
#SBATCH --error=/vol/aimspace/projects/practical_SoSe24/mri_inr/logs/train_%A.err  # Standard error of the script
#SBATCH --mem=32G
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # number of processor cores (i.e. tasks)
#SBATCH --gres=gpu:1  # replace 0 with 1 if gpu needed
#SBATCH --partition=course

# load python module
. "/opt/anaconda3/etc/profile.d/conda.sh"

# activate corresponding environment
conda deactivate
conda activate adlm

cd "/vol/aimspace/projects/practical_SoSe24/mri_inr/code/mri-inr/"

python3 main.py