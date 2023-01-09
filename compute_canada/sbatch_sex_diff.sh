#!/bin/bash
#SBATCH --account=def-sbramba
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/cbedetti/dataset/HCP/derivatives/training_sex_diff/log/slurm-%x-%j.out

study_dir="/scratch/cbedetti/dataset/HCP/derivatives/training_sex_diff"
script_dir="${study_dir}/code/sex_diff/compute_canada"

module load python/3.10.2
source ${script_dir}/venv/training/bin/activate
python ${script_dir}/sex_diff.py

