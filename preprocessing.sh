#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=preprocessing

source /scratch/$USER/.envs/thesis_env/bin/activate

module purge
module load Pillow/9.1.0-GCCcore-10.3.0 
module load Python/3.9.5-GCCcore-10.3.0 
module load SoX/14.4.2-GCCcore-10.3.0
module load CUDA/10.2.89

pip install -r /scratch/s5007453/hifi-gan-laughnet/requirements.txt

python preprocessing.py $1