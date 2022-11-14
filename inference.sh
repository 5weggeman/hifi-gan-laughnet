#!/bin/bash
#SBATCH --time=30:00
#SBATCH --partition=regular
#SBATCH --job-name=inference
#SBATCH --mem=2GB

source /scratch/$USER/.envs/thesis_env/bin/activate

module purge
module load Pillow/9.1.0-GCCcore-10.3.0 
module load Python/3.9.5-GCCcore-10.3.0 
module load SoX/14.4.2-GCCcore-10.3.0
module load CUDA/10.2.89

pip install -r /scratch/s5007453/hifi-gan-laughnet/requirements.txt
pip install pandas

python inference.py --checkpoint_file="/scratch/s5007453/hifi-gan-laughnet/cp_hifigan/g_00200000"