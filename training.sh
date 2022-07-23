#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=training

source /scratch/$USER/.envs/thesis_env/bin/activate

module purge
module load Pillow/9.1.0-GCCcore-10.3.0 
module load Python/3.9.5-GCCcore-10.3.0 
module load SoX/14.4.2-GCCcore-10.3.0
module load CUDA/10.2.89

pip install -r /scratch/s5007453/hifi-gan-laughnet/requirements.txt

python train.py --input_wavs_dir='/scratch/s5007453/hifi-gan-laughnet/VCTK-0.92/wavs/' --input_training_file='/scratch/s5007453/hifi-gan-laughnet/VCTK-0.92/training.txt' --input_validation_file='/scratch/s5007453/hifi-gan-laughnet/VCTK-0.92/validation.txt' --checkpoint_path='/scratch/s5007453/hifi-gan-laughnet/cp_hifigan' --config='config_v1.json' --checkpoint_interval='1500'